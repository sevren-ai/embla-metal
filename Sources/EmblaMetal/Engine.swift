import Foundation
import Metal
import QuartzCore
import Synchronization

/// Embla 1's shape (`embla1`).
public enum Embla {
    public static let d = 2048, layers = 32, headDim = 128, vocab = 65536
    public static let window = 128, topk = 1024, recent = 128, blockSize = 4
    /// Layers of the prefill half: a prompt token whose logits are not needed stops at layer 16's K/V.
    public static let prefillLayers = 16
    public static let experts = 96, expertDim = 512, sharedDim = 1024

    public enum Kind: Sendable { case sliding, full, sparse }
    public static let kinds: [Kind] = (0 ..< 32).map { i in
        i < 16 ? (i == 7 ? .full : .sliding) : ((i - 16) % 4 == 0 ? .full : .sparse)
    }
    /// Full layer -> the layer whose (normed, mixed) input its K/V project.
    public static let kvSources = [7: 7, 16: 16, 20: 16, 24: 7, 28: 7]
    /// K/V source layer -> the later full layers that project its input.
    public static let bridges: [Int: [Int]] = [7: [24, 28], 16: [20]]
    public static func blockFull(_ i: Int) -> Int { (0 ..< i).last { kinds[$0] == .full }! }
    /// Full layers whose keys the sparse layers after them select from.
    public static let selects: Set<Int> = Set((0 ..< 32).filter { i in
        kinds[i] == .full && (i + 1 ..< 32).contains { kinds[$0] == .sparse && blockFull($0) == i }
    })
}

/// One stacked-GEMV segment, as `Seg` in gemv.metal.
struct Seg {
    var w: UInt64, s: UInt64, b: UInt64
    var end: UInt32, x: UInt32, y: UInt32, pad: UInt32 = 0
}

struct KVJob { var k: UInt32, v: UInt32, kc: UInt32, vc: UInt32 }

struct AttnParams {
    var S: Int32, CH: Int32, C: Int32, maxS: Int32, pos: Int32, ng: Int32, pad0: Int32 = 0, pad1: Int32 = 0
}

struct MoEWeights {
    var router: UInt64
    var fc_w: UInt64, fc_s: UInt64, fc_b: UInt64, sh_w: UInt64, sh_s: UInt64, sh_b: UInt64, bias: UInt64
    var pw: UInt64, ps: UInt64, pb: UInt64, spw: UInt64, sps: UInt64, spb: UInt64
}

/// The decode engine: the weights, every activation and cache buffer, and the encoding of one token's
/// step into one command buffer.
public final class Engine: @unchecked Sendable {
    public let ctx: MetalContext
    public let weights: SafeTensors
    public let table: EngramTable
    public let maxContext: Int
    public private(set) var position = 0

    let W: MTLBuffer, nm: MTLBuffer
    let state, x, hs, qkv, o, hmlp, rlogits, route, part, apart, counter, scores, sel: MTLBuffer
    let e, ekv, hist, cache, tokens, greedy, m0, stats: MTLBuffer
    /// Learn each step's token by polling the word the argmax writes for the CPU (greedy[-1]), which
    /// the CPU sees ~60 µs before the command buffer's completion (EMBLA_POLL=0: wait for it).
    /// The other direction, CPU to GPU, goes through `engramReady`: a GPU kernel spinning on memory
    /// the CPU writes does not see the write for about a millisecond.
    public let poll = ProcessInfo.processInfo.environment["EMBLA_POLL"] != "0"
    /// Signaled by the CPU with position + 1 once that position's engram rows are in `e`.
    let engramReady: MTLSharedEvent
    /// engramReady's value before this sequence's first position: a shared event only counts up,
    /// so each sequence continues from where the last one left it (resetting it to 0 did not take
    /// effect on the A20 Pro, and later sequences' steps then read their engram rows before the CPU
    /// had written them).
    var engramBase: UInt64 = 0
    /// Orders the GPU work of decode steps and prefill chunks: every such command buffer waits for
    /// the one before it to signal. One queue did not: the next step's embedding and layer 0 could
    /// start while the previous step was still running.
    let gpuChain: MTLEvent
    var chainValue: UInt64 = 0

    /// A command buffer that starts after the last chained one has finished.
    func chainedCommandBuffer() -> MTLCommandBuffer {
        let cb = ctx.queue.makeCommandBuffer()!
        cb.encodeWaitForEvent(gpuChain, value: chainValue)
        return cb
    }

    /// Ends a chained command buffer's work (after its last encoder): the next may start.
    func endChained(_ cb: MTLCommandBuffer) {
        chainValue += 1
        cb.encodeSignalEvent(gpuChain, value: chainValue)
    }
    var promptEnd = 0
    /// The position being encoded.
    var encPos = 0
    public let logits: MTLBuffer
    private(set) var queries: MTLBuffer!
    /// When set, every mix output of a step is copied here in order (for checking against a reference).
    public var debugMixes: MTLBuffer?
    /// Profiling only: parts of the step to leave out ("mix", "qkv", "attn", "oproj", "router",
    /// "moe", "reduce", "dense", "engram", "lmhead"). The output is then wrong; the time is not.
    public var skip: Set<String> = []
    /// Profiling only: parts of the step to encode twice (after a barrier), so that the step's time
    /// grows by what they cost in place.
    public var twice: Set<String> = []

    var ids: [UInt64] = [UInt64(Embla.vocab), UInt64(Embla.vocab)]

    // ------------------------------------------------------------------ pipelines and plans

    struct Mix {
        let pipeline: MTLComputePipelineState
        let col: Int32, k: Int
    }

    struct Layer {
        let i: Int
        let kind: Embla.Kind
        /// The q/k/v GEMV, which computes its input mixes (columns attnCol ..) itself, and the router
        /// (or layer 0's c_fc), which computes its own (mlpCol).
        let qkvPipe: MTLComputePipelineState, mlpPipe: MTLComputePipelineState
        let attnCol: Int32, attnK: Int, mlpCol: Int32
        /// When the sources change before this layer: prep's pipeline and its columns.
        let prep: (pipe: MTLComputePipelineState, col: Int32, count: Int)?
        let qkv: [Seg], oproj: [Seg], oFirst: Bool
        let qGain: Int, kGain: Int, nGains: Int32
        let kvJobs: [KVJob]
        let kOffset: Int, vOffset: Int   // bytes into `cache`: sliding ring, or the attended full layer
        let dense: (fc: [Seg], proj: [Seg])?
        let moe: MoEWeights?
        /// The q/k/v GEMV reads its input mixes from hs, written by the layer before's MoE reduce
        /// (or by prep at a block start), instead of computing them in every threadgroup.
        var premixed = false
        /// The MoE reduce's pipeline, and the next layer's mix columns when it writes them.
        var reduce: (pipe: MTLComputePipelineState, col: Int32)? = nil
    }

    var layers: [Layer] = []
    var engramMix: Mix!, finalMix: Mix!
    var engramSegs: [Seg] = [], lmHead: [Seg] = []
    var wte: (UInt64, UInt64, UInt64) = (0, 0, 0)
    var convWeight = 0

    let gemvPipes: [MTLComputePipelineState]
    var prep0: (pipe: MTLComputePipelineState, col: Int32, count: Int)!
    var prep1: (pipe: MTLComputePipelineState, col: Int32, count: Int)!
    let pSliding, pKV, pFull, pFullScores, pSparse, pSparseSel, pSparseShort, pSelect: MTLComputePipelineState
    var reduceTGs: Int { Embla.d * reduceGroups / 256 }
    let dots: MTLBuffer
    let pFused, pShared, pReduce, pEmbed, pGate, pArgmax, pSample, pCopy: MTLComputePipelineState
    /// How `generate` picks each token (greedy unless it is told otherwise).
    public private(set) var sampling = Sampling.greedy
    let sampleScratch: MTLBuffer
    /// The shared expert's MoE slices beside the router (they need no routing), the routed ones
    /// after it (EMBLA_SHARED=1).
    static let sharedBeside = ProcessInfo.processInfo.environment["EMBLA_SHARED"] == "1"

    /// Tuning overrides from the environment, e.g. EMBLA_MOE="UNR=8,RPI=2".
    /// Rows a simdgroup takes in the GEMVs, and in the GEMVs that compute their input mixes (a
    /// threadgroup is 8 simdgroups).
    static let gemvRows = envMacros("EMBLA_GEMV")["ROWS"] ?? 1
    static let mixRows = envMacros("EMBLA_GEMV")["MIXROWS"] ?? 2

    /// Positions of engram rows the CPU may read ahead of the GPU (the prompt's are known).
    static let engramRing = 64
    /// The full layers' caches in int8 (common.metal's KV8), EMBLA_KV8=0: bf16.
    public static let kv8 = ProcessInfo.processInfo.environment["EMBLA_KV8"] != "0"
    /// Simdgroups a head in sliding attention (each takes every NS-th of the 128 keys).
    static let slidingSG = envMacros("EMBLA_SLIDING")["NS"] ?? 16
    /// Positions in the sliding layers' caches and the engram's history (RING, HRING in the
    /// kernels): the window or the conv's reach, plus a prefill chunk.
    static let ring = 4096

    public static func envMacros(_ name: String) -> [String: Int] {
        var m: [String: Int] = [:]
        for kv in (ProcessInfo.processInfo.environment[name] ?? "").split(separator: ",") {
            let p = kv.split(separator: "=")
            if p.count == 2, let v = Int(p[1]) { m[String(p[0])] = v }
        }
        return m
    }

    /// A kernel file on the neural accelerators: MLX's NAX tiles (nax.metal), then as kernelSource.
    public static func naxSource(_ name: String) throws -> String {
        try kernelFile("nax") + "\n" + kernelSource(name)
    }

    public static func kernelSource(_ name: String) throws -> String {
        try (kv8 ? "#define KV8 1\n" : "") + kernelFile("common") + "\n" + kernelFile(name)
    }

    /// Kernel work on a device: a directory whose `<name>.metal` files replace the built-in ones
    /// (EMBLA_KERNELS; relative to the app's Documents on iOS), so a kernel can change without a
    /// rebuild. The kernels compile from source at load either way.
    static let kernelOverrides: URL? = {
        guard let dir = ProcessInfo.processInfo.environment["EMBLA_KERNELS"], !dir.isEmpty else { return nil }
        if dir.hasPrefix("/") { return URL(fileURLWithPath: dir) }
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent(dir)
    }()

    static func kernelFile(_ n: String) throws -> String {
        if let dir = kernelOverrides {
            let url = dir.appendingPathComponent(n + ".metal")
            if FileManager.default.fileExists(atPath: url.path) { return try String(contentsOf: url, encoding: .utf8) }
        }
        guard let url = Bundle.module.url(forResource: n, withExtension: "metal", subdirectory: "Kernels") else {
            throw EmblaError.badFile("Kernels/\(n).metal")
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    public init(modelDirectory: URL, maxContext: Int = 8192, ctx: MetalContext? = nil) throws {
        let ctx = try ctx ?? MetalContext()
        self.ctx = ctx
        self.maxContext = maxContext
        weights = try SafeTensors(url: modelDirectory.appendingPathComponent("model.safetensors"), device: ctx.device)
        table = try EngramTable(url: modelDirectory.appendingPathComponent("engram_table.safetensors"))
        W = weights.buffer
        let dev = ctx.device

        // kernels
        let gemv = try Self.kernelSource("gemv"), attn = try Self.kernelSource("attention")
        let moe = try Self.kernelSource("moe"), misc = try Self.kernelSource("misc")
        gemvPipes = try (0 ..< 5).map { try ctx.pipeline("gemv", source: gemv, macros: ["EPI": $0, "ROWS": Self.gemvRows]) }
        pSliding = try ctx.pipeline("attn_sliding", source: attn, macros: ["SCORES": 0, "SEL": 0, "NS": Self.slidingSG])
        pKV = try ctx.pipeline("kv_append", source: attn, macros: ["SCORES": 0, "SEL": 0])
        let fullName = Self.kv8 ? "attn_full8" : "attn_full"
        pFull = try ctx.pipeline(fullName, source: attn, macros: ["SCORES": 0, "SEL": 0])
        pFullScores = try ctx.pipeline(fullName, source: attn, macros: ["SCORES": 1, "SEL": 0])
        let sparseName = Self.kv8 ? "attn_sparse8" : "attn_sparse"
        pSparse = try ctx.pipeline(sparseName, source: attn, macros: ["SCORES": 0, "SEL": 0])
        pSparseShort = try ctx.pipeline("attn_sparse_short", source: attn, macros: ["SCORES": 0, "SEL": 0])
        pSparseSel = try ctx.pipeline(sparseName, source: attn, macros: ["SCORES": 0, "SEL": 1])
        pSelect = try ctx.pipeline("select_keys", source: attn, macros: ["SCORES": 0, "SEL": 0])
        pFused = try ctx.pipeline("moe_fused", source: moe, macros: Self.envMacros("EMBLA_FUSED").merging(["PART": Self.sharedBeside ? 1 : 0]) { a, _ in a })
        pShared = try ctx.pipeline("moe_fused", source: moe, macros: Self.envMacros("EMBLA_FUSED").merging(["PART": 2]) { a, _ in a })
        pReduce = try ctx.pipeline("moe_reduce", source: moe, macros: ["RG": 1])
        dots = buf(64 * 8 * 4)
        pEmbed = try ctx.pipeline("embed", source: misc)
        pGate = try ctx.pipeline("engram_gate", source: misc)
        pArgmax = try ctx.pipeline("argmax", source: misc)
        pSample = try ctx.pipeline("sample", source: misc)
        pCopy = try ctx.pipeline("copy16", source: misc)

        // activations
        func buf(_ bytes: Int) -> MTLBuffer { dev.makeBuffer(length: bytes, options: .storageModeShared)! }
        let d = Embla.d
        state = buf(9 * d * 2)
        x = buf(d * 4)
        hs = buf(3 * d * 2)
        qkv = buf(4096 * 2)
        o = buf(d * 2)
        hmlp = buf(d * 2)
        rlogits = buf(Embla.experts * 4)
        route = buf(16 * 4)
        part = buf(80 * d * 4)
        apart = buf((maxContext / 64 + 2) * 16 * 130 * 4)
        counter = buf(64)
        scores = buf(2 * maxContext * 4)
        sel = buf(2 * (Embla.topk + Embla.recent) * 4)
        e = buf(Self.engramRing * d * 2)   // a ring by position, so the CPU can read ahead
        ekv = buf(2 * d * 2)
        hist = buf(Self.ring * d * 2)
        tokens = buf((maxContext + 2) * 4)
        greedy = buf((maxContext + 3) * 4)
        engramReady = dev.makeSharedEvent()!
        gpuChain = dev.makeEvent()!
        m0 = buf(80 * d * 2)
        stats = buf(80 * 16)
        logits = buf(Embla.vocab * 4)
        sampleScratch = buf(Embla.vocab * 4)

        // caches: 15 sliding rings [4][128][128], then 5 full [2][maxContext][128], k then v each
        // (with KV8 int8 rows, then their [2][maxContext] float scales)
        var cacheBytes = 0
        var slidingOffset: [Int: (Int, Int)] = [:], fullOffset: [Int: (Int, Int)] = [:]
        for i in 0 ..< Embla.layers where Embla.kinds[i] == .sliding {
            slidingOffset[i] = (cacheBytes, cacheBytes + 4 * Self.ring * 128 * 2)
            cacheBytes += 2 * 4 * Self.ring * 128 * 2
        }
        for i in 0 ..< Embla.layers where Embla.kinds[i] == .full {
            let n = Self.kv8 ? 2 * maxContext * (128 + 4) : 2 * maxContext * 128 * 2
            fullOffset[i] = (cacheBytes, cacheBytes + n)
            cacheBytes += 2 * n
        }
        cache = dev.makeBuffer(length: cacheBytes, options: .storageModePrivate)!

        // neuron-major down projections, built on the GPU
        let perLayer = 96 * 512 * 256 * 4 + 2 * 96 * 8 * 2048 * 2 + 1024 * 256 * 4 + 2 * 16 * 2048 * 2
        nm = dev.makeBuffer(length: 31 * perLayer, options: .storageModePrivate)!

        try buildPlans(slidingOffset: slidingOffset, fullOffset: fullOffset, perLayer: perLayer)
        try buildNeuronMajor(perLayer: perLayer, source: misc)
        try buildQueries()
        reset()
        if batchedPrefill { _ = prefiller }   // compile its pipelines now, not in the first prompt
    }

    /// Clears the caches and the position for a new sequence.
    public func reset() {
        position = 0
        ids = [UInt64(Embla.vocab), UInt64(Embla.vocab)]
        memset(hist.contents(), 0, hist.length)
        memset(counter.contents(), 0, counter.length)
        memset(greedy.contents(), 0, 4)
    }

    // ------------------------------------------------------------------ weights

    private func t(_ name: String) throws -> SafeTensors.Tensor { try weights.tensor(name) }

    /// A 4-bit matrix as a segment: rows end at `end`, input at `x`, output at `y`.
    private func seg(_ name: String, end: inout Int, x: Int, y: Int) throws -> Seg {
        let w = try t(name + ".weight"), s = try t(name + ".scales"), b = try t(name + ".biases")
        precondition(w.shape[1] == 256, "\(name): segments take 2048 inputs")
        end += w.shape[0]
        return Seg(w: UInt64(w.offset), s: UInt64(s.offset), b: UInt64(b.offset), end: UInt32(end), x: UInt32(x), y: UInt32(y))
    }

    private func mix(n: Int, k: Int, hasX: Bool, norm: Bool, append: Bool, col: Int) throws -> Mix {
        let p = try ctx.pipeline("mix", source: Self.kernelSource("mix"), macros: [
            "N": n, "K": k, "HAS_X": hasX ? 1 : 0, "NORM": norm ? 1 : 0, "APPEND": append ? 1 : 0,
        ])
        return Mix(pipeline: p, col: Int32(col), k: k)
    }

    /// prep for the columns cols, over n sources (the last one appended from x with `append`).
    private func prep(n: Int, append: Bool, cols: Range<Int>, kout: Int = 0) throws -> (pipe: MTLComputePipelineState, col: Int32, count: Int) {
        let p = try ctx.pipeline("prep", source: Self.kernelSource("mix"), macros: [
            "N": n, "K": 1, "HAS_X": 0, "NORM": 0, "APPEND": append ? 1 : 0, "KOUT": kout,
        ])
        return (p, Int32(cols.lowerBound), cols.count)
    }

    var queryNames: [String] = []

    private func buildPlans(slidingOffset: [Int: (Int, Int)], fullOffset: [Int: (Int, Int)], perLayer: Int) throws {
        // pseudo-query columns, in the reference's order
        var cols: [String: Int] = [:]
        for i in 0 ..< Embla.layers {
            cols["attn\(i)"] = queryNames.count
            queryNames.append("transformer.h.\(i).attn_res_query")
            for j in Embla.bridges[i] ?? [] { queryNames.append("transformer.h.\(j).kv_res_query") }
            cols["mlp\(i)"] = queryNames.count
            queryNames.append("transformer.h.\(i).mlp_res_query")
        }
        cols["engram"] = queryNames.count
        queryNames.append("engram_res_query")
        cols["final"] = queryNames.count
        queryNames.append("final_res_query")

        for i in 0 ..< Embla.layers {
            let kind = Embla.kinds[i], p = "transformer.h.\(i)."
            let blockStart = i % Embla.blockSize == 0
            // sources: the embedding, the engram (after layer 0), each completed block
            let n = i == 0 ? 1 : 2 + i / Embla.blockSize
            let bridged = Embla.bridges[i] ?? []
            let attnK = 1 + bridged.count
            let gemvSource = try Self.kernelSource("gemv")
            let qkvPipe = premix && i >= 2
                ? try ctx.pipeline("gemv", source: gemvSource, macros: ["EPI": 0, "ROWS": Self.gemvRows])
                : try ctx.pipeline("gemv", source: gemvSource, macros: [
                    "EPI": 0, "MIX": 1, "K": attnK, "HAS_X": blockStart ? 0 : 1, "ROWS": Self.mixRows])
            let mlpPipe = i == 0
                ? try ctx.pipeline("gemv", source: gemvSource, macros: ["EPI": 1, "MIX": 1, "K": 1, "HAS_X": 1, "ROWS": Self.mixRows])
                : try ctx.pipeline("moe_router", source: Self.kernelSource("moe"), macros: ["HAS_X": 1, "RSG": routerRows])
            // sources change at each block start (a completed block is appended), and after the
            // engram (layer 1; prep1, encoded after the engram gate)
            let premixed = premix && i >= 2
            let blockPrep = blockStart && i > 0
                ? try prep(n: n, append: true, cols: cols["attn\(i)"]! ..< cols["mlp\(i + 3)"]! + 1, kout: premixed ? attnK : 0) : nil

            var end = 0
            var qkvSegs = [try seg(p + "attn.c_q", end: &end, x: 0, y: 0)]
            var jobs: [KVJob] = []
            var kOff = 0, vOff = 0
            switch kind {
            case .sliding:
                qkvSegs.append(try seg(p + "attn.c_k", end: &end, x: 0, y: 2048))
                qkvSegs.append(try seg(p + "attn.c_v", end: &end, x: 0, y: 2560))
                (kOff, vOff) = slidingOffset[i]!
            case .full:
                if Embla.kvSources[i] == i {
                    for (m, j) in ([i] + bridged).enumerated() {
                        let pj = "transformer.h.\(j)."
                        let ky = m == 0 ? 2048 : 2560 + (m - 1) * 512
                        qkvSegs.append(try seg(pj + "attn.c_k", end: &end, x: m * 2048, y: ky))
                        qkvSegs.append(try seg(pj + "attn.c_v", end: &end, x: m * 2048, y: ky + 256))
                        let (kc, vc) = fullOffset[j]!
                        jobs.append(KVJob(k: UInt32(ky), v: UInt32(ky + 256), kc: UInt32(Self.kv8 ? kc : kc / 2), vc: UInt32(Self.kv8 ? vc : vc / 2)))
                    }
                }
                (kOff, vOff) = fullOffset[i]!
            case .sparse:
                (kOff, vOff) = fullOffset[Embla.blockFull(i)]!
            }
            var oend = 0
            let oproj = [try seg(p + "attn.c_proj", end: &oend, x: 0, y: 0)]

            var dense: ([Seg], [Seg])? = nil
            var moe: MoEWeights? = nil
            if i == 0 {
                var e1 = 0, e2 = 0
                dense = ([try seg(p + "mlp.c_fc", end: &e1, x: 0, y: 0)], [try seg(p + "mlp.c_proj", end: &e2, x: 0, y: 0)])
            } else {
                let base = UInt64((i - 1) * perLayer)
                let pw = base, ps = pw + UInt64(96 * 512 * 256 * 4), pb = ps + UInt64(96 * 8 * 2048 * 2)
                let spw = pb + UInt64(96 * 8 * 2048 * 2), sps = spw + UInt64(1024 * 256 * 4), spb = sps + UInt64(16 * 2048 * 2)
                func off(_ n: String) throws -> UInt64 { UInt64(try t(p + "mlp." + n).offset) }
                moe = MoEWeights(
                    router: try off("router.weight"), fc_w: try off("experts_fc.weight"), fc_s: try off("experts_fc.scales"), fc_b: try off("experts_fc.biases"),
                    sh_w: try off("shared_fc.weight"), sh_s: try off("shared_fc.scales"), sh_b: try off("shared_fc.biases"),
                    bias: try off("expert_bias"), pw: pw, ps: ps, pb: pb, spw: spw, sps: sps, spb: spb)
            }
            let sliding = kind == .sliding
            var layer = Layer(
                i: i, kind: kind, qkvPipe: qkvPipe, mlpPipe: mlpPipe, attnCol: Int32(cols["attn\(i)"]!), attnK: attnK,
                mlpCol: Int32(cols["mlp\(i)"]!), prep: blockPrep, qkv: qkvSegs, oproj: oproj, oFirst: blockStart,
                qGain: try t(p + "attn.q_norm.weight").offset, kGain: try t(p + "attn.k_norm.weight").offset,
                nGains: sliding ? 1 : 2, kvJobs: jobs, kOffset: kOff, vOffset: vOff, dense: dense, moe: moe)
            layer.premixed = premixed
            layers.append(layer)
        }
        // each MoE layer's reduce: plain, or with the next layer's attention mixes (when that layer
        // is premixed and not a block start, whose prep writes them)
        for i in 1 ..< Embla.layers {
            let next = i + 1
            if premix && next < Embla.layers && next >= 2 && next % Embla.blockSize != 0 {
                let k = layers[next].attnK
                layers[i].reduce = (try ctx.pipeline("moe_reduce", source: Self.kernelSource("moe"), macros: ["RG": reduceGroups, "MIXK": k]),
                                    layers[next].attnCol)
            } else {
                layers[i].reduce = (try ctx.pipeline("moe_reduce", source: Self.kernelSource("moe"), macros: ["RG": reduceGroups]), 0)
            }
        }
        prep0 = try prep(n: 1, append: false, cols: cols["attn0"]! ..< cols["mlp0"]! + 1)
        prep1 = try prep(n: 2, append: false, cols: cols["attn1"]! ..< cols["mlp3"]! + 1)
        engramMix = try mix(n: 1, k: 1, hasX: true, norm: false, append: false, col: cols["engram"]!)
        finalMix = try mix(n: 9, k: 1, hasX: true, norm: true, append: false, col: cols["final"]!)
        var end = 0
        engramSegs = [try seg("engram.w_k", end: &end, x: 0, y: 0), try seg("engram.w_v", end: &end, x: 0, y: 2048)]
        end = 0
        lmHead = [try seg("lm_head", end: &end, x: 0, y: 0)]
        let w = try t("transformer.wte.weight"), s = try t("transformer.wte.scales"), b = try t("transformer.wte.biases")
        wte = (UInt64(w.offset), UInt64(s.offset), UInt64(b.offset))
        convWeight = try t("engram.conv_weight").offset
    }

    private func buildQueries() throws {
        let d = Embla.d
        queries = ctx.buffer(Float.self, count: queryNames.count * d)
        let out = queries.contents().assumingMemoryBound(to: Float.self)
        for (q, name) in queryNames.enumerated() {
            let src = (W.contents() + (try t(name)).offset).assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< d { out[q * d + i] = BF16.float(src[i]) }
        }
    }

    private func buildNeuronMajor(perLayer: Int, source: String) throws {
        let pT = try ctx.pipeline("nibble_transpose", source: source), p16 = try ctx.pipeline("transpose16", source: source)
        for i in 1 ..< Embla.layers {
            let cb = ctx.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
            let m = layers[i].moe!
            let p = "transformer.h.\(i).mlp."
            func words(_ name: String, to: UInt64, e: Int, r: Int, c: Int) throws {
                enc.setComputePipelineState(pT)
                enc.setBuffer(W, offset: try t(name).offset, index: 0)
                enc.setBuffer(nm, offset: Int(to), index: 1)
                var rc = SIMD2<UInt32>(UInt32(r), UInt32(c))
                enc.setBytes(&rc, length: 8, index: 2)
                enc.dispatchThreads(MTLSize(width: r / 8, height: c, depth: e), threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
            }
            func groups(_ name: String, to: UInt64, e: Int, r: Int, g: Int) throws {
                enc.setComputePipelineState(p16)
                enc.setBuffer(W, offset: try t(name).offset, index: 0)
                enc.setBuffer(nm, offset: Int(to), index: 1)
                var rg = SIMD2<UInt32>(UInt32(r), UInt32(g))
                enc.setBytes(&rg, length: 8, index: 2)
                enc.dispatchThreads(MTLSize(width: r, height: g, depth: e), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
            try words(p + "experts_proj.weight", to: m.pw, e: 96, r: 2048, c: 512)
            try groups(p + "experts_proj.scales", to: m.ps, e: 96, r: 2048, g: 8)
            try groups(p + "experts_proj.biases", to: m.pb, e: 96, r: 2048, g: 8)
            try words(p + "shared_proj.weight", to: m.spw, e: 1, r: 2048, c: 1024)
            try groups(p + "shared_proj.scales", to: m.sps, e: 1, r: 2048, g: 16)
            try groups(p + "shared_proj.biases", to: m.spb, e: 1, r: 2048, g: 16)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
    }

    // ------------------------------------------------------------------ profiling

    /// GPU µs per layer of the fused MoE alone (with `macros` for moe.metal), over the 31 MoE
    /// layers on the current buffers (run a few steps first). Best of `reps`.
    /// GPU µs per call of the o projection GEMV chained over the 32 layers (weights from DRAM), and of
    /// layer 0's alone (cached after the first call); `macros` for gemv.metal.
    public func timeGEMV(macros: [String: Int], reps: Int = 5) throws -> (dram: Double, cached: Double) {
        let p = try ctx.pipeline("gemv", source: Self.kernelSource("gemv"), macros: macros.merging(["EPI": 0]) { a, _ in a })
        let rowsPerTG = 8 * (macros["NSG"] ?? 8) * (macros["ROWS"] ?? 4) / 8
        func run(_ pick: (Int) -> Layer) -> Double {
            var best = Double.infinity
            for _ in 0 ..< reps {
                let cb = ctx.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
                for i in 0 ..< 64 {
                    let segs = pick(i).oproj
                    enc.setComputePipelineState(p)
                    enc.setBuffer(W, offset: 0, index: 0)
                    enc.setBuffer(o, offset: 0, index: 1)
                    enc.setBuffer(qkv, offset: 0, index: 2)
                    segs.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 3) }
                    var n = Int32(1)
                    enc.setBytes(&n, length: 4, index: 4)
                    enc.dispatchThreadgroups(MTLSize(width: 2048 / rowsPerTG, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32 * (macros["NSG"] ?? 8), height: 1, depth: 1))
                    enc.memoryBarrier(scope: .buffers)
                }
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / 64 * 1e6)
            }
            return best
        }
        return (run { self.layers[$0 % 32] }, run { _ in self.layers[0] })
    }

    // ------------------------------------------------------------------ encoding

    /// The decode step's encoders: serial (each dispatch after the one before, no barriers) or
    /// concurrent with a barrier between dependent dispatches (EMBLA_SERIAL=0).
    static let serial = ProcessInfo.processInfo.environment["EMBLA_SERIAL"] == "1"
    /// While set, every dependent group of the step's dispatches is an encoder of its own, timed.
    var profiler: Profiler?

    /// Profiling: GPU µs each part of a decode step takes in the real step (every dependent group of
    /// dispatches its own timestamped encoder, so the step itself runs slower), averaged over `n`
    /// steps at the current position, and the time between encoders.
    public func profileSteps(_ n: Int) throws -> (parts: [(String, Double)], gaps: Double, total: Double) {
        let p = try Profiler(device: ctx.device)
        profiler = p
        defer { profiler = nil }
        let (c0, g0) = ctx.device.sampleTimestamps()
        let pos = position
        var last: MTLCommandBuffer?
        for _ in 0 ..< n {
            // one step at a time: with several sampled command buffers in flight only the first
            // got its timestamps
            engramReady.signaledValue = max(engramReady.signaledValue, engramBase + UInt64(pos + 1))
            let c = encodeStep(pos: pos)
            c.0.commit(); c.1.commit()
            c.1.waitUntilCompleted()
            last = c.1
        }
        last?.waitUntilCompleted()
        let (c1, g1) = ctx.device.sampleTimestamps()
        let ratio = g1 > g0 ? Double(c1 - c0) / Double(g1 - g0) : 1
        return p.report(steps: n, ticksToNs: ratio)
    }

    private func barrier(_ enc: StepEncoder) { enc.barrier() }

    /// With `mix`, the GEMV (that pipeline) computes its input mixes from column `col` itself, into X.
    private func gemv(_ enc: StepEncoder, _ segs: [Seg], epi: Int, x: MTLBuffer, y: MTLBuffer, part: String,
                      mix: MTLComputePipelineState? = nil, col: Int32 = 0, plain: MTLComputePipelineState? = nil) {
        if skip.contains(part) { return }
        enc.setComputePipelineState(mix ?? plain ?? gemvPipes[epi])
        enc.setBuffer(W, offset: 0, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        if mix != nil {
            enc.setBuffer(queries, offset: 0, index: 7)
            enc.setBuffer(m0, offset: 0, index: 8)
            enc.setBuffer(stats, offset: 0, index: 9)
            enc.setBuffer(self.x, offset: 0, index: 10)
            var c = col
            enc.setBytes(&c, length: 4, index: 11)
        }
        segs.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        var n = Int32(segs.count)
        enc.setBytes(&n, length: 4, index: 4)
        let rows = Int(segs.last!.end), perTG = 8 * (mix == nil ? Self.gemvRows : Self.mixRows)
        dispatch(enc, part, MTLSize(width: (rows + perTG - 1) / perTG, height: 1, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Dispatches, and again after a barrier if `part` is in `twice`.
    private func dispatch(_ enc: StepEncoder, _ part: String, _ groups: MTLSize, _ threads: MTLSize) {
        enc.label = part
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        if twice.contains(part) {
            barrier(enc)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        }
    }

    var debugCount = 0

    private func encodeMix(_ enc: StepEncoder, _ m: Mix) {
        enc.setComputePipelineState(m.pipeline)
        enc.setBuffer(state, offset: 0, index: 0)
        enc.setBuffer(queries, offset: 0, index: 1)
        enc.setBuffer(x, offset: 0, index: 2)
        enc.setBuffer(hs, offset: 0, index: 3)
        var col = m.col
        enc.setBytes(&col, length: 4, index: 4)
        dispatch(enc, "mix", MTLSize(width: 1, height: 1, depth: 1), MTLSize(width: 1024, height: 1, depth: 1))
        debugCopy(enc, m.k)
    }

    private func encodePrep(_ enc: StepEncoder, _ p: (pipe: MTLComputePipelineState, col: Int32, count: Int)) {
        enc.setComputePipelineState(p.pipe)
        enc.setBuffer(state, offset: 0, index: 0)
        enc.setBuffer(queries, offset: 0, index: 1)
        enc.setBuffer(x, offset: 0, index: 2)
        enc.setBuffer(m0, offset: 0, index: 3)
        enc.setBuffer(stats, offset: 0, index: 4)
        var c = p.col
        enc.setBytes(&c, length: 4, index: 5)
        enc.setBuffer(hs, offset: 0, index: 6)
        dispatch(enc, "prep", MTLSize(width: p.count, height: 1, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
        barrier(enc)
    }

    /// With `debugMixes`: copies the k mixes just written to `hs` there.
    private func debugCopy(_ enc: StepEncoder, _ k: Int) {
        guard let dbg = debugMixes else { return }
        barrier(enc)
        enc.setComputePipelineState(pCopy)
        enc.setBuffer(hs, offset: 0, index: 0)
        enc.setBuffer(dbg, offset: debugCount * Embla.d * 2, index: 1)
        enc.dispatchThreads(MTLSize(width: k * Embla.d, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        debugCount += k
    }

    /// Chunking of `keys` keys for the split attention kernels.
    private func chunks(_ keys: Int) -> (ch: Int, c: Int) {
        let ch = keys <= 2048 ? 64 : keys <= 8192 ? 128 : keys <= 16384 ? 256 : 512
        return (ch, (keys + ch - 1) / ch)
    }

    /// Threadgroups a kv head in the full layers' attention (EMBLA_CHUNKS=TG=..): 32 on the M5 Pro,
    /// 8 on the iPhone 18 Pro's 6 cores (full attention 126 -> 93 µs a token at 1.1k keys).
    lazy var fullTGs = Self.envMacros("EMBLA_CHUNKS")["TG"] ?? (Self.isPhone(ctx.device) ? 8 : 32)

    /// Router rows a threadgroup (EMBLA_ROUTER=RSG=..; 96 / RSG threadgroups).
    lazy var routerRows = Self.envMacros("EMBLA_ROUTER")["RSG"] ?? 8

    /// The attention inputs' mixes computed once a layer (in the MoE reduce, or prep at a block
    /// start) rather than in each q/k/v threadgroup: decode 6.24 -> 6.05 ms a token on the iPhone
    /// 18 Pro, 3.16 -> 3.08 on the M5 Pro (1k keys; the same at 16k). EMBLA_PREMIX=0 turns it off.
    lazy var premix = ProcessInfo.processInfo.environment["EMBLA_PREMIX"] != "0"
    /// Groups of MoE slices a column's sum is split over in the reduce (EMBLA_REDUCE=RG=..): 2 on
    /// iPhones.
    lazy var reduceGroups = Self.envMacros("EMBLA_REDUCE")["RG"] ?? (Self.isPhone(ctx.device) ? 2 : 1)

    /// An iPhone's GPU (an A-series chip), which gets its own launch settings.
    static func isPhone(_ device: MTLDevice) -> Bool { device.name.hasPrefix("Apple A") }

    /// Keys a chunk in the sparse layers' attention (EMBLA_CHUNKS=SP=..): 64 on the M5 Pro, 128 on
    /// the iPhone 18 Pro (sparse attention 265 -> 192 µs a token at 1.1k keys, the same output).
    lazy var sparseCH = Self.envMacros("EMBLA_CHUNKS")["SP"] ?? (Self.isPhone(ctx.device) ? 128 : 64)
    /// Up to this many keys (and no selection) the sparse layers attend a threadgroup per query head.
    static let shortKeys = envMacros("EMBLA_CHUNKS")["SHORT"] ?? 512

    /// attn_sparse8's chunking: sparseCH keys (at most 256).
    private func sparseChunks(_ keys: Int) -> (ch: Int, c: Int) {
        (sparseCH, (keys + sparseCH - 1) / sparseCH)
    }

    /// attn_full8's chunking: about fullTGs threadgroups a kv head, at least 64 keys each.
    private func chunks8(_ keys: Int) -> (ch: Int, c: Int) {
        let ch = max(64, (keys / fullTGs + 31) / 32 * 32)
        return (ch, (keys + ch - 1) / ch)
    }

    /// With `kvOnly`, only the layer's q/k/v GEMV and its k/v append (a prompt token's last layer).
    private func encodeAttention(_ enc: StepEncoder, _ L: Layer, pos: Int, kvOnly: Bool = false) {
        if let p = L.prep { encodePrep(enc, p) }
        if L.premixed {
            gemv(enc, L.qkv, epi: 0, x: hs, y: qkv, part: "qkv", plain: L.qkvPipe)
        } else {
            gemv(enc, L.qkv, epi: 0, x: hs, y: qkv, part: "qkv", mix: L.qkvPipe, col: L.attnCol)
        }
        debugCopy(enc, L.attnK)
        barrier(enc)
        if kvOnly {
            if !L.kvJobs.isEmpty {
                enc.label = "kv"
                enc.label = "kv"
            enc.setComputePipelineState(pKV)
                enc.setBuffer(qkv, offset: 0, index: 0)
                enc.setBuffer(cache, offset: 0, index: 1)
                L.kvJobs.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
                var pm = SIMD2<Int32>(Int32(pos), Int32(maxContext))
                enc.setBytes(&pm, length: 8, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: 2 * L.kvJobs.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                barrier(enc)
            }
            return
        }
        var p = Int32(pos)
        let skipKind = skip.contains("attn") || skip.contains(L.kind == .sliding ? "sliding" : L.kind == .full ? "full" : "sparse")
        switch skipKind ? nil : L.kind {
        case nil: break
        case .sliding:
            enc.setComputePipelineState(pSliding)
            enc.setBuffer(qkv, offset: 0, index: 0)
            enc.setBuffer(W, offset: L.qGain, index: 1)
            enc.setBuffer(W, offset: L.kGain, index: 2)
            enc.setBuffer(cache, offset: L.kOffset, index: 3)
            enc.setBuffer(cache, offset: L.vOffset, index: 4)
            enc.setBuffer(o, offset: 0, index: 5)
            enc.setBytes(&p, length: 4, index: 6)
            dispatch(enc, "attn", MTLSize(width: 16, height: 1, depth: 1), MTLSize(width: 32 * Self.slidingSG, height: 1, depth: 1))
        case .full, .sparse:
            if !L.kvJobs.isEmpty {
                enc.label = "kv"
                enc.label = "kv"
            enc.setComputePipelineState(pKV)
                enc.setBuffer(qkv, offset: 0, index: 0)
                enc.setBuffer(cache, offset: 0, index: 1)
                L.kvJobs.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
                var pm = SIMD2<Int32>(Int32(pos), Int32(maxContext))
                enc.setBytes(&pm, length: 8, index: 3)
                enc.dispatchThreadgroups(MTLSize(width: 2 * L.kvJobs.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                barrier(enc)
            }
            // past topk + recent keys, the full layers before sparse ones score every key and select,
            // and the sparse layers attend the selected rows only
            let keys = pos + 1, selecting = keys > Embla.topk + Embla.recent
            let scoring = selecting && L.kind == .full && Embla.selects.contains(L.i)
            let rows = selecting && L.kind == .sparse ? Embla.topk + Embla.recent : keys
            let short = Self.kv8 && L.kind == .sparse && !selecting && rows <= Self.shortKeys
            let (ch, c) = !Self.kv8 ? chunks(rows) : L.kind == .full ? chunks8(rows) : sparseChunks(rows)
            var ap = AttnParams(S: Int32(rows), CH: Int32(ch), C: Int32(c), maxS: Int32(maxContext), pos: Int32(pos), ng: L.nGains)
            enc.setComputePipelineState(L.kind == .full ? (scoring ? pFullScores : pFull) : short ? pSparseShort : (selecting ? pSparseSel : pSparse))
            enc.setBuffer(qkv, offset: 0, index: 0)
            enc.setBuffer(W, offset: L.qGain, index: 1)
            enc.setBuffer(W, offset: L.kGain, index: 2)
            enc.setBuffer(cache, offset: L.kOffset, index: 3)
            enc.setBuffer(cache, offset: L.vOffset, index: 4)
            enc.setBuffer(apart, offset: 0, index: 5)
            enc.setBuffer(L.kind == .full ? scores : sel, offset: 0, index: 6)
            enc.setBuffer(counter, offset: 0, index: 7)
            enc.setBuffer(o, offset: 0, index: 8)
            enc.setBytes(&ap, length: MemoryLayout<AttnParams>.size, index: 9)
            if short {   // a threadgroup per query head (attn_sparse_short)
                dispatch(enc, "sparse", MTLSize(width: 16, height: 1, depth: 1), MTLSize(width: 512, height: 1, depth: 1))
            } else {
                let threads = Self.kv8 ? 128 : 256
                dispatch(enc, L.kind == .full ? "full" : "sparse", MTLSize(width: c, height: 2, depth: 1), MTLSize(width: threads, height: 1, depth: 1))
            }
            if scoring {
                barrier(enc)
                enc.setComputePipelineState(pSelect)
                enc.setBuffer(scores, offset: 0, index: 0)
                enc.setBuffer(sel, offset: 0, index: 1)
                var sm = SIMD2<Int32>(Int32(keys), Int32(maxContext))
                enc.setBytes(&sm, length: 8, index: 2)
                dispatch(enc, "select", MTLSize(width: 2, height: 1, depth: 1), MTLSize(width: 1024, height: 1, depth: 1))
            }
        }
        barrier(enc)
        gemv(enc, L.oproj, epi: L.oFirst ? 3 : 2, x: o, y: x, part: "oproj")
        barrier(enc)
    }

    private func setMoEBuffers(_ enc: StepEncoder, _ mw: inout MoEWeights, col: Int32) {
        enc.setBuffer(W, offset: 0, index: 0)
        enc.setBuffer(nm, offset: 0, index: 1)
        enc.setBuffer(hs, offset: 0, index: 2)
        enc.setBuffer(route, offset: 0, index: 3)
        enc.setBuffer(part, offset: 0, index: 4)
        enc.setBytes(&mw, length: MemoryLayout<MoEWeights>.size, index: 5)
        enc.setBuffer(queries, offset: 0, index: 6)
        enc.setBuffer(m0, offset: 0, index: 7)
        enc.setBuffer(stats, offset: 0, index: 8)
        enc.setBuffer(x, offset: 0, index: 9)
        var c = col
        enc.setBytes(&c, length: 4, index: 10)
    }

    private func encodeMLP(_ enc: StepEncoder, _ L: Layer) {
        if let (fc, proj) = L.dense {
            gemv(enc, fc, epi: 1, x: hs, y: hmlp, part: "dense", mix: L.mlpPipe, col: L.mlpCol)
            debugCopy(enc, 1)
            barrier(enc)
            gemv(enc, proj, epi: 2, x: hmlp, y: x, part: "dense")
            barrier(enc)
            return
        }
        var mw = L.moe!
        if !skip.contains("router") {
            enc.setComputePipelineState(L.mlpPipe)
            enc.setBuffer(W, offset: 0, index: 0)
            enc.setBytes(&mw, length: MemoryLayout<MoEWeights>.size, index: 1)
            enc.setBuffer(queries, offset: 0, index: 2)
            enc.setBuffer(m0, offset: 0, index: 3)
            enc.setBuffer(stats, offset: 0, index: 4)
            enc.setBuffer(x, offset: 0, index: 5)
            enc.setBuffer(hs, offset: 0, index: 6)
            enc.setBuffer(rlogits, offset: 0, index: 7)
            enc.setBuffer(counter, offset: 16, index: 8)
            enc.setBuffer(route, offset: 0, index: 9)
            var col = L.mlpCol
            enc.setBytes(&col, length: 4, index: 10)
            dispatch(enc, "router", MTLSize(width: Embla.experts / routerRows, height: 1, depth: 1), MTLSize(width: 32 * routerRows, height: 1, depth: 1))
            debugCopy(enc, 1)
            if Self.sharedBeside && !skip.contains("moe") {   // no barrier: beside the router
                enc.setComputePipelineState(pShared)
                setMoEBuffers(enc, &mw, col: L.mlpCol)
                dispatch(enc, "moe", MTLSize(width: Embla.sharedDim / 64, height: 1, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
            }
            barrier(enc)
        }
        if !skip.contains("moe") {
            enc.setComputePipelineState(pFused)
            setMoEBuffers(enc, &mw, col: L.mlpCol)
            let slices = Self.sharedBeside ? 8 * Embla.expertDim / 64 : 80
            dispatch(enc, "moe", MTLSize(width: slices, height: 1, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
            barrier(enc)
        }
        if !skip.contains("reduce") {
            let r = L.reduce!
            enc.setComputePipelineState(r.pipe)
            enc.setBuffer(part, offset: 0, index: 0)
            enc.setBuffer(x, offset: 0, index: 1)
            enc.setBuffer(queries, offset: 0, index: 2)
            enc.setBuffer(m0, offset: 0, index: 3)
            enc.setBuffer(stats, offset: 0, index: 4)
            var c = r.col
            enc.setBytes(&c, length: 4, index: 5)
            enc.setBuffer(counter, offset: 32, index: 6)
            enc.setBuffer(dots, offset: 0, index: 7)
            enc.setBuffer(hs, offset: 0, index: 8)
            dispatch(enc, "reduce", MTLSize(width: reduceTGs, height: 1, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
            barrier(enc)
        }
    }

    private func encodeEngram(_ enc: StepEncoder, pos: Int) {
        if skip.contains("engram") { return }
        encodeMix(enc, engramMix)
        gemv(enc, engramSegs.map { var s = $0; s.x += UInt32(pos % Self.engramRing * Embla.d); return s }, epi: 0, x: e, y: ekv, part: "engram")
        barrier(enc)
        enc.label = "engram"
        enc.setComputePipelineState(pGate)
        enc.setBuffer(ekv, offset: 0, index: 0)
        enc.setBuffer(hs, offset: 0, index: 1)
        enc.setBuffer(W, offset: convWeight, index: 2)
        enc.setBuffer(hist, offset: 0, index: 3)
        enc.setBuffer(state, offset: 0, index: 4)
        var p = Int32(pos)
        enc.setBytes(&p, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        barrier(enc)
        encodePrep(enc, prep1)
    }

    /// Encodes the step of position `pos` as two command buffers: the embedding and layer 0, which
    /// need only the token (in `tokens`, written by the CPU for the prompt and by the previous step's
    /// argmax after it), then the rest, which waits for the CPU to signal the engram rows.
    ///
    /// With `prompt` (a prompt token whose logits nobody reads) the step stops at layer 16's q/k/v:
    /// every key and value the decode half will read projects a mix of layers 0 to 16 (layers 20,
    /// 24 and 28 take theirs from layers 16 and 7), so layers 16 to 31 run only for the last token.
    private func encodeStep(pos: Int, prompt: Bool = false) -> (MTLCommandBuffer, MTLCommandBuffer) {
        encPos = pos
        debugCount = 0
        let first = chainedCommandBuffer()
        var enc = StepEncoder(first, serial: Self.serial, profiler: profiler)
        enc.label = "embed"
        enc.setComputePipelineState(pEmbed)
        enc.setBuffer(W, offset: 0, index: 0)
        var off = SIMD3<UInt64>(wte.0, wte.1, wte.2)
        enc.setBytes(&off, length: 32, index: 1)
        enc.setBuffer(tokens, offset: 0, index: 2)
        enc.setBuffer(state, offset: 0, index: 3)
        var p = Int32(pos)
        enc.setBytes(&p, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        barrier(enc)
        encodePrep(enc, prep0)
        encodeAttention(enc, layers[0], pos: pos)
        encodeMLP(enc, layers[0])

        enc.endEncoding()
        endChained(first)
        let rest = chainedCommandBuffer()
        rest.encodeWaitForEvent(engramReady, value: engramBase + UInt64(pos + 1))
        enc = StepEncoder(rest, serial: Self.serial, profiler: profiler)
        encodeEngram(enc, pos: pos)
        for L in layers.dropFirst() {
            if prompt && L.i == Embla.prefillLayers {
                encodeAttention(enc, L, pos: pos, kvOnly: true)
                enc.endEncoding()
                endChained(rest)
                return (first, rest)
            }
            encodeAttention(enc, L, pos: pos)
            encodeMLP(enc, L)
        }
        encodeMix(enc, finalMix)
        barrier(enc)
        gemv(enc, lmHead, epi: 4, x: hs, y: logits, part: "lmhead")
        barrier(enc)
        enc.label = "argmax"
        enc.setComputePipelineState(sampling.isGreedy ? pArgmax : pSample)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(tokens, offset: 0, index: 1)
        var v = Int32(Embla.vocab)
        enc.setBytes(&v, length: 4, index: 2)
        enc.setBuffer(greedy, offset: 4, index: 3)
        var pp = SIMD2<Int32>(Int32(pos), Int32(promptEnd))
        enc.setBytes(&pp, length: 8, index: 4)
        if !sampling.isGreedy {
            var sp = (1 / sampling.temperature, sampling.topP, sampling.seed)
            enc.setBytes(&sp, length: 12, index: 5)
            enc.setBuffer(sampleScratch, offset: 0, index: 6)
        }
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        enc.endEncoding()
        endChained(rest)
        return (first, rest)
    }

    private func token(at p: Int) -> Int { Int(tokens.contents().assumingMemoryBound(to: UInt32.self)[p]) }

    /// The sequence's tokens as the CPU knows them (the prompt's, then each step's from the word it
    /// polls): what the engram rows are read for. The GPU's copy in `tokens`, read before its step
    /// completes, could still hold an old value, and the rows were then another token's.
    private var known: [Int] = []
    private func knownToken(_ q: Int) -> UInt64 { q < 0 ? UInt64(Embla.vocab) : UInt64(known[q]) }

    /// CPU seconds spent reading engram rows, and waiting for steps, in `generate` (profiling).
    public private(set) var engramSeconds = 0.0, waitSeconds = 0.0
    /// Seconds from the start of `generate` to the first new token (the prompt).
    public private(set) var prefillSeconds = 0.0
    /// Prefill in batches (the prompt's matrices as GEMMs); EMBLA_BATCH=0 runs the prompt as decode steps.
    public var batchedPrefill = ProcessInfo.processInfo.environment["EMBLA_BATCH"] != "0"
    lazy var prefiller: Prefill = try! Prefill(self)
    /// Profiling only: prefill dispatch categories to encode twice (see Prefill.twice).
    public var prefillTwice: Set<String> { get { prefiller.twice } set { prefiller.twice = newValue } }
    /// The last batched prefill: GPU seconds, and CPU seconds reading engram rows.
    public private(set) var prefillGPU = 0.0, prefillRead = 0.0

    /// Signaled by the CPU once a prefill chunk's engram rows are in the prefill's `e` (a running count).
    lazy var prefillRows: MTLSharedEvent = ctx.device.makeSharedEvent()!
    var prefillRowsValue: UInt64 = 0

    /// Batched prefill of positions 0 ..< n (their tokens in `tokens`): chunks of up to
    /// `Prefill.chunk` tokens, each submitted before its engram rows are read. A chunk's command
    /// buffer waits for them only after layer 0 (an event the CPU signals), so the reads overlap
    /// the embedding and layer 0, and no chunk has to be short for them: a short chunk costs the
    /// routed experts nearly as much as a full one (a tile for almost every expert).
    /// Returns the chunks' command buffers, committed and not waited for: the GPU runs the
    /// prompt's last step after them (one queue), so the caller encodes it meanwhile. It sets
    /// `prefillGPU` with `finishPrefill` once they are done.
    /// Profiling: EMBLA_TRACE=1 prints when the prompt's parts happen, ms from the prompt's start.
    let trace = ProcessInfo.processInfo.environment["EMBLA_TRACE"] == "1"
    var traceStart: Double?
    func mark(_ what: String) {
        if trace, let t0 = traceStart { print(String(format: "trace: %@ %.2f ms", what, (CACurrentMediaTime() - t0) * 1e3)) }
    }

    func prefill(upTo n: Int) -> [MTLCommandBuffer] {
        let pf = prefiller, T = Prefill.chunk
        prefillGPU = 0
        prefillRead = 0
        let id = { (q: Int) -> UInt64 in self.knownToken(q) }
        let e = pf.e.contents().assumingMemoryBound(to: UInt16.self)
        // the engram rows of positions p0 ..< p1 into half k of e
        func read(_ p0: Int, _ k: Int, _ p1: Int) {
            let r0 = Date()
            DispatchQueue.concurrentPerform(iterations: p1 - p0) { j in
                let q = p0 + j
                self.table.read(rows: self.table.rows([id(q - 2), id(q - 1), id(q)]), into: e + (k * T + j) * Embla.d)
            }
            prefillRead += Date().timeIntervalSince(r0)
        }
        let starts = Array(stride(from: 0, to: n, by: T)) + [n]
        var submitted: [MTLCommandBuffer] = []
        for c in 0 ..< starts.count - 1 {
            // half c % 2 of e was chunk c - 2's
            if c >= 2 { submitted[c - 2].waitUntilCompleted() }
            prefillRowsValue += 1
            mark("chunk \(c) encode")
            let cb = pf.encode(p0: starts[c], tokens: starts[c + 1] - starts[c], half: c % 2, wait: (prefillRows, prefillRowsValue))
            cb.commit()
            mark("chunk \(c) committed")
            submitted.append(cb)
            read(starts[c], c % 2, starts[c + 1])
            prefillRows.signaledValue = prefillRowsValue
            mark("chunk \(c) rows signaled")
        }
        return submitted
    }

    /// Waits for the prefill's command buffers; `prefillGPU` is their GPU time (the GPU's wait for
    /// the engram rows included).
    func finishPrefill(_ cbs: [MTLCommandBuffer]) {
        for cb in cbs {
            cb.waitUntilCompleted()
            if let err = cb.error { fatalError("prefill failed: \(err)") }
        }
        if trace, let t0 = traceStart {
            for (i, cb) in cbs.enumerated() {
                print(String(format: "trace: prefill cb %d GPU %.2f .. %.2f ms", i, (cb.gpuStartTime - t0) * 1e3, (cb.gpuEndTime - t0) * 1e3))
            }
        }
        if let a = cbs.first, let b = cbs.last { prefillGPU = b.gpuEndTime - a.gpuStartTime }
    }

    /// Run every prompt token through the whole model (EMBLA_FULL_PROMPT=1; for comparison).
    public var fullPrompt = ProcessInfo.processInfo.environment["EMBLA_FULL_PROMPT"] == "1"
    /// Medians from the last `generate`, µs: first command buffer, gap to the second, second, gap to the next step.
    public private(set) var gaps = (0.0, 0.0, 0.0, 0.0)
    /// Medians from the last `generate`, µs: step end to the CPU waking, the CPU's engram read and
    /// signal, the signal to the GPU starting the second buffer.
    public private(set) var latency = (0.0, 0.0, 0.0)

    /// Reads position p's engram rows into e[p % engramRing] and signals them.
    private func readEngram(_ p: Int) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        defer { engramSeconds += Double(DispatchTime.now().uptimeNanoseconds - t0) * 1e-9 }
        table.read(rows: table.rows([knownToken(p - 2), knownToken(p - 1), knownToken(p)]), into: e.contents().assumingMemoryBound(to: UInt16.self) + p % Self.engramRing * Embla.d)
        engramReady.signaledValue = engramBase + UInt64(p + 1)
    }

    /// Waits for step p's token: polls the word its argmax writes for the CPU, or waits for the
    /// command buffer.
    /// Returns the step's token: from the polled word, which carries it ((pos + 1) << 16 | token),
    /// or from `tokens` once the step has completed.
    private func awaitStep(_ p: Int, _ cb: MTLCommandBuffer) -> Int {
        if poll {
            let done = UnsafeMutableRawPointer(greedy.contents()).assumingMemoryBound(to: UInt32.self)
            let want = UInt32((p + 1) & 0xFFFF)
            while cb.status != .completed && cb.status != .error {
                let v = loadAcquire(done)
                if v >> 16 == want { return Int(v & 0xFFFF) }
            }
        }
        cb.waitUntilCompleted()
        return token(at: p + 1)
    }

    /// Checks the last step's key selection (layer 28's) against a CPU sort: per kv head, distinct
    /// rows, the top-k older keys by score, then the recent ones. Returns a failure, or nil.
    public func verifySelection() -> String? {
        let keys = position, k = Embla.topk, r = Embla.recent
        guard keys > k + r else { return "context too short to select" }
        let sc = scores.contents().assumingMemoryBound(to: Float.self)
        let ix = sel.contents().assumingMemoryBound(to: UInt32.self)
        for h in 0 ..< 2 {
            let n = keys - r
            let row = (0 ..< k + r).map { Int(ix[h * (k + r) + $0]) }
            if Set(row).count != k + r { return "head \(h): repeated rows" }
            if Array(row[k...]) != Array(n ..< keys) { return "head \(h): recent rows" }
            let s = (0 ..< n).map { sc[h * maxContext + $0] }
            let kth = s.sorted(by: >)[k - 1]
            if row[..<k].contains(where: { $0 >= n || s[$0] < kth }) { return "head \(h): a selected key below the top \(k)" }
            if s.filter({ $0 > kth }).count > k { return "head \(h): count" }
        }
        return nil
    }

    /// Profiling only: GPU ms per step for `n` steps at the current position submitted back to back
    /// (the GPU never idles between them), against waiting for each (the state is not advanced).
    public func timeBackToBack(_ n: Int) -> (backToBack: Double, oneAtATime: Double) {
        let pos = position
        var cbs: [(MTLCommandBuffer, MTLCommandBuffer)] = []
        for _ in 0 ..< n {
            let c = encodeStep(pos: pos)
            c.0.commit(); c.1.commit()
            cbs.append(c)
        }
        cbs.last!.1.waitUntilCompleted()
        let b2b = cbs.map { $0.1.gpuEndTime - $0.0.gpuStartTime }.sorted()[n / 2]
        var single: [Double] = []
        for _ in 0 ..< n {
            let c = encodeStep(pos: pos)
            c.0.commit(); c.1.commit()
            c.1.waitUntilCompleted()
            usleep(300)
            single.append(c.1.gpuEndTime - c.0.gpuStartTime)
        }
        return (b2b * 1e3, single.sorted()[n / 2] * 1e3)
    }

    /// Runs the step of `tokenID` at the current position, waiting for it; returns the greedy next
    /// token and the step's GPU time.
    @discardableResult
    public func step(_ tokenID: Int) -> (next: Int, gpuSeconds: Double) {
        precondition(position < maxContext, "context full")
        if position == 0 { engramBase = engramReady.signaledValue }
        tokens.contents().assumingMemoryBound(to: UInt32.self)[position] = UInt32(tokenID)
        if position == 0 { known = [] }
        known.append(tokenID)
        promptEnd = position + 2   // the argmax does not write the next token
        let (a, b) = encodeStep(pos: position)
        a.commit(); b.commit()
        readEngram(position)
        b.waitUntilCompleted()
        if let err = b.error { fatalError("step failed: \(err)") }
        let next = Int(greedy.contents().assumingMemoryBound(to: UInt32.self)[position + 1])
        position += 1
        return (next, b.gpuEndTime - a.gpuStartTime)
    }

    /// Greedy decoding of `count` new tokens after `prompt`, pipelined: the GPU runs one step while
    /// the CPU encodes the next and reads its engram rows. Returns the new tokens, the wall time per
    /// generated token and the median GPU time per step.
    /// Profiling only: do not wait for a step's token before reading the next step's engram rows
    /// (the rows are then wrong; the time shows what the CPU round trip costs).
    public var noWait = ProcessInfo.processInfo.environment["EMBLA_NOWAIT"] == "1"

    /// `sampling` picks the tokens; `onToken`, called with each new token as soon as the next
    /// step is under way, ends generation early by returning false (one step more has then run,
    /// and its token is dropped).
    public func generate(prompt: [Int], count: Int, sampling: Sampling = .greedy,
                         onToken: (Int) -> Bool = { _ in true }) -> (tokens: [Int], wallPerToken: Double, gpuPerStep: Double) {
        precondition(!prompt.isEmpty && count > 0, "need a prompt and a token to generate")
        reset()
        self.sampling = sampling
        engramBase = engramReady.signaledValue
        engramSeconds = 0
        waitSeconds = 0
        precondition(prompt.count + count < maxContext, "context full")
        let t = tokens.contents().assumingMemoryBound(to: UInt32.self)
        for (i, id) in prompt.enumerated() { t[i] = UInt32(id) }
        known = prompt
        promptEnd = prompt.count
        let steps = prompt.count + count - 1, P = prompt.count
        var submitted: [(MTLCommandBuffer, MTLCommandBuffer)] = []
        var wakeTimes = [Double](repeating: 0, count: steps + 1), signalTimes = wakeTimes
        // the prompt: every step submitted as soon as its engram rows are read (the tokens are
        // known), up to engramRing ahead of the GPU; all but the last stop at layer 16
        let promptStart = Date()
        traceStart = CACurrentMediaTime()
        var first = 0
        var prefilling: [MTLCommandBuffer] = []
        if batchedPrefill && !fullPrompt && P > 1 {
            prefilling = prefill(upTo: P - 1)
            first = P - 1
        }
        for q in first ..< P {
            if q - first >= Self.engramRing { submitted[q - first - Self.engramRing].1.waitUntilCompleted() }
            let c = encodeStep(pos: q, prompt: q < P - 1 && !fullPrompt)
            c.0.commit(); c.1.commit()
            submitted.append(c)
            readEngram(q)
        }
        // then generation: step p + 1 is encoded while step p runs; its engram rows need step p's token
        var genStart = Date()
        var stoppedAt: Int?
        for p in P - 1 ..< steps {
            if p + 1 < steps {
                let c = encodeStep(pos: p + 1)
                c.0.commit(); c.1.commit()
                submitted.append(c)
            }
            let w0 = DispatchTime.now().uptimeNanoseconds
            if !noWait { known.append(awaitStep(p, submitted[p - first].1)) } else { known.append(0) }
            if p == P - 1 {
                mark("first token")
                if trace {
                    let s = submitted[p - first]
                    print(String(format: "trace: last prompt step GPU %.2f .. %.2f ms", (s.0.gpuStartTime - traceStart!) * 1e3, (s.1.gpuEndTime - traceStart!) * 1e3))
                }
                finishPrefill(prefilling)
                prefillSeconds = Date().timeIntervalSince(promptStart)
                genStart = Date()
            } else {
                waitSeconds += Double(DispatchTime.now().uptimeNanoseconds - w0) * 1e-9
            }
            wakeTimes[p] = CACurrentMediaTime()
            if p + 1 < steps { readEngram(p + 1) }
            signalTimes[p + 1] = CACurrentMediaTime()
            if !onToken(known[p + 1]), p + 1 < steps {
                stoppedAt = p
                break
            }
        }
        submitted.last!.1.waitUntilCompleted()
        // an early stop: the steps that made the tokens kept (the one after them ran for nothing)
        let made = stoppedAt.map { $0 - (P - 1) + 1 } ?? count
        if let stop = stoppedAt { submitted = Array(submitted.prefix(stop - first + 1)) }
        let count = made
        let wall = Date().timeIntervalSince(genStart) / Double(max(1, count - 1))
        position = P + count - 1
        let gpu = submitted.suffix(count).map { $0.1.gpuEndTime - $0.0.gpuStartTime }.sorted()[count / 2]
        func median(_ v: [Double]) -> Double { v.isEmpty ? 0 : v.sorted()[v.count / 2] * 1e6 }
        let tail = Array(submitted.suffix(count))
        let base = submitted.count - count
        latency = (median((base ..< submitted.count - 1).map { wakeTimes[$0] - submitted[$0].1.gpuEndTime }),
                   median((base ..< submitted.count - 1).map { signalTimes[$0 + 1] - wakeTimes[$0] }),
                   median((base + 1 ..< submitted.count).map { submitted[$0].1.gpuStartTime - signalTimes[$0] }))
        gaps = (median(tail.map { $0.0.gpuEndTime - $0.0.gpuStartTime }), median(tail.map { $0.1.gpuStartTime - $0.0.gpuEndTime }),
                median(tail.map { $0.1.gpuEndTime - $0.1.gpuStartTime }),
                median(zip(tail.dropFirst(), tail).map { $0.0.0.gpuStartTime - $0.1.1.gpuEndTime }))
        return (Array(known[P ..< P + count]), wall, gpu)
    }
}

/// A load the compiler keeps in a polling loop, ordered before the loads after it.
@inline(never)
func loadAcquire(_ p: UnsafeMutablePointer<UInt32>) -> UInt32 {
    let v = UnsafeRawPointer(p).load(as: UInt32.self)
    atomicMemoryFence(ordering: .acquiring)
    return v
}

/// How a step picks its token: the argmax, or a draw from softmax(logits / temperature) cut to
/// its top-p nucleus (on the GPU, so decoding stays pipelined).
public struct Sampling: Sendable, Equatable {
    public var temperature: Float
    public var topP: Float
    public var seed: UInt32

    public static let greedy = Sampling(temperature: 0, topP: 1, seed: 0)

    public init(temperature: Float, topP: Float = 1, seed: UInt32 = .random(in: 0 ... .max)) {
        self.temperature = temperature
        self.topP = min(max(topP, 0), 1)
        self.seed = seed
    }

    public var isGreedy: Bool { temperature <= 0 || topP <= 0 }
}
