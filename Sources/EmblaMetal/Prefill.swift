import Foundation
import Metal

/// As `GemmParams` in gemm.metal.
struct GemmParams {
    var T: Int32, O: Int32, ldx: Int32, ldy: Int32, K: Int32, pad0: Int32 = 0, pad1: Int32 = 0, pad2: Int32 = 0
    var ew = SIMD3<UInt64>(0, 0, 0)
}

/// Batched prefill: the prefill half (layers 0 to 15 and layer 16's K/V) for chunks of up to
/// `Prefill.chunk` prompt tokens at once. The matrices go through the 4-bit GEMM (each weight read
/// once a chunk, not once a token); the per-token pieces through prefill.metal. It fills the same
/// caches as the decode steps, so decoding continues from where it stops.
final class Prefill {
    /// Whether the GEMMs run on the neural accelerators (gemm_nax.metal, Metal 4 tensor ops); set
    /// at init: only where the GPU has them (`MetalContext.hasNeuralAccelerators`, the M5 on; EMBLA_NAX
    /// overrides) and the kernels compile.
    let nax: Bool
    /// Tokens a GEMM threadgroup takes, and so entries an expert tile holds.
    var bm: Int { nax ? Self.expertRows * Self.expertTiles : Engine.envMacros("EMBLA_GEMM")["BM"] ?? 32 }
    /// Token subtiles a routed expert's threadgroup takes, sharing each dequantized weight tile.
    static let expertTiles = Engine.envMacros("EMBLA_PREFILL")["EXPERT_TILES"] ?? 1
    /// Entries of a routed expert's tile (and tokens of its GEMM threadgroups) on the accelerators.
    static let expertRows = Engine.envMacros("EMBLA_PREFILL")["EXPERT_ROWS"] ?? 64
    var threads: Int { nax ? 128 : 256 }
    static let chunk = Engine.envMacros("EMBLA_PREFILL")["CHUNK"] ?? 2048
    /// The sliding layers' attention on simdgroup matrices, a kv head and two tokens a threadgroup
    /// (attn_sliding8_b), not a query head and token a threadgroup: a 1024-token prompt 245 -> 225 ms
    /// on the iPhone 18 Pro, 109 -> 102 ms on the M5 Pro (16k: 1530 -> 1400 ms); against the reference
    /// logits as close as the old kernel (9 prompts). EMBLA_PREFILL=SLIDING8=0 for the old one.
    let sliding8: Bool
    /// Token subtiles of 64 a dense GEMM threadgroup takes on the accelerators, sharing each
    /// dequantized W tile (the dequantization is then done once for 64 · denseTiles tokens).
    static let denseTiles = Engine.envMacros("EMBLA_PREFILL")["DENSE_TILES"] ?? 1

    unowned let engine: Engine
    let state, x, inp, qkv, o, hmlp, hex, hsh, yex, ysh, e, ekv, gated: MTLBuffer
    let logits, experts, gates, entryTok, slotEntry, tiles, ntiles, counter: MTLBuffer
    var part: MTLBuffer
    /// The neural accelerator attention (attn_nax.metal) for the full layer, its normed queries and
    /// thresholds (0).
    var pAttnNax: MTLComputePipelineState? = nil
    let qn, thresholds: MTLBuffer
    let pQPrep: MTLComputePipelineState

    struct GemmPipes { let plain: [MTLComputePipelineState], experts: [MTLComputePipelineState], gather: [MTLComputePipelineState] }
    /// W4A8 (EMBLA_W4A8=1, with the neural accelerators): every prefill GEMM's input quantized to
    /// int8 (q8_act), then gemm_w4a8. Faster, but no longer bf16-exact with the reference.
    var w4a8: (quant: MTLComputePipelineState, gemm: GemmPipes)? = nil
    var xq: MTLBuffer! = nil, sx: MTLBuffer! = nil, xz: MTLBuffer! = nil
    let gemm: GemmPipes
    private var eOffset = 0
    /// Profiling only: categories of dispatches to encode twice ("gemm", "mix", "attn", "router",
    /// "moe", "engram", "embed").
    var twice: Set<String> = []
    private func dispatch(_ enc: MTLComputeCommandEncoder, _ part: String, _ g: MTLSize, _ t: MTLSize, threads: Bool = false) {
        for k in 0 ..< (twice.contains(part) ? 2 : 1) {
            if k == 1 { enc.memoryBarrier(scope: .buffers) }
            if threads { enc.dispatchThreads(g, threadsPerThreadgroup: t) } else { enc.dispatchThreadgroups(g, threadsPerThreadgroup: t) }
        }
    }
    let pEmbed, pGate, pConv, pKVSliding, pSliding, pKV, pFull, pRouter, pRoute, pGroup, pCombine: MTLComputePipelineState
    /// Per layer: the attention and MLP input mixes (one threadgroup a token).
    var attnMix: [MTLComputePipelineState] = [], mlpMix: [MTLComputePipelineState] = []
    let engramMix: MTLComputePipelineState
    /// Per MoE layer: byte offsets of the down projections in their published layout.
    var down: [Int: (routed: (UInt64, UInt64, UInt64), shared: (UInt64, UInt64, UInt64))] = [:]

    init(_ engine: Engine) throws {
        self.engine = engine
        let ctx = engine.ctx, T = Self.chunk, d = Embla.d
        sliding8 = (Engine.envMacros("EMBLA_PREFILL")["SLIDING8"] ?? 1) != 0
        func buf(_ bytes: Int) -> MTLBuffer { ctx.device.makeBuffer(length: bytes, options: .storageModePrivate)! }
        state = buf(T * 9 * d * 2)
        x = buf(T * d * 4)
        inp = buf(T * 3 * d * 2)
        qkv = buf(T * 4096 * 2)
        o = buf(T * d * 2)
        hmlp = buf(T * d * 2)
        hex = buf(T * 8 * 512 * 2)
        hsh = buf(T * 1024 * 2)
        yex = buf(T * 8 * d * 2)
        ysh = buf(T * d * 2)
        e = ctx.device.makeBuffer(length: 2 * T * d * 2, options: .storageModeShared)!   // two chunks' rows
        ekv = buf(T * 2 * d * 2)
        gated = buf(T * d * 2)
        logits = buf(T * Embla.experts * 4)
        experts = buf(T * 8 * 4)
        gates = buf(T * 8 * 4)
        entryTok = buf(T * 8 * 4)
        slotEntry = buf(T * 8 * 4)
        tiles = buf((T * 8 / 32 + Embla.experts) * 16)
        ntiles = buf(16)
        part = buf(16)   // the per-key full attention's partial sums: allocated below if it is used
        counter = ctx.device.makeBuffer(length: T * 2 * 4, options: .storageModeShared)!
        qn = buf(16 * T * 128 * 2)
        thresholds = ctx.device.makeBuffer(length: 16 * 4, options: .storageModeShared)!
        memset(thresholds.contents(), 0, 16 * 4)
        memset(counter.contents(), 0, counter.length)

        let mix = try Engine.kernelSource("mix")
        func pipes(_ name: String, _ source: String, _ extra: [String: Int]) throws -> GemmPipes {
            GemmPipes(
                plain: try (0 ..< 5).map { try ctx.pipeline(name, source: source, macros: extra.merging(["EPI": $0]) { a, _ in a }) },
                experts: try (0 ..< 5).map { try ctx.pipeline(name, source: source, macros: extra.merging(["EPI": $0, "EXPERTS": 1]) { a, _ in a }) },
                gather: try (0 ..< 5).map { try ctx.pipeline(name, source: source, macros: extra.merging(["EPI": $0, "EXPERTS": 1, "GATHER": 1]) { a, _ in a }) })
        }
        var g: GemmPipes? = nil
        if ctx.hasNeuralAccelerators {
            let pad = Engine.envMacros("EMBLA_PREFILL")["PAD"].map { "#define NPAD \($0)\n" } ?? ""
            let src = "#define NT bfloat\n" + pad + (try Engine.naxSource("gemm_nax"))
            if let dense = try? pipes("gemm_nax", src, Self.denseTiles > 1 ? ["MT": Self.denseTiles] : [:]) {
                // the routed experts' tiles: expertRows tokens (fewer padding rows when an expert gets few)
                let small = try pipes("gemm_nax", src, ["NBM": Self.expertRows, "MT": Self.expertTiles])
                g = GemmPipes(plain: dense.plain, experts: small.experts, gather: small.gather)
            }
        }
        nax = g != nil
        let bm = nax ? Self.expertRows : Engine.envMacros("EMBLA_GEMM")["BM"] ?? 32
        if g == nil { g = try pipes("gemm", Engine.kernelSource("gemm"), ["BM": bm]) }
        gemm = g!
        if nax && ProcessInfo.processInfo.environment["EMBLA_W4A8"] == "1" {
            let src = try Engine.kernelSource("w4a8")
            w4a8 = (try ctx.pipeline("q8_act", source: src), try pipes("gemm_w4a8", src, [:]))
            let rows = max(T * 8, T)
            xq = buf(rows * 2048)
            sx = buf(rows * 32 * 4)
            xz = buf(rows * 32 * 4)
        }
        let pf = "#define TILE_ENTRIES \(bm)\n" + (try Engine.kernelSource("prefill"))
        pEmbed = try ctx.pipeline("embed_b", source: pf)
        pGate = try ctx.pipeline("engram_gate_b", source: pf)
        pConv = try ctx.pipeline("engram_conv_b", source: pf)
        pKVSliding = try ctx.pipeline("kv_sliding_b", source: pf)
        pSliding = try ctx.pipeline(sliding8 ? "attn_sliding8_b" : "attn_sliding_b", source: pf)
        pKV = try ctx.pipeline("kv_append_b", source: pf)
        pFull = try ctx.pipeline("attn_full_b", source: pf)
        pRouter = try ctx.pipeline("router_b", source: pf)
        pRoute = try ctx.pipeline("route_b", source: pf)
        pGroup = try ctx.pipeline("group_b", source: pf)
        pCombine = try ctx.pipeline("moe_combine_b", source: pf)
        pQPrep = try ctx.pipeline("q_prep_b", source: pf, macros: Engine.envMacros("EMBLA_ATT8"))
        if nax {
            do { pAttnNax = try ctx.pipeline("relu2_prefill_nax", source: Engine.naxSource("attn_nax"), macros: Engine.envMacros("EMBLA_ATT8")) } catch {
                if ProcessInfo.processInfo.environment["EMBLA_VERBOSE"] != nil { print("attn_nax: \(error)") }
            }
        }
        if pAttnNax == nil { part = buf(T * 16 * 16 * 129 * 4) }
        if ProcessInfo.processInfo.environment["EMBLA_VERBOSE"] != nil {
            print("prefill: GEMM on \(nax ? "neural accelerators" : "simdgroup matrices"), full attention \(pAttnNax != nil ? "on neural accelerators" : "a key at a time")")
        }
        func mixPipe(n: Int, k: Int, hasX: Bool, norm: Bool, append: Bool) throws -> MTLComputePipelineState {
            try ctx.pipeline("mix", source: mix, macros: ["N": n, "K": k, "HAS_X": hasX ? 1 : 0, "NORM": norm ? 1 : 0, "APPEND": append ? 1 : 0])
        }
        for L in engine.layers where L.i <= Embla.prefillLayers {
            let i = L.i, n = i == 0 ? 1 : 2 + i / Embla.blockSize, blockStart = i % Embla.blockSize == 0
            attnMix.append(try mixPipe(n: n, k: L.attnK, hasX: !blockStart, norm: true, append: blockStart && i > 0))
            mlpMix.append(try mixPipe(n: n, k: 1, hasX: true, norm: true, append: false))
            if i > 0 {
                func off(_ name: String) throws -> (UInt64, UInt64, UInt64) {
                    let p = "transformer.h.\(i).mlp.\(name)."
                    return (UInt64(try engine.weights.tensor(p + "weight").offset), UInt64(try engine.weights.tensor(p + "scales").offset),
                            UInt64(try engine.weights.tensor(p + "biases").offset))
                }
                down[i] = (try off("experts_proj"), try off("shared_proj"))
            }
        }
        engramMix = try mixPipe(n: 1, k: 1, hasX: true, norm: false, append: false)
    }

    // ------------------------------------------------------------------ encoding helpers

    private func barrier(_ enc: MTLComputeCommandEncoder) { enc.memoryBarrier(scope: .buffers) }

    enum Kind { case plain, experts, gather }

    private static func pick(_ g: GemmPipes, _ kind: Kind, _ epi: Int) -> MTLComputePipelineState {
        switch kind { case .plain: return g.plain[epi]; case .experts: return g.experts[epi]; case .gather: return g.gather[epi] }
    }

    /// Y = X Wᵀ for `tokens` rows of X (EPI `epi`); W at byte offsets `w`; `kind` .experts / .gather for
    /// the routed experts' tiles (X rows gathered by token).
    private func gemm(_ enc: MTLComputeCommandEncoder, _ kind: Kind, epi: Int, w: (UInt64, UInt64, UInt64),
                      x: MTLBuffer, xOffset: Int = 0, ldx: Int, y: MTLBuffer, yOffset: Int = 0, ldy: Int,
                      tokens: Int, rows: Int, inputs: Int = Embla.d, experts: (stride: SIMD3<UInt64>, tiles: Int)? = nil) {
        if let q = w4a8 {
            // quantize X's rows (all T of them for a gathered GEMM: it reads them by token), then the int8 GEMM
            let xrows = kind == .gather ? tokens / 8 : tokens
            enc.setComputePipelineState(q.quant)
            enc.setBuffer(x, offset: xOffset, index: 0)
            enc.setBuffer(xq, offset: 0, index: 1)
            enc.setBuffer(sx, offset: 0, index: 2)
            enc.setBuffer(xz, offset: 0, index: 3)
            var kl = SIMD2<Int32>(Int32(inputs), Int32(ldx))
            enc.setBytes(&kl, length: 8, index: 4)
            dispatch(enc, "quant", MTLSize(width: 32 * inputs / 64, height: xrows, depth: 1), MTLSize(width: 256, height: 1, depth: 1), threads: true)
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(Self.pick(q.gemm, kind, epi))
            enc.setBuffer(engine.W, offset: 0, index: 0)
            var wo = SIMD3<UInt64>(w.0, w.1, w.2)
            enc.setBytes(&wo, length: 32, index: 1)
            enc.setBuffer(xq, offset: 0, index: 2)
            enc.setBuffer(y, offset: yOffset, index: 3)
            var p = GemmParams(T: Int32(tokens), O: Int32(rows), ldx: Int32(inputs), ldy: Int32(ldy), K: Int32(inputs), ew: experts?.stride ?? .zero)
            enc.setBytes(&p, length: MemoryLayout<GemmParams>.size, index: 4)
            enc.setBuffer(tiles, offset: 0, index: 5)
            enc.setBuffer(ntiles, offset: 0, index: 6)
            enc.setBuffer(entryTok, offset: 0, index: 7)
            enc.setBuffer(sx, offset: 0, index: 8)
            enc.setBuffer(xz, offset: 0, index: 9)
            dispatch(enc, experts != nil ? "experts" : "gemm", MTLSize(width: (rows + 63) / 64, height: experts?.tiles ?? (tokens + 63) / 64, depth: 1),
                     MTLSize(width: 128, height: 1, depth: 1))
            return
        }
        enc.setComputePipelineState(Self.pick(gemm, kind, epi))
        enc.setBuffer(engine.W, offset: 0, index: 0)
        var wo = SIMD3<UInt64>(w.0, w.1, w.2)
        enc.setBytes(&wo, length: 32, index: 1)
        enc.setBuffer(x, offset: xOffset, index: 2)
        enc.setBuffer(y, offset: yOffset, index: 3)
        var p = GemmParams(T: Int32(tokens), O: Int32(rows), ldx: Int32(ldx), ldy: Int32(ldy), K: Int32(inputs), ew: experts?.stride ?? .zero)
        enc.setBytes(&p, length: MemoryLayout<GemmParams>.size, index: 4)
        enc.setBuffer(tiles, offset: 0, index: 5)
        enc.setBuffer(ntiles, offset: 0, index: 6)
        enc.setBuffer(entryTok, offset: 0, index: 7)
        // threadgroups: on the accelerators 64 x 64 (dense) or expertRows x (4096 / expertRows)
        let tokensPer = nax ? (experts != nil ? Self.expertRows * Self.expertTiles : 64 * Self.denseTiles) : bm
        let rowsPer = nax ? 4096 / (experts != nil ? Self.expertRows : 64) : 64
        dispatch(enc, experts != nil ? "experts" : "gemm", MTLSize(width: (rows + rowsPer - 1) / rowsPer, height: experts?.tiles ?? (tokens + tokensPer - 1) / tokensPer, depth: 1),
                 MTLSize(width: threads, height: 1, depth: 1))
    }

    /// A segment of the stacked GEMVs as a GEMM over the chunk: its rows of W, its input mix, its
    /// output columns of qkv.
    private func segmentGEMM(_ enc: MTLComputeCommandEncoder, _ segs: [Seg], tokens: Int, skipFirst: Bool = false) {
        var start: UInt32 = 0
        for (i, s) in segs.enumerated() {
            defer { start = s.end }
            if skipFirst && i == 0 { continue }
            gemm(enc, .plain, epi: 0, w: (s.w, s.s, s.b), x: inp, xOffset: Int(s.x) * 2, ldx: 3 * Embla.d,
                 y: qkv, yOffset: Int(s.y) * 2, ldy: 4096, tokens: tokens, rows: Int(s.end - start))
        }
    }

    private func mix(_ enc: MTLComputeCommandEncoder, _ pipe: MTLComputePipelineState, col: Int32, tokens: Int) {
        enc.setComputePipelineState(pipe)
        enc.setBuffer(state, offset: 0, index: 0)
        enc.setBuffer(engine.queries, offset: 0, index: 1)
        enc.setBuffer(x, offset: 0, index: 2)
        enc.setBuffer(inp, offset: 0, index: 3)
        var c = col
        enc.setBytes(&c, length: 4, index: 4)
        dispatch(enc, "mix", MTLSize(width: tokens, height: 1, depth: 1), MTLSize(width: 1024, height: 1, depth: 1))
    }

    // ------------------------------------------------------------------ the chunk

    /// Encodes the prefill half for positions p0 ..< p0 + tokens (their ids in engine.tokens, their
    /// engram rows in half `half` of `e`).
    ///
    /// Returns the chunk's last command buffer, encoded but not committed. With `wait`, the
    /// embedding and layer 0 are a command buffer of their own, committed as soon as they are
    /// encoded (the GPU starts while the CPU encodes the rest), and the rest waits for the event to
    /// reach that value before the engram stage: the CPU reads the chunk's rows meanwhile.
    func encode(p0: Int, tokens T: Int, half: Int, wait: (MTLSharedEvent, UInt64)? = nil) -> MTLCommandBuffer {
        eOffset = half * Self.chunk * Embla.d * 2
        let E = engine, d = Embla.d
        var cb = E.chainedCommandBuffer()
        var enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        var pos0 = Int32(p0)

        enc.setComputePipelineState(pEmbed)
        enc.setBuffer(E.W, offset: 0, index: 0)
        var off = SIMD3<UInt64>(E.wte.0, E.wte.1, E.wte.2)
        enc.setBytes(&off, length: 32, index: 1)
        enc.setBuffer(E.tokens, offset: 0, index: 2)
        enc.setBuffer(state, offset: 0, index: 3)
        enc.setBytes(&pos0, length: 4, index: 4)
        dispatch(enc, "embed", MTLSize(width: 1, height: T, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
        barrier(enc)

        for L in E.layers where L.i <= Embla.prefillLayers {
            let i = L.i
            // ---- attention
            mix(enc, attnMix[i], col: L.attnCol, tokens: T)
            barrier(enc)
            let last = i == Embla.prefillLayers
            segmentGEMM(enc, L.qkv, tokens: T, skipFirst: last)   // layer 16: its K/V only
            barrier(enc)
            switch L.kind {
            case .sliding:
                enc.setComputePipelineState(pKVSliding)
                enc.setBuffer(qkv, offset: 0, index: 0)
                enc.setBuffer(E.W, offset: L.kGain, index: 1)
                enc.setBuffer(E.cache, offset: L.kOffset, index: 2)
                enc.setBuffer(E.cache, offset: L.vOffset, index: 3)
                enc.setBytes(&pos0, length: 4, index: 4)
                dispatch(enc, "attn", MTLSize(width: 4, height: T, depth: 1), MTLSize(width: 32, height: 1, depth: 1))
                barrier(enc)
                enc.setComputePipelineState(pSliding)
                enc.setBuffer(qkv, offset: 0, index: 0)
                enc.setBuffer(E.W, offset: L.qGain, index: 1)
                enc.setBuffer(E.cache, offset: L.kOffset, index: 2)
                enc.setBuffer(E.cache, offset: L.vOffset, index: 3)
                enc.setBuffer(o, offset: 0, index: 4)
                enc.setBytes(&pos0, length: 4, index: 5)
                if sliding8 {
                    var t32 = Int32(T)
                    enc.setBytes(&t32, length: 4, index: 6)
                    dispatch(enc, "attn", MTLSize(width: 4, height: (T + 1) / 2, depth: 1), MTLSize(width: 128, height: 1, depth: 1))
                } else {
                    dispatch(enc, "attn", MTLSize(width: 16, height: T, depth: 1), MTLSize(width: 128, height: 1, depth: 1))
                }
            case .full, .sparse:
                enc.setComputePipelineState(pKV)
                enc.setBuffer(qkv, offset: 0, index: 0)
                enc.setBuffer(E.cache, offset: 0, index: 1)
                L.kvJobs.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
                var pm = SIMD2<Int32>(pos0, Int32(E.maxContext))
                enc.setBytes(&pm, length: 8, index: 3)
                dispatch(enc, "attn", MTLSize(width: 2 * L.kvJobs.count, height: T, depth: 1), MTLSize(width: 32, height: 1, depth: 1))
                barrier(enc)
                if last { enc.endEncoding(); E.endChained(cb); return cb }
                if let nax = pAttnNax {
                    encodeFullNAX(enc, L, nax, p0: p0, tokens: T)
                    break
                }
                let keys = p0 + T, ch = max(64, (keys + 15) / 16 + 63) / 64 * 64, c = (keys + ch - 1) / ch
                var ap = SIMD4<Int32>(pos0, Int32(ch), Int32(c), Int32(E.maxContext))
                enc.setComputePipelineState(pFull)
                enc.setBuffer(qkv, offset: 0, index: 0)
                enc.setBuffer(E.W, offset: L.qGain, index: 1)
                enc.setBuffer(E.W, offset: L.kGain, index: 2)
                enc.setBuffer(E.cache, offset: L.kOffset, index: 3)
                enc.setBuffer(E.cache, offset: L.vOffset, index: 4)
                enc.setBuffer(part, offset: 0, index: 5)
                enc.setBuffer(counter, offset: 0, index: 6)
                enc.setBuffer(o, offset: 0, index: 7)
                enc.setBytes(&ap, length: 16, index: 8)
                dispatch(enc, "attn", MTLSize(width: c, height: 2, depth: T), MTLSize(width: 256, height: 1, depth: 1))
            }
            barrier(enc)
            let os = L.oproj[0]
            gemm(enc, .plain, epi: L.oFirst ? 3 : 2, w: (os.w, os.s, os.b), x: o, ldx: d, y: x, ldy: d, tokens: T, rows: d)
            barrier(enc)

            // ---- MLP
            mix(enc, mlpMix[i], col: L.mlpCol, tokens: T)
            barrier(enc)
            if let (fc, proj) = L.dense {
                gemm(enc, .plain, epi: 1, w: (fc[0].w, fc[0].s, fc[0].b), x: inp, ldx: 3 * d, y: hmlp, ldy: d, tokens: T, rows: d)
                barrier(enc)
                gemm(enc, .plain, epi: 2, w: (proj[0].w, proj[0].s, proj[0].b), x: hmlp, ldx: d, y: x, ldy: d, tokens: T, rows: d)
                barrier(enc)
            } else {
                encodeMoE(enc, L, tokens: T)
            }
            if i == 0 {
                if let (event, value) = wait {
                    enc.endEncoding()
                    E.endChained(cb)
                    cb.commit()
                    E.mark("layer 0 committed")
                    cb = E.chainedCommandBuffer()
                    cb.encodeWaitForEvent(event, value: value)
                    enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
                }
                encodeEngram(enc, tokens: T, pos0: &pos0)
            }
        }
        enc.endEncoding()
        E.endChained(cb)
        return cb
    }

    /// The full layer's attention on the neural accelerators: the queries normed head-major, then
    /// the ReLU² prefill kernel (attn_nax.metal) over the cache (causal: the chunk is the last T of p0 + T keys).
    private func encodeFullNAX(_ enc: MTLComputeCommandEncoder, _ L: Engine.Layer, _ pipe: MTLComputePipelineState, p0: Int, tokens T: Int) {
        let E = engine
        var t32 = Int32(T)
        enc.setComputePipelineState(pQPrep)
        enc.setBuffer(qkv, offset: 0, index: 0)
        enc.setBuffer(E.W, offset: L.qGain, index: 1)
        enc.setBuffer(E.W, offset: L.kGain, index: 2)
        enc.setBuffer(qn, offset: 0, index: 3)
        enc.setBytes(&t32, length: 4, index: 4)
        dispatch(enc, "attn", MTLSize(width: 16, height: T, depth: 1), MTLSize(width: 32, height: 1, depth: 1))
        barrier(enc)
        enc.setComputePipelineState(pipe)
        enc.setBuffer(qn, offset: 0, index: 0)
        enc.setBuffer(E.cache, offset: L.kOffset, index: 1)   // bf16, or int8 rows and their scales
        enc.setBuffer(E.cache, offset: L.vOffset, index: 2)
        var strides: [Int64] = [0, Int64(E.maxContext * 128), 128]
        enc.setBytes(&strides, length: 24, index: 3)
        enc.setBytes(&strides, length: 24, index: 4)
        var tk = Int32(p0 + T), heads: Int32 = 16, g: Int32 = 8
        var scale = 1 / Float(128).squareRoot(), sink = Float(128).squareRoot()
        enc.setBytes(&t32, length: 4, index: 5)
        enc.setBytes(&tk, length: 4, index: 6)
        enc.setBytes(&heads, length: 4, index: 7)
        enc.setBytes(&g, length: 4, index: 8)
        enc.setBytes(&scale, length: 4, index: 9)
        enc.setBytes(&sink, length: 4, index: 10)
        enc.setBuffer(thresholds, offset: 0, index: 11)
        enc.setBuffer(o, offset: 0, index: 12)
        dispatch(enc, "attn", MTLSize(width: (T + 63) / 64, height: 16, depth: 1), MTLSize(width: 128, height: 1, depth: 1))
    }

    private func encodeMoE(_ enc: MTLComputeCommandEncoder, _ L: Engine.Layer, tokens T: Int) {
        let E = engine, d = Embla.d, mw = L.moe!
        enc.setComputePipelineState(pRouter)
        enc.setBuffer(E.W, offset: Int(mw.router), index: 0)
        enc.setBuffer(inp, offset: 0, index: 1)
        enc.setBuffer(logits, offset: 0, index: 2)
        dispatch(enc, "router", MTLSize(width: Embla.experts / 8, height: T, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
        // the shared expert's up projection needs no routing
        gemm(enc, .plain, epi: 1, w: (mw.sh_w, mw.sh_s, mw.sh_b), x: inp, ldx: 3 * d, y: hsh, ldy: 1024, tokens: T, rows: 1024)
        barrier(enc)
        enc.setComputePipelineState(pRoute)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(E.W, offset: Int(mw.bias), index: 1)
        enc.setBuffer(experts, offset: 0, index: 2)
        enc.setBuffer(gates, offset: 0, index: 3)
        dispatch(enc, "moe", MTLSize(width: T, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1))
        barrier(enc)
        enc.setComputePipelineState(pGroup)
        enc.setBuffer(experts, offset: 0, index: 0)
        enc.setBuffer(entryTok, offset: 0, index: 1)
        enc.setBuffer(slotEntry, offset: 0, index: 2)
        enc.setBuffer(tiles, offset: 0, index: 3)
        enc.setBuffer(ntiles, offset: 0, index: 4)
        var t32 = Int32(T)
        enc.setBytes(&t32, length: 4, index: 5)
        dispatch(enc, "moe", MTLSize(width: 1, height: 1, depth: 1), MTLSize(width: 128, height: 1, depth: 1))
        // the shared expert's down projection meanwhile
        let (routed, shared) = down[L.i]!
        gemm(enc, .plain, epi: 0, w: shared, x: hsh, ldx: 1024, y: ysh, ldy: d, tokens: T, rows: d, inputs: 1024)
        barrier(enc)
        let maxTiles = T * 8 / bm + Embla.experts
        // routed experts: up (rows of each expert, X gathered by token), then down (K = 512)
        gemm(enc, .gather, epi: 1, w: (mw.fc_w, mw.fc_s, mw.fc_b), x: inp, ldx: 3 * d, y: hex, ldy: 512, tokens: T * 8, rows: 512,
             experts: (SIMD3(UInt64(512 * 256 * 4), UInt64(512 * 32 * 2), UInt64(512 * 32 * 2)), maxTiles))
        barrier(enc)
        gemm(enc, .experts, epi: 0, w: routed, x: hex, ldx: 512, y: yex, ldy: d, tokens: T * 8, rows: d, inputs: 512,
             experts: (SIMD3(UInt64(2048 * 64 * 4), UInt64(2048 * 8 * 2), UInt64(2048 * 8 * 2)), maxTiles))
        barrier(enc)
        enc.setComputePipelineState(pCombine)
        enc.setBuffer(ysh, offset: 0, index: 0)
        enc.setBuffer(yex, offset: 0, index: 1)
        enc.setBuffer(gates, offset: 0, index: 2)
        enc.setBuffer(slotEntry, offset: 0, index: 3)
        enc.setBuffer(x, offset: 0, index: 4)
        dispatch(enc, "moe", MTLSize(width: d, height: T, depth: 1), MTLSize(width: 256, height: 1, depth: 1), threads: true)
        barrier(enc)
    }

    private func encodeEngram(_ enc: MTLComputeCommandEncoder, tokens T: Int, pos0: inout Int32) {
        let E = engine, d = Embla.d
        mix(enc, engramMix, col: E.engramMix.col, tokens: T)
        for (j, s) in E.engramSegs.enumerated() {
            gemm(enc, .plain, epi: 0, w: (s.w, s.s, s.b), x: e, xOffset: eOffset, ldx: d, y: ekv, yOffset: j * d * 2, ldy: 2 * d, tokens: T, rows: d)
        }
        barrier(enc)
        enc.setComputePipelineState(pGate)
        enc.setBuffer(ekv, offset: 0, index: 0)
        enc.setBuffer(inp, offset: 0, index: 1)
        enc.setBuffer(gated, offset: 0, index: 2)
        enc.setBuffer(E.hist, offset: 0, index: 3)
        enc.setBytes(&pos0, length: 4, index: 4)
        dispatch(enc, "engram", MTLSize(width: 1, height: T, depth: 1), MTLSize(width: 1024, height: 1, depth: 1))
        barrier(enc)
        enc.setComputePipelineState(pConv)
        enc.setBuffer(gated, offset: 0, index: 0)
        enc.setBuffer(E.hist, offset: 0, index: 1)
        enc.setBuffer(E.W, offset: E.convWeight, index: 2)
        enc.setBuffer(state, offset: 0, index: 3)
        enc.setBytes(&pos0, length: 4, index: 4)
        dispatch(enc, "engram", MTLSize(width: d, height: T, depth: 1), MTLSize(width: 256, height: 1, depth: 1), threads: true)
        barrier(enc)
    }
}
