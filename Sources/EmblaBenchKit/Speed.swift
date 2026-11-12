import EmblaMetal
import Metal
import Foundation

/// `embla-bench speed <model dir> [tokens] [context]`: pipelined greedy decoding after a short prompt
/// (padded to `context` tokens, or the first `context` of EMBLA_PROMPT_FILE's token ids); wall and GPU time per token, then each part of the step alone (the
/// rest left out, so it runs on the buffers of a real step), less the step with everything left out.
func runSpeed(_ args: [String]) throws {
    let n = args.count > 1 ? Int(args[1])! : 128
    let context = args.count > 2 ? Int(args[2])! : 0
    let engine = try Engine(modelDirectory: URL(fileURLWithPath: args[0]), maxContext: max(4096, context + 2 * n + 64))
    var prompt = [758, 6426, 314, 29103, 368]
    while prompt.count < context { prompt += [13, 198, 758, 6426, 314, 29103, 368, 46447] }
    if let file = ProcessInfo.processInfo.environment["EMBLA_PROMPT_FILE"] {   // real text instead of the pattern
        let ids = try String(contentsOfFile: file, encoding: .utf8).split(whereSeparator: \.isWhitespace).map { Int($0)! }
        prompt = Array(ids.prefix(max(context, 5)))
    }
    if let parts = ProcessInfo.processInfo.environment["EMBLA_SKIP"] {   // profiling: leave parts out
        engine.skip = Set(parts.split(separator: ",").map(String.init))
    }
    _ = engine.generate(prompt: prompt, count: 16)
    let base = engine.generate(prompt: prompt, count: n)
    print(String(format: "prompt of %d tokens: %.0f ms (%.0f tok/s)", prompt.count, engine.prefillSeconds * 1e3, Double(prompt.count) / engine.prefillSeconds))
    print(String(format: "context %d: wall %.3f ms/token (%.0f tok/s), GPU %.3f ms/step (%.0f tok/s)",
                 prompt.count, base.wallPerToken * 1e3, 1 / base.wallPerToken, base.gpuPerStep * 1e3, 1 / base.gpuPerStep))
    print(String(format: "  GPU: first buffer %.0f µs, gap %.0f µs, second buffer %.0f µs, gap to next step %.0f µs", engine.gaps.0, engine.gaps.1, engine.gaps.2, engine.gaps.3))
    print(String(format: "  round trip: GPU done -> CPU awake %.0f µs, engram read + signal %.0f µs, signal -> GPU resumes %.0f µs", engine.latency.0, engine.latency.1, engine.latency.2))
    print(String(format: "  CPU per token: engram read %.0f µs, waiting for the step %.0f µs",
                 engine.engramSeconds / Double(n + prompt.count) * 1e6, engine.waitSeconds / Double(n) * 1e6))
    if ProcessInfo.processInfo.environment["EMBLA_TWICE"] != nil {
        // each part encoded twice in the real step: the growth is what it costs in place
        var total = 0.0
        let parts = ProcessInfo.processInfo.environment["EMBLA_TWICE"].map { $0.split(separator: ",").map(String.init) }.flatMap { $0 == ["1"] ? nil : $0 }
        for part in parts ?? ["mix", "qkv", "attn", "oproj", "router", "moe", "reduce", "lmhead"] {
            engine.twice = [part]
            let r = engine.generate(prompt: prompt, count: 64).gpuPerStep
            engine.twice = []
            let b = engine.generate(prompt: prompt, count: 64).gpuPerStep
            total += r - b
            print(String(format: "  %-7@ %6.3f ms in place", part as NSString, (r - b) * 1e3))
        }
        print(String(format: "  sum     %6.3f ms", total * 1e3))
    }
    if ProcessInfo.processInfo.environment["EMBLA_PARTS"] == nil { return }
    let parts = ["mix", "qkv", "attn", "oproj", "router", "moe", "reduce", "dense", "engram", "lmhead"]
    engine.skip = Set(parts)
    let empty = engine.generate(prompt: prompt, count: 32).gpuPerStep
    var total = 0.0
    for part in parts {
        engine.skip = Set(parts).subtracting([part])
        let r = engine.generate(prompt: prompt, count: 32).gpuPerStep
        total += r - empty
        print(String(format: "  %-7@ %6.3f ms", part as NSString, (r - empty) * 1e3))
    }
    print(String(format: "  sum     %6.3f ms, empty step %.3f ms", total * 1e3, empty * 1e3))
    engine.skip = []
}

/// `embla-bench gemv <model dir>`: the o projection GEMV alone, from DRAM and cached.
func runGEMV(_ args: [String]) throws {
    let engine = try Engine(modelDirectory: URL(fileURLWithPath: args[0]), maxContext: 4096)
    for (rows, nsg) in [(4, 8), (2, 8), (1, 8)] {
        let r = try engine.timeGEMV(macros: ["ROWS": rows, "NSG": nsg])
        print(String(format: "ROWS %d NSG %d: DRAM %.2f µs, cached %.2f µs (ideal 8.40)", rows, nsg, r.dram, r.cached))
    }
}

/// `embla-bench gemm <model dir>`: the prefill GEMM on the layers' o projections, by batch size.
func runGEMM(_ args: [String]) throws {
    let ctx = try MetalContext()
    let st = try SafeTensors(url: URL(fileURLWithPath: args[0]).appendingPathComponent("model.safetensors"), device: ctx.device)
    let macros = ["EPI": 0].merging(Engine.envMacros("EMBLA_GEMM")) { _, b in b }
    if ProcessInfo.processInfo.environment["EMBLA_W4A8"] != nil { return try runGEMMW4A8(ctx, st) }
    let nt = ProcessInfo.processInfo.environment["EMBLA_NAX"]   // half or bfloat: the neural accelerator GEMM
    let p = nt != nil
        ? try ctx.pipeline("gemm_nax", source: "#define NT \(nt!)\n" + Engine.naxSource("gemm_nax"), macros: macros)
        : try ctx.pipeline("gemm", source: Engine.kernelSource("gemm"), macros: macros)
    let bm = nt != nil ? 64 : macros["BM"] ?? 32, bn = 64, threads = nt != nil ? 128 : 256
    for T in [64, 256, 1024] {
        let x = ctx.buffer([UInt16](repeating: BF16.bits(0.01), count: T * 2048)), y = ctx.buffer(UInt16.self, count: T * 2048)
        var best = Double.infinity
        for _ in 0 ..< 3 {
            let cb = ctx.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
            for l in 0 ..< 32 {
                let pre = "transformer.h.\(l).attn.c_proj."
                var off = SIMD3<UInt64>(UInt64(try st.tensor(pre + "weight").offset), UInt64(try st.tensor(pre + "scales").offset), UInt64(try st.tensor(pre + "biases").offset))
                enc.setComputePipelineState(p)
                enc.setBuffer(st.buffer, offset: 0, index: 0)
                enc.setBytes(&off, length: 32, index: 1)
                enc.setBuffer(x, offset: 0, index: 2)
                enc.setBuffer(y, offset: 0, index: 3)
                var gp = [Int32(T), 2048, 2048, 2048, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                enc.setBytes(&gp, length: 64, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: 2048 / bn, height: (T + bm - 1) / bm, depth: 1), threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / 32)
        }
        print(String(format: "T %4d: %.1f µs a 2048x2048 GEMM, %.2f TFLOPS, %.1f µs a token", T, best * 1e6, 2.0 * Double(T) * 2048 * 2048 / best / 1e12, best * 1e6 / Double(T)))
    }
}

/// The W4A8 GEMM (int8 activations quantized once beforehand) on the o projections, by batch size.
func runGEMMW4A8(_ ctx: MetalContext, _ st: SafeTensors) throws {
    let src = try Engine.kernelSource("w4a8")
    let pq = try ctx.pipeline("q8_act", source: src), p = try ctx.pipeline("gemm_w4a8", source: src, macros: ["EPI": 0])
    for T in [64, 256, 1024] {
        let x = ctx.buffer([UInt16](repeating: BF16.bits(0.01), count: T * 2048)), y = ctx.buffer(UInt16.self, count: T * 2048)
        let xq = ctx.buffer(Int8.self, count: T * 2048), sx = ctx.buffer(Float.self, count: T * 32), xz = ctx.buffer(Float.self, count: T * 32)
        var best = Double.infinity
        for rep in 0 ..< 4 {
            let cb = ctx.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
            if rep == 0 {
                enc.setComputePipelineState(pq)
                enc.setBuffer(x, offset: 0, index: 0); enc.setBuffer(xq, offset: 0, index: 1)
                enc.setBuffer(sx, offset: 0, index: 2); enc.setBuffer(xz, offset: 0, index: 3)
                var kl = SIMD2<Int32>(2048, 2048)
                enc.setBytes(&kl, length: 8, index: 4)
                enc.dispatchThreads(MTLSize(width: 1024, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                continue
            }
            for l in 0 ..< 32 {
                let pre = "transformer.h.\(l).attn.c_proj."
                var off = SIMD3<UInt64>(UInt64(try st.tensor(pre + "weight").offset), UInt64(try st.tensor(pre + "scales").offset), UInt64(try st.tensor(pre + "biases").offset))
                enc.setComputePipelineState(p)
                enc.setBuffer(st.buffer, offset: 0, index: 0)
                enc.setBytes(&off, length: 32, index: 1)
                enc.setBuffer(xq, offset: 0, index: 2)
                enc.setBuffer(y, offset: 0, index: 3)
                var gp = [Int32(T), 2048, 2048, 2048, 2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                enc.setBytes(&gp, length: 64, index: 4)
                for i in 5 ... 7 { enc.setBuffer(xq, offset: 0, index: i) }
                enc.setBuffer(sx, offset: 0, index: 8)
                enc.setBuffer(xz, offset: 0, index: 9)
                enc.dispatchThreadgroups(MTLSize(width: 32, height: (T + 63) / 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / 32)
        }
        print(String(format: "W4A8 T %4d: %.1f µs a 2048x2048 GEMM, %.2f TOPS, %.2f µs a token", T, best * 1e6, 2.0 * Double(T) * 2048 * 2048 / best / 1e12, best * 1e6 / Double(T)))
    }
}
