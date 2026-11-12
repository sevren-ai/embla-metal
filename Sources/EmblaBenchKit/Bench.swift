import EmblaMetal
import Foundation
import Metal

/// GPU time per call of a kernel chained `n` times in one encoder, each call depending on the one
/// before (a memory barrier between them), over weights spread across enough memory that the reads
/// come from DRAM. Best of `reps` command buffers.
func gpuMicros(_ ctx: MetalContext, n: Int, reps: Int = 5, _ body: (MTLComputeCommandEncoder, Int) throws -> Void) rethrows -> Double {
    var best = Double.infinity
    for _ in 0 ..< reps {
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        for i in 0 ..< n {
            try body(enc, i)
            enc.memoryBarrier(scope: .buffers)
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / Double(n) * 1e6)
    }
    return best
}

/// GB/s: measured on the M5 Pro with MLX; ~103 on the iPhone 18 Pro (EMBLA_BANDWIDTH to override).
let bandwidth = Double(ProcessInfo.processInfo.environment["EMBLA_BANDWIDTH"] ?? "") ?? 281.0

func benchGEMV(_ ctx: MetalContext) throws {
    print("4-bit GEMV, one token (GPU µs per call; ideal = bytes at \(Int(bandwidth)) GB/s)")
    let shapes = [("q / o projection", 2048, 2048), ("k or v (sliding)", 512, 2048), ("shared up", 1024, 2048),
                  ("shared down", 2048, 1024), ("q+k+v (sliding)", 3072, 2048)]
    for (name, rows, inputs) in shapes {
        let bytes = rows * inputs / 2 + rows * inputs / 64 * 4
        let copies = max(4, 512 << 20 / bytes)   // ~512 MB of weights
        let w = ctx.device.makeBuffer(length: copies * rows * inputs / 2, options: .storageModeShared)!
        let s = ctx.device.makeBuffer(length: copies * rows * inputs / 64 * 2, options: .storageModeShared)!
        let b = ctx.device.makeBuffer(length: copies * rows * inputs / 64 * 2, options: .storageModeShared)!
        memset(w.contents(), 0x5A, w.length)
        memset(s.contents(), 0x3C, s.length)
        memset(b.contents(), 0x00, b.length)
        let x = ctx.buffer([UInt16](repeating: BF16.bits(0.01), count: inputs))
        let y = ctx.buffer(UInt16.self, count: rows)
        let ideal = Double(bytes) / (bandwidth * 1e3)
        var line = String(format: "  %-18@ %5dx%-5d ideal %6.2f µs |", name as NSString, rows, inputs, ideal)
        for config in [GEMV4.Config(rows: 1, simdgroups: 8), GEMV4.Config(rows: 2, simdgroups: 8),
                       GEMV4.Config(rows: 4, simdgroups: 8), GEMV4.Config(rows: 4, simdgroups: 4), GEMV4.Config(rows: 8, simdgroups: 4)] {
            let us = try gpuMicros(ctx, n: copies) { enc, i in
                try GEMV4.encode(ctx, enc, w: (w, i * rows * inputs / 2), scales: (s, i * rows * inputs / 64 * 2),
                                 biases: (b, i * rows * inputs / 64 * 2), x: (x, 0), y: (y, 0),
                                 rows: rows, inputs: inputs, config: config)
            }
            line += String(format: " r%d×%d %6.2f (%3.0f%%)", config.rows, config.simdgroups, us, ideal / us * 100)
        }
        // the same matrix every call: its bytes come from the cache after the first
        let cached = try gpuMicros(ctx, n: 64) { enc, _ in
            try GEMV4.encode(ctx, enc, w: (w, 0), scales: (s, 0), biases: (b, 0), x: (x, 0), y: (y, 0),
                             rows: rows, inputs: inputs, config: GEMV4.Config(rows: 4, simdgroups: 8))
        }
        line += String(format: " | cached %6.2f", cached)
        print(line)
    }
}

/// The model's own projections: layer 0's q projection against a CPU reference from the mapped
/// bytes, then chains over every layer's matrices of one kind (reads from DRAM, as in decode).
func benchRealWeights(_ ctx: MetalContext, _ path: String) throws {
    let t0 = Date()
    let st = try SafeTensors(url: URL(fileURLWithPath: path), device: ctx.device)
    print(String(format: "mapped %@ (%.2f GB, %d tensors) in %.0f ms", (path as NSString).lastPathComponent,
                 Double(st.buffer.length) / 1e9, st.tensors.count, Date().timeIntervalSince(t0) * 1e3))
    func q(_ name: String) throws -> ((MTLBuffer, Int), (MTLBuffer, Int), (MTLBuffer, Int), Int, Int) {
        let w = try st.tensor(name + ".weight"), s = try st.tensor(name + ".scales"), b = try st.tensor(name + ".biases")
        return ((st.buffer, w.offset), (st.buffer, s.offset), (st.buffer, b.offset), w.shape[0], w.shape[1] * 8)
    }

    // correctness on real weights
    let (w, s, b, rows, inputs) = try q("transformer.h.0.attn.c_q")
    var g = UInt64(7)
    let xb: [UInt16] = (0 ..< inputs).map { _ in g = g &* 6364136223846793005 &+ 1; return BF16.bits(Float(Int(g >> 33) % 2001 - 1000) / 1000) }
    let y = ctx.buffer(UInt16.self, count: rows)
    let cb = ctx.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
    try GEMV4.encode(ctx, enc, w: w, scales: s, biases: b, x: (ctx.buffer(xb), 0), y: (y, 0), rows: rows, inputs: inputs,
                     config: GEMV4.Config(rows: 4, simdgroups: 4))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let words = (st.buffer.contents() + w.1).assumingMemoryBound(to: UInt32.self)
    let sc = (st.buffer.contents() + s.1).assumingMemoryBound(to: UInt16.self)
    let bi = (st.buffer.contents() + b.1).assumingMemoryBound(to: UInt16.self)
    var maxErr: Float = 0, maxRef: Float = 0
    let out = y.contents().assumingMemoryBound(to: UInt16.self)
    for r in 0 ..< rows {
        var acc: Double = 0
        for i in 0 ..< inputs {
            let qv = Float((words[(r * inputs + i) / 8] >> (4 * UInt32(i % 8))) & 0xF)
            let gi = (r * inputs + i) / 64
            acc += Double(BF16.float(sc[gi]) * qv + BF16.float(bi[gi])) * Double(BF16.float(xb[i]))
        }
        maxErr = max(maxErr, abs(BF16.float(out[r]) - Float(acc)))
        maxRef = max(maxRef, abs(Float(acc)))
    }
    print(String(format: "layer 0 c_q on real weights: max error %.4g (max |y| %.3g, relative %.2e)", maxErr, maxRef, maxErr / maxRef))

    // chains over all 32 layers
    let x = ctx.buffer([UInt16](repeating: BF16.bits(0.01), count: 2048))
    let yy = ctx.buffer(UInt16.self, count: 4096)
    for kind in ["attn.c_q", "attn.c_proj"] {
        let mats = try (0 ..< 32).map { try q("transformer.h.\($0).\(kind)") }
        let bytes = mats[0].3 * mats[0].4 / 2 + mats[0].3 * mats[0].4 / 64 * 4
        let us = try gpuMicros(ctx, n: 32 * 8) { enc, i in
            let m = mats[i % 32]
            try GEMV4.encode(ctx, enc, w: m.0, scales: m.1, biases: m.2, x: (x, 0), y: (yy, 0), rows: m.3, inputs: m.4,
                             config: GEMV4.Config(rows: 4, simdgroups: 4))
        }
        print(String(format: "  %-12@ x32 layers: %.2f µs per call, %.0f%% of %.0f GB/s", kind as NSString, us,
                     Double(bytes) / (bandwidth * 1e3) / us * 100, bandwidth))
    }
}

/// Runs one `embla-bench` command (`args` without the program name), as the command line does.
public func runBench(_ args: [String]) throws {
    let ctx = try MetalContext()
    print("device: \(ctx.device.name)")
    let args = ["embla-bench"] + args
    if args.count > 2, args[1] == "gemm" {
        try runGEMM(Array(args[2...]))
    } else if args.count > 6, args[1] == "gen" {
        try runGen(Array(args[2...]))
    } else if args.count > 2, args[1] == "gemv" {
        try runGEMV(Array(args[2...]))
    } else if args.count > 2, args[1] == "speed" {
        try runSpeed(Array(args[2...]))
    } else if args.count > 2, args[1] == "det" {
        try runDet(Array(args[2...]))
    } else if args.count > 1, args[1] == "chain" {
        try runChain([])
    } else if args.count > 2, args[1] == "e2e" {
        try runE2E(Array(args[2...]))
    } else if args.count > 4, args[1] == "check" {
        try runCheck(Array(args[2...]))
    } else if args.count > 2, args[1] == "weights" {
        try benchRealWeights(ctx, args[2])
    } else {
        try benchGEMV(ctx)
    }
}
