import Metal
import XCTest

@testable import EmblaMetal

/// Random 4-bit weights in MLX's layout, with a CPU reference of y = W x.
struct QuantizedMatrix {
    let rows: Int, inputs: Int
    var words: [UInt32]
    var scales: [UInt16]
    var biases: [UInt16]

    init(rows: Int, inputs: Int, seed: UInt64) {
        self.rows = rows
        self.inputs = inputs
        var g = SplitMix(seed)
        words = (0 ..< rows * inputs / 8).map { _ in UInt32(truncatingIfNeeded: g.next()) }
        scales = (0 ..< rows * inputs / 64).map { _ in BF16.bits(Float(g.unit()) * 0.02 + 0.001) }
        biases = (0 ..< rows * inputs / 64).map { _ in BF16.bits(Float(g.unit() - 0.5) * 0.1) }
    }

    func reference(_ x: [Float]) -> [Float] {
        (0 ..< rows).map { r in
            var acc: Double = 0
            for i in 0 ..< inputs {
                let q = (words[(r * inputs + i) / 8] >> (4 * UInt32(i % 8))) & 0xF
                let g = (r * inputs + i) / 64
                acc += Double(BF16.float(scales[g]) * Float(q) + BF16.float(biases[g])) * Double(x[i])
            }
            return Float(acc)
        }
    }
}

struct SplitMix {
    var s: UInt64
    init(_ seed: UInt64) { s = seed }
    mutating func next() -> UInt64 {
        s &+= 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }
}

final class GEMV4Tests: XCTestCase {
    func testAgainstReference() throws {
        let ctx = try MetalContext()
        for (rows, inputs) in [(512, 2048), (1024, 2048), (2048, 1024), (37, 2048)] {
            for config in [GEMV4.Config(rows: 4, simdgroups: 8), GEMV4.Config(rows: 8, simdgroups: 4), GEMV4.Config(rows: 1, simdgroups: 2)] {
                let m = QuantizedMatrix(rows: rows, inputs: inputs, seed: UInt64(rows * 7 + inputs))
                var g = SplitMix(42)
                let xb = (0 ..< inputs).map { _ in BF16.bits(Float(g.unit() - 0.5) * 2) }
                let x = xb.map(BF16.float)
                let y = ctx.buffer(UInt16.self, count: rows)
                let cb = ctx.queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                try GEMV4.encode(
                    ctx, enc, w: (ctx.buffer(m.words), 0), scales: (ctx.buffer(m.scales), 0),
                    biases: (ctx.buffer(m.biases), 0), x: (ctx.buffer(xb), 0), y: (y, 0),
                    rows: rows, inputs: inputs, config: config)
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                let out = UnsafeBufferPointer(start: y.contents().assumingMemoryBound(to: UInt16.self), count: rows).map(BF16.float)
                let ref = m.reference(x)
                let scale = ref.map(abs).max()!
                let err = zip(out, ref).map { abs($0 - $1) }.max()!
                XCTAssertLessThan(err / scale, 1e-2, "\(rows)x\(inputs) \(config)")
            }
        }
    }
}
