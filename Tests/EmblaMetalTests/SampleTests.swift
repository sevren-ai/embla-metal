import Metal
import XCTest

@testable import EmblaMetal

/// The GPU sampler (misc.metal's `sample`) against the distribution it should draw from.
final class SampleTests: XCTestCase {
    /// Ten likely tokens over a flat tail: draws for many positions, against softmax(c / T)
    /// cut to its top-p nucleus.
    func testDistribution() throws {
        let ctx = try MetalContext()
        let p = try ctx.pipeline("sample", source: Engine.kernelSource("misc"))
        let V = Embla.vocab, n = 6000
        // raw logits whose softcapped values are 3.0, 2.7, ... (the tail at -10)
        let capped = (0 ..< V).map { i -> Float in i < 10 ? 3.0 - 0.3 * Float(i) : -10 }
        let raw = capped.map { 15 * atanh($0 / 15) }
        let logits = ctx.buffer(raw)
        let tokens = ctx.buffer(UInt32.self, count: n + 2), greedy = ctx.buffer(UInt32.self, count: n + 3)
        let scratch = ctx.buffer(Float.self, count: V)
        for (T, topP) in [(Float(1), Float(1)), (0.6, 0.9), (1.0, 0.95), (3.0, 0.9)] {
            let cb = ctx.queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(p)
            enc.setBuffer(logits, offset: 0, index: 0)
            enc.setBuffer(tokens, offset: 0, index: 1)
            var v = Int32(V)
            enc.setBytes(&v, length: 4, index: 2)
            enc.setBuffer(greedy, offset: 4, index: 3)
            var sp = (1 / T, topP, UInt32(1234))
            enc.setBytes(&sp, length: 12, index: 5)
            enc.setBuffer(scratch, offset: 0, index: 6)
            for pos in 0 ..< n {
                var pp = SIMD2<Int32>(Int32(pos), 0)
                enc.setBytes(&pp, length: 8, index: 4)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let got = greedy.contents().assumingMemoryBound(to: UInt32.self) + 1
            var counts = [Int](repeating: 0, count: 11)
            for i in 0 ..< n { counts[min(10, Int(got[i]))] += 1 }
            // the expected distribution
            let w = capped.map { Double(exp(($0 - 3) / T)) }
            let z = w.reduce(0, +)
            var keep = 0, mass = 0.0
            while mass < Double(topP) * z && keep < V { mass += w[keep]; keep += 1 }
            let q = (0 ..< 10).map { $0 < keep ? w[$0] / mass : 0 }
            for i in 0 ..< 10 {
                let f = Double(counts[i]) / Double(n)
                let sd = (q[i] * (1 - q[i]) / Double(n)).squareRoot()
                XCTAssertLessThan(abs(f - q[i]), 5 * sd + 1e-9, "T \(T) top-p \(topP): token \(i) drawn \(f), expected \(q[i])")
            }
            if keep <= 10 { XCTAssertEqual(counts[10] + counts[keep...].dropLast().reduce(0, +), 0, "draws outside the nucleus") }
            print("T \(T) top-p \(topP): nucleus \(keep), drawn \(counts.map { String(format: "%.3f", Double($0) / Double(n)) })")
        }
    }
}
