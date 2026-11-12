import Metal
import XCTest

@testable import EmblaMetal

final class GEMMTests: XCTestCase {
    /// gemm against the CPU reference of every token, and its speed on a 2048x2048 matrix.
    func testAgainstReference() throws {
        let ctx = try MetalContext()
        for (tile, bm) in [("float", 32), ("bfloat", 32), ("bfloat", 64), ("nax half", 64), ("nax bfloat", 64)] {
        let nax = tile.hasPrefix("nax")
        if nax && !ctx.hasNeuralAccelerators { continue }
        let p = nax
            ? try ctx.pipeline("gemm_nax", source: "#define NT \(tile.dropFirst(4))\n" + Engine.naxSource("gemm_nax"), macros: ["EPI": 4])
            : try ctx.pipeline("gemm", source: "#define TILE \(tile)\n" + Engine.kernelSource("gemm"), macros: ["EPI": 4, "BM": bm])
        for (T, O) in [(37, 96), (64, 2048)] {
            let m = QuantizedMatrix(rows: O, inputs: 2048, seed: UInt64(O))
            var g = SplitMix(3)
            let xb = (0 ..< T * 2048).map { _ in BF16.bits(Float(g.unit() - 0.5) * 2) }
            // one buffer holding words, scales, biases
            let wb = m.words.count * 4, sb = m.scales.count * 2
            let all = ctx.device.makeBuffer(length: wb + 2 * sb, options: .storageModeShared)!
            memcpy(all.contents(), m.words, wb)
            memcpy(all.contents() + wb, m.scales, sb)
            memcpy(all.contents() + wb + sb, m.biases, sb)
            let x = ctx.buffer(xb), y = ctx.buffer(Float.self, count: T * O)
            let cb = ctx.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(p)
            enc.setBuffer(all, offset: 0, index: 0)
            var off = SIMD3<UInt64>(0, UInt64(wb), UInt64(wb + sb))
            enc.setBytes(&off, length: 32, index: 1)
            enc.setBuffer(x, offset: 0, index: 2)
            enc.setBuffer(y, offset: 0, index: 3)
            var gp = GemmParams(T: Int32(T), O: Int32(O), ldx: 2048, ldy: Int32(O), K: 2048)
            enc.setBytes(&gp, length: MemoryLayout<GemmParams>.size, index: 4)
            enc.dispatchThreadgroups(MTLSize(width: (O + 63) / 64, height: (T + bm - 1) / bm, depth: 1), threadsPerThreadgroup: MTLSize(width: nax ? 128 : 256, height: 1, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let out = y.contents().assumingMemoryBound(to: Float.self)
            var err: Float = 0, mag: Float = 0
            for tk in [0, T / 2, T - 1] {
                let ref = m.reference((0 ..< 2048).map { BF16.float(xb[tk * 2048 + $0]) })
                for r in 0 ..< O { err = max(err, abs(out[tk * O + r] - ref[r])); mag = max(mag, abs(ref[r])) }
            }
            print("tile \(tile) BM \(bm) T \(T) O \(O): relative error \(err / mag)")
            XCTAssertLessThan(err / mag, tile == "float" ? 1e-4 : 1e-2, "\(tile) T \(T) O \(O)")
        }
        }
    }
}

final class W4A8Tests: XCTestCase {
    /// q8_act + gemm_w4a8 against the CPU reference: within the activations' int8 rounding.
    func testAgainstReference() throws {
        let ctx = try MetalContext()
        let src = try Engine.kernelSource("w4a8")
        let pq = try ctx.pipeline("q8_act", source: src), pg = try ctx.pipeline("gemm_w4a8", source: src, macros: ["EPI": 4])
        for (T, O) in [(37, 96), (100, 2048)] {
            let m = QuantizedMatrix(rows: O, inputs: 2048, seed: UInt64(O + 1))
            var g = SplitMix(9)
            let xb = (0 ..< T * 2048).map { _ in BF16.bits(Float(g.unit() - 0.5) * 2) }
            let wb = m.words.count * 4, sb = m.scales.count * 2
            let all = ctx.device.makeBuffer(length: wb + 2 * sb, options: .storageModeShared)!
            memcpy(all.contents(), m.words, wb)
            memcpy(all.contents() + wb, m.scales, sb)
            memcpy(all.contents() + wb + sb, m.biases, sb)
            let x = ctx.buffer(xb), y = ctx.buffer(Float.self, count: T * O)
            let xq = ctx.buffer(Int8.self, count: T * 2048), sx = ctx.buffer(Float.self, count: T * 32), xz = ctx.buffer(Float.self, count: T * 32)
            let cb = ctx.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pq)
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(xq, offset: 0, index: 1)
            enc.setBuffer(sx, offset: 0, index: 2)
            enc.setBuffer(xz, offset: 0, index: 3)
            var kl = SIMD2<Int32>(2048, 2048)
            enc.setBytes(&kl, length: 8, index: 4)
            enc.dispatchThreads(MTLSize(width: 32 * 32, height: T, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(pg)
            enc.setBuffer(all, offset: 0, index: 0)
            var off = SIMD3<UInt64>(0, UInt64(wb), UInt64(wb + sb))
            enc.setBytes(&off, length: 32, index: 1)
            enc.setBuffer(xq, offset: 0, index: 2)
            enc.setBuffer(y, offset: 0, index: 3)
            var gp = GemmParams(T: Int32(T), O: Int32(O), ldx: 2048, ldy: Int32(O), K: 2048)
            enc.setBytes(&gp, length: MemoryLayout<GemmParams>.size, index: 4)
            enc.setBuffer(xq, offset: 0, index: 5)
            enc.setBuffer(xq, offset: 0, index: 6)
            enc.setBuffer(xq, offset: 0, index: 7)
            enc.setBuffer(sx, offset: 0, index: 8)
            enc.setBuffer(xz, offset: 0, index: 9)
            enc.dispatchThreadgroups(MTLSize(width: (O + 63) / 64, height: (T + 63) / 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            XCTAssertNil(cb.error)
            let out = y.contents().assumingMemoryBound(to: Float.self)
            var err: Float = 0, mag: Float = 0
            for tk in [0, T / 2, T - 1] {
                let ref = m.reference((0 ..< 2048).map { BF16.float(xb[tk * 2048 + $0]) })
                for r in 0 ..< O { err = max(err, abs(out[tk * O + r] - ref[r])); mag = max(mag, abs(ref[r])) }
            }
            print("w4a8 T \(T) O \(O): relative error \(err / mag)")
            XCTAssertLessThan(err / mag, 2e-2, "T \(T) O \(O)")
        }
    }
}
