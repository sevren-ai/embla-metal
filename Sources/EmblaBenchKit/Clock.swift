import EmblaMetal
import Foundation
import Metal

/// A GPU clock probe: one thread's chain of dependent multiply-adds, so its GPU time is inversely
/// the shader clock (relative: compare runs, not absolute MHz).
final class ClockProbe {
    let ctx: MetalContext, p: MTLComputePipelineState, buf: MTLBuffer
    init(_ ctx: MetalContext) throws {
        self.ctx = ctx
        p = try ctx.pipeline("chain", source: """
        #include <metal_stdlib>
        using namespace metal;
        kernel void chain(device float* a [[buffer(0)]]) {
            float x = a[0];
            for (int i = 0; i < 200000; i++) x = fma(x, 0.999999f, 1e-7f);
            a[1] = x;
        }
        """)
        buf = ctx.buffer(Float.self, count: 4)
    }
    /// µs of GPU time for the chain.
    func run() -> Double {
        let cb = ctx.queue.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return (cb.gpuEndTime - cb.gpuStartTime) * 1e6
    }
}
