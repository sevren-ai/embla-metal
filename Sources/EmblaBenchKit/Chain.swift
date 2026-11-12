import EmblaMetal
import Foundation
import Metal

/// `embla-bench chain`: GPU µs per dependent dispatch (each reads what the one before wrote), for
/// the ways an encoder can order them, at a tiny grid and at a GEMV-sized one. A decode step is a
/// chain of ~220 such dispatches, so on a GPU where each costs microseconds this is a floor.
func runChain(_ args: [String]) throws {
    let ctx = try MetalContext()
    let src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void step(device float* a [[buffer(0)]], uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
        a[i] = a[(i + 1) % n] * 0.5f + 1.0f;
    }
    """
    let p = try ctx.pipeline("step", source: src)
    let buf = ctx.buffer(Float.self, count: 1 << 20)
    let d = ctx.device
    print("counter sampling: dispatch \(d.supportsCounterSampling(.atDispatchBoundary)), stage \(d.supportsCounterSampling(.atStageBoundary)), sets \((d.counterSets ?? []).map { $0.name })")
    let n = 400
    for (grid, label) in [(256, "1 threadgroup"), (256 * 256, "256 threadgroups")] {
        func time(_ name: String, _ make: (MTLCommandBuffer) -> MTLComputeCommandEncoder, _ between: (MTLComputeCommandEncoder) -> Void) {
            var best = Double.infinity
            for _ in 0 ..< 5 {
                let cb = ctx.queue.makeCommandBuffer()!
                let enc = make(cb)
                enc.setComputePipelineState(p)
                enc.setBuffer(buf, offset: 0, index: 0)
                for i in 0 ..< n {
                    if i > 0 { between(enc) }
                    enc.dispatchThreads(MTLSize(width: grid, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                best = min(best, (cb.gpuEndTime - cb.gpuStartTime) / Double(n) * 1e6)
            }
            print(String(format: "  %-16@ %-40@ %6.2f µs a dispatch", label as NSString, name as NSString, best))
        }
        time("concurrent, no barrier (independent)", { $0.makeComputeCommandEncoder(dispatchType: .concurrent)! }) { _ in }
        time("concurrent + barrier(.buffers)", { $0.makeComputeCommandEncoder(dispatchType: .concurrent)! }) { $0.memoryBarrier(scope: .buffers) }
        time("concurrent + barrier(resources)", { $0.makeComputeCommandEncoder(dispatchType: .concurrent)! }) { $0.memoryBarrier(resources: [buf]) }
        time("serial encoder", { $0.makeComputeCommandEncoder(dispatchType: .serial)! }) { _ in }
    }
}
