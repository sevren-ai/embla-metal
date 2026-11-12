import Metal

/// The decode step's compute encoder. Normally one encoder with a memory barrier between dependent
/// dispatches; while profiling (`Profiler`), each barrier ends the encoder and starts another, its
/// start and end timestamped and labelled with the part of the step it ran, so that every part's
/// GPU time is measured in the real step on real data.
final class StepEncoder {
    private(set) var e: MTLComputeCommandEncoder
    let cb: MTLCommandBuffer
    let serial: Bool
    let profiler: Profiler?
    /// The part the next dispatches belong to (set by the engine before it dispatches).
    var label = "other"
    private var labels: [String] = []

    init(_ cb: MTLCommandBuffer, serial: Bool, profiler: Profiler?) {
        self.cb = cb
        self.serial = serial
        self.profiler = profiler
        try? profiler?.begin()
        e = Self.make(cb, serial: serial, profiler: profiler)
    }

    private static func make(_ cb: MTLCommandBuffer, serial: Bool, profiler: Profiler?) -> MTLComputeCommandEncoder {
        guard let p = profiler else { return cb.makeComputeCommandEncoder(dispatchType: serial ? .serial : .concurrent)! }
        let d = MTLComputePassDescriptor()
        d.dispatchType = serial ? .serial : .concurrent
        let i = p.next()
        d.sampleBufferAttachments[0].sampleBuffer = p.samples
        d.sampleBufferAttachments[0].startOfEncoderSampleIndex = i
        d.sampleBufferAttachments[0].endOfEncoderSampleIndex = i + 1
        return cb.makeComputeCommandEncoder(descriptor: d)!
    }

    func setComputePipelineState(_ p: MTLComputePipelineState) { e.setComputePipelineState(p) }
    func setBuffer(_ b: MTLBuffer?, offset: Int, index: Int) { e.setBuffer(b, offset: offset, index: index) }
    func setBytes(_ p: UnsafeRawPointer, length: Int, index: Int) { e.setBytes(p, length: length, index: index) }
    func dispatchThreadgroups(_ g: MTLSize, threadsPerThreadgroup t: MTLSize) {
        labels.append(label)
        e.dispatchThreadgroups(g, threadsPerThreadgroup: t)
    }
    func dispatchThreads(_ g: MTLSize, threadsPerThreadgroup t: MTLSize) {
        labels.append(label)
        e.dispatchThreads(g, threadsPerThreadgroup: t)
    }

    /// A barrier between dependent dispatches (none in a serial encoder); a new encoder while profiling.
    func barrier() {
        if let p = profiler {
            guard !labels.isEmpty else { return }
            p.record(labels)
            labels = []
            e.endEncoding()
            e = Self.make(cb, serial: serial, profiler: p)
        } else if !serial {
            e.memoryBarrier(scope: .buffers)
        }
    }

    func endEncoding() {
        profiler?.record(labels)
        labels = []
        e.endEncoding()
    }
}

/// Timestamps at the start and end of every encoder of profiled steps, and what each one ran: a
/// sample buffer a command buffer (on the A20 Pro a buffer shared by several kept only the first's).
public final class Profiler {
    let device: MTLDevice
    let set: MTLCounterSet
    private var buffers: [(MTLCounterSampleBuffer, Int)] = []
    private var parts: [[String]] = []
    private var owner: [Int] = []

    init(device: MTLDevice) throws {
        self.device = device
        guard let set = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else { throw EmblaError.noMetal }
        self.set = set
    }

    /// Starts a new command buffer's samples.
    func begin() throws {
        let d = MTLCounterSampleBufferDescriptor()
        d.counterSet = set
        d.sampleCount = 4096
        d.storageMode = .shared
        buffers.append((try device.makeCounterSampleBuffer(descriptor: d), 0))
    }

    var samples: MTLCounterSampleBuffer { buffers[buffers.count - 1].0 }
    func next() -> Int { buffers[buffers.count - 1].1 += 2; return buffers[buffers.count - 1].1 - 2 }
    func record(_ labels: [String]) { parts.append(labels); owner.append(buffers.count - 1) }

    /// µs a part took (summed over its encoders) and the time between encoders, over the steps
    /// sampled, divided by `steps`.
    func report(steps: Int, ticksToNs: Double) -> (parts: [(String, Double)], gaps: Double, total: Double) {
        let ts = buffers.map { b, used -> [UInt64] in
            let data = (try? b.resolveCounterRange(0 ..< used)) ?? Data()
            return data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
        }
        var sums: [String: Double] = [:], gaps = 0.0, total = 0.0
        var index = [Int](repeating: 0, count: buffers.count), lastEnd: UInt64 = 0
        for (k, labels) in parts.enumerated() {
            let b = owner[k], i = index[b]
            index[b] += 2
            guard i + 1 < ts[b].count else { continue }
            let s0 = ts[b][i], s1 = ts[b][i + 1]
            guard s0 > 0, s1 >= s0 else { continue }
            let us = Double(s1 - s0) * ticksToNs / 1e3
            let name = Set(labels).count == 1 ? labels[0] : labels.joined(separator: "+")
            sums[name, default: 0] += us
            total += us
            if lastEnd > 0, s0 >= lastEnd { gaps += Double(s0 - lastEnd) * ticksToNs / 1e3 }
            lastEnd = s1
        }
        return (sums.sorted { $0.value > $1.value }.map { ($0.key, $0.value / Double(steps)) }, gaps / Double(steps), total / Double(steps))
    }
}
