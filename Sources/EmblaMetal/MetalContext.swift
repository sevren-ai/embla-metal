import Foundation
import Metal

/// The GPU, its command queue and the compiled kernels.
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var libraries: [String: MTLLibrary] = [:]

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EmblaError.noMetal
        }
        self.device = device
        self.queue = queue
    }

    /// Whether the GPU has neural accelerators (the M5 and A19, Metal's apple10 family) for the
    /// Metal 4 tensor ops. On macOS 26 those kernels also compile on earlier GPUs but give wrong
    /// results there, so a kernel that compiles is not enough. EMBLA_NAX=0 or 1 overrides it.
    public var hasNeuralAccelerators: Bool {
        switch ProcessInfo.processInfo.environment["EMBLA_NAX"] {
        case "0": return false
        case "1": return true
        default:
            if #available(macOS 26.0, iOS 26.0, *) { return device.supportsFamily(.apple10) }
            return false
        }
    }

    /// The pipeline of `function` in the Metal source `source`, compiled once per (source, macros,
    /// constants). `macros` are preprocessor definitions (sizes that arrays need at compile time).
    public func pipeline(
        _ function: String, source: String, constants: [String: Int] = [:], macros: [String: Int] = [:]
    ) throws -> MTLComputePipelineState {
        func describe(_ d: [String: Int]) -> String { d.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: ",") }
        let libraryKey = describe(macros) + "|" + source
        let key = function + "|" + describe(constants) + "|" + libraryKey
        if let p = pipelines[key] { return p }
        let library: MTLLibrary
        if let l = libraries[libraryKey] {
            library = l
        } else {
            let options = MTLCompileOptions()
            options.mathMode = .fast
            // Metal 4 for the tensor ops (MetalPerformancePrimitives, macOS 26), 3.2 otherwise
            options.languageVersion = .version3_2
            if source.contains("MetalPerformancePrimitives") {
                guard #available(macOS 26.0, iOS 26.0, *) else { throw EmblaError.noMetal }
                options.languageVersion = .version4_0
            }
            options.preprocessorMacros = macros.mapValues { NSNumber(value: $0) }
            library = try device.makeLibrary(source: source, options: options)
            libraries[libraryKey] = library
        }
        let values = MTLFunctionConstantValues()
        for (name, value) in constants {
            var v = Int32(value)
            values.setConstantValue(&v, type: .int, withName: name)
        }
        let function = try library.makeFunction(name: function, constantValues: values)
        let p = try device.makeComputePipelineState(function: function)
        pipelines[key] = p
        return p
    }

    /// A GPU buffer of `count` elements of `T`, shared with the CPU.
    public func buffer<T>(_ type: T.Type, count: Int) -> MTLBuffer {
        device.makeBuffer(length: max(1, count) * MemoryLayout<T>.stride, options: .storageModeShared)!
    }

    public func buffer<T>(_ values: [T]) -> MTLBuffer {
        values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max(1, $0.count), options: .storageModeShared)! }
    }
}

public enum EmblaError: Error, CustomStringConvertible {
    case noMetal
    case badFile(String)
    case missingTensor(String)

    public var description: String {
        switch self {
        case .noMetal: return "no Metal device"
        case .badFile(let s): return "bad file: \(s)"
        case .missingTensor(let s): return "missing tensor: \(s)"
        }
    }
}
