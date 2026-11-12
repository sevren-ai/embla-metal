import Foundation
import Metal

/// A safetensors file, memory-mapped and visible to the GPU without a copy: one `MTLBuffer` over the
/// whole file, and each tensor an offset into it.
public final class SafeTensors: @unchecked Sendable {
    public struct Tensor {
        public let dtype: String
        public let shape: [Int]
        /// Byte offset of the tensor's data in `buffer`.
        public let offset: Int
        public let byteCount: Int
    }

    public let url: URL
    public let buffer: MTLBuffer
    public let tensors: [String: Tensor]
    private let base: UnsafeMutableRawPointer
    private let mappedLength: Int

    public init(url: URL, device: MTLDevice) throws {
        self.url = url
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw EmblaError.badFile(url.path) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0 else { throw EmblaError.badFile(url.path) }
        let size = Int(st.st_size)
        // Metal needs a page-aligned pointer and a length in whole pages; mmap gives the first and
        // maps the tail of the last page as zeros.
        let page = Int(getpagesize())
        mappedLength = (size + page - 1) / page * page
        guard let p = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
            throw EmblaError.badFile("mmap \(url.path)")
        }
        base = p
        guard let buffer = device.makeBuffer(bytesNoCopy: p, length: mappedLength, options: .storageModeShared, deallocator: nil) else {
            munmap(p, mappedLength)
            throw EmblaError.badFile("no Metal buffer over \(url.path)")
        }
        self.buffer = buffer

        let headerLength = Int(p.loadUnaligned(as: UInt64.self))
        let header = Data(bytes: p + 8, count: headerLength)
        guard let json = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw EmblaError.badFile("header of \(url.path)")
        }
        var tensors: [String: Tensor] = [:]
        for (name, v) in json where name != "__metadata__" {
            guard let d = v as? [String: Any], let dtype = d["dtype"] as? String,
                  let shape = d["shape"] as? [Int], let off = d["data_offsets"] as? [Int] else { continue }
            tensors[name] = Tensor(dtype: dtype, shape: shape, offset: 8 + headerLength + off[0], byteCount: off[1] - off[0])
        }
        self.tensors = tensors
    }

    deinit { munmap(base, mappedLength) }

    public subscript(name: String) -> Tensor? { tensors[name] }

    public func tensor(_ name: String) throws -> Tensor {
        guard let t = tensors[name] else { throw EmblaError.missingTensor(name) }
        return t
    }
}
