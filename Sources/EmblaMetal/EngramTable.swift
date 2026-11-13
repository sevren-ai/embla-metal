import Foundation

/// The engram row table (`engram_table.safetensors`, 1.69 GB), read from the SSD a token at a time.
///
/// A token reads 16 rows: for each n-gram order (2 and 3) and each of 2 hash groups, 4 rows of one
/// page (page-local multi-head hashing). The file stores the rows'
/// 4-bit words, fp16 biases and fp16 scales as three arrays, so a page is three `pread`s: the span
/// of its 4 rows in each array. The 4 pages are read in parallel.
public final class EngramTable: @unchecked Sendable {
    public static let pages = 209_716, rowsPerPage = 56, orders: [UInt64] = [2, 3]
    public static let heads = 8, pageHeads = 4, dim = 128
    public static let rowsPerToken = 16

    private let fd: Int32
    private let weightOffset: Int, scalesOffset: Int, biasesOffset: Int
    private let units: [UInt64]

    public init(url: URL) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw EmblaError.badFile(url.path) }
        self.fd = fd
        var n: UInt64 = 0
        guard pread(fd, &n, 8, 0) == 8 else { throw EmblaError.badFile(url.path) }
        var header = Data(count: Int(n))
        let got = header.withUnsafeMutableBytes { pread(fd, $0.baseAddress, Int(n), 8) }
        guard got == Int(n), let json = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw EmblaError.badFile(url.path)
        }
        func offset(_ name: String, _ dtype: String, _ cols: Int) throws -> Int {
            let rows = 2 * Self.pages * Self.rowsPerPage
            guard let t = json[name] as? [String: Any], t["dtype"] as? String == dtype,
                  t["shape"] as? [Int] == [rows, cols], let off = t["data_offsets"] as? [Int] else {
                throw EmblaError.badFile("\(url.lastPathComponent): \(name)")
            }
            return 8 + Int(n) + off[0]
        }
        weightOffset = try offset("weight", "U32", Self.dim / 8)
        scalesOffset = try offset("scales", "F16", Self.dim / 64)
        biasesOffset = try offset("biases", "F16", Self.dim / 64)
        let s = UInt64(Self.rowsPerPage)
        units = (1 ..< s).filter { gcd($0, s) == 1 }
    }

    deinit { close(fd) }

    private static let k1: UInt64 = 0x9E37_79B9_7F4A_7C15, k2: UInt64 = 0xBF58_476D_1CE4_E5B9
    private static let k3: UInt64 = 0x94D0_49BB_1331_11EB
    private static let seed: UInt64 = 0x6A09_E667_F3BC_C909, slotSalt: UInt64 = 0xD6E8_FEB8_6659_FD93

    private static func finalize(_ x: UInt64) -> UInt64 {
        var h = (x ^ (x >> 30)) &* k2
        h = (h ^ (h >> 27)) &* k3
        return h ^ (h >> 31)
    }

    /// The table rows of the n-grams ending at the last of `ids` (the 3 latest token ids, oldest
    /// first; positions before the start are the vocabulary size), in the reference's order.
    public func rows(_ ids: [UInt64]) -> [Int] {
        precondition(ids.count == 3)
        let s = UInt64(Self.rowsPerPage), pages = UInt64(Self.pages)
        var out: [Int] = []
        for (k, n) in Self.orders.enumerated() {
            var h = Self.seed
            for j in 0 ..< Int(n) {
                h = (h ^ ids[3 - Int(n) + j]) &* Self.k1
                h ^= h >> 29
            }
            let key = Self.finalize(h ^ (0x243F_6A88_85A3_08D3 &* (n + 1)))
            for g in 0 ..< UInt64(Self.heads / Self.pageHeads) {
                let hp = Self.finalize(key ^ (Self.k1 &* (1 + 64 * n + g)))
                let page = (hp >> 1) % pages
                let hs = Self.finalize(hp ^ Self.slotSalt)
                let a = (hs >> 1) % s
                let step = units[Int((hs >> 40) % UInt64(units.count))]
                for j in 0 ..< UInt64(Self.pageHeads) {
                    out.append(Int((UInt64(k) * pages + page) * s + (a + j * step) % s))
                }
            }
        }
        return out
    }

    /// Reads `rows` (groups of 4 sharing a page) and writes them dequantized as bf16 bits,
    /// `rows.count × 128` values: q·scale + bias in fp32, rounded to bf16, as the reference does.
    public func read(rows: [Int], into e: UnsafeMutablePointer<UInt16>) {
        nonisolated(unsafe) let e = e
        let groups = rows.count / Self.pageHeads
        let fd = self.fd, wo = weightOffset, so = scalesOffset, bo = biasesOffset
        DispatchQueue.concurrentPerform(iterations: groups) { g in
            let rs = rows[g * Self.pageHeads ..< (g + 1) * Self.pageHeads]
            let lo = rs.min()!, n = rs.max()! - lo + 1
            var words = [UInt32](repeating: 0, count: n * 16)
            var scales = [Float16](repeating: 0, count: n * 2), biases = [Float16](repeating: 0, count: n * 2)
            _ = words.withUnsafeMutableBytes { pread(fd, $0.baseAddress, n * 64, off_t(wo + lo * 64)) }
            _ = scales.withUnsafeMutableBytes { pread(fd, $0.baseAddress, n * 4, off_t(so + lo * 4)) }
            _ = biases.withUnsafeMutableBytes { pread(fd, $0.baseAddress, n * 4, off_t(bo + lo * 4)) }
            for (i, r) in rs.enumerated() {
                let k = r - lo, out = e + (g * Self.pageHeads + i) * Self.dim
                for c in 0 ..< Self.dim {
                    let q = Float((words[k * 16 + c / 8] >> (4 * UInt32(c % 8))) & 0xF)
                    let gi = k * 2 + c / 64
                    out[c] = BF16.bits(q * Float(scales[gi]) + Float(biases[gi]))
                }
            }
        }
    }
}

private func gcd(_ a: UInt64, _ b: UInt64) -> UInt64 { b == 0 ? a : gcd(b, a % b) }
