import Foundation

/// Embla 1's tokenizer, from the bundle's `tokenizer.json`: byte-level BPE behind an NFC
/// normalizer and a regex split, with added tokens (`<|im_start|>`, `<think>`, ...) matched first.
public final class Tokenizer: @unchecked Sendable {
    public let vocabSize: Int
    private let vocab: [String: Int]
    private let pieces: [Int: String]          // id -> token string (byte-level chars, or an added token)
    private let ranks: [Pair: Int]
    private let added: [(text: String, id: Int)]  // longest first
    private let addedIDs: [String: Int]
    public let specialIDs: Set<Int>
    private let split: NSRegularExpression
    private var cache: [String: [Int]] = [:]
    private let lock = NSLock()

    struct Pair: Hashable { let a: String, b: String }

    /// Byte -> printable char, GPT-2's table (and back).
    private static let byteChars: [Character] = {
        var bs = Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255)
        var cs = bs
        var n = 0
        for b in 0 ..< 256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) { table[b] = Character(Unicode.Scalar(UInt32(c))!) }
        return table
    }()
    private static let charBytes: [Character: UInt8] = {
        var m: [Character: UInt8] = [:]
        for (b, c) in byteChars.enumerated() { m[c] = UInt8(b) }
        return m
    }()

    public init(url: URL) throws {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
        guard let model = json["model"] as? [String: Any], model["type"] as? String == "BPE",
              let vocab = model["vocab"] as? [String: Int], let merges = model["merges"] as? [Any]
        else { throw ChatError.badModel("\(url.lastPathComponent) is not a byte-level BPE tokenizer") }
        var ranks: [Pair: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (r, m) in merges.enumerated() {
            if let p = m as? [String], p.count == 2 {
                ranks[Pair(a: p[0], b: p[1])] = r
            } else if let s = m as? String, let sp = s.firstIndex(of: " ") {
                ranks[Pair(a: String(s[..<sp]), b: String(s[s.index(after: sp)...]))] = r
            }
        }
        var pieces: [Int: String] = [:]
        for (k, v) in vocab { pieces[v] = k }
        var added: [(String, Int)] = [], special: Set<Int> = []
        for t in json["added_tokens"] as? [[String: Any]] ?? [] {
            guard let id = t["id"] as? Int, let text = t["content"] as? String else { continue }
            added.append((text, id))
            pieces[id] = text
            if t["special"] as? Bool == true { special.insert(id) }
        }
        var pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        if let pre = json["pre_tokenizer"] as? [String: Any] {
            let all = (pre["pretokenizers"] as? [[String: Any]]) ?? [pre]
            for p in all where p["type"] as? String == "Split" {
                if let r = (p["pattern"] as? [String: Any])?["Regex"] as? String { pattern = r }
            }
        }
        self.vocab = vocab
        self.pieces = pieces
        self.ranks = ranks
        self.added = added.sorted { $0.0.count > $1.0.count }
        self.addedIDs = Dictionary(added, uniquingKeysWith: { a, _ in a })
        self.specialIDs = special
        self.vocabSize = (pieces.keys.max() ?? 0) + 1
        self.split = try NSRegularExpression(pattern: pattern)
    }

    public func id(of token: String) -> Int? { addedIDs[token] ?? vocab[token] }

    /// Token ids of `text`; added tokens in it are matched as such.
    public func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            // the earliest added token, longest at a position
            var hit: (Range<Substring.Index>, Int)?
            for (t, id) in added {
                if let r = rest.range(of: t), hit == nil || r.lowerBound < hit!.0.lowerBound {
                    hit = (r, id)
                }
            }
            guard let (r, id) = hit else {
                encodePlain(String(rest), into: &out)
                break
            }
            encodePlain(String(rest[..<r.lowerBound]), into: &out)
            out.append(id)
            rest = rest[r.upperBound...]
        }
        return out
    }

    private func encodePlain(_ text: String, into out: inout [Int]) {
        guard !text.isEmpty else { return }
        let s = text.precomposedStringWithCanonicalMapping as NSString
        for m in split.matches(in: s as String, range: NSRange(location: 0, length: s.length)) {
            let word = s.substring(with: m.range)
            lock.lock()
            let cached = cache[word]
            lock.unlock()
            if let c = cached { out += c; continue }
            let ids = bpe(word)
            lock.lock()
            if cache.count > 50_000 { cache.removeAll() }
            cache[word] = ids
            lock.unlock()
            out += ids
        }
    }

    private func bpe(_ word: String) -> [Int] {
        var parts = word.utf8.map { String(Self.byteChars[Int($0)]) }
        while parts.count > 1 {
            var best = Int.max, at = -1
            for i in 0 ..< parts.count - 1 {
                if let r = ranks[Pair(a: parts[i], b: parts[i + 1])], r < best { best = r; at = i }
            }
            if at < 0 { break }
            // merge every occurrence of the best pair, left to right
            let a = parts[at], b = parts[at + 1]
            var merged: [String] = []
            merged.reserveCapacity(parts.count)
            var i = 0
            while i < parts.count {
                if i < parts.count - 1 && parts[i] == a && parts[i + 1] == b {
                    merged.append(a + b)
                    i += 2
                } else {
                    merged.append(parts[i])
                    i += 1
                }
            }
            parts = merged
        }
        return parts.map { vocab[$0] ?? 0 }
    }

    /// The bytes of a token (an added token's text as UTF-8).
    public func bytes(_ id: Int) -> [UInt8] {
        guard let p = pieces[id] else { return [] }
        if addedIDs[p] == id { return Array(p.utf8) }
        return p.compactMap { Self.charBytes[$0] }
    }

    public func decode(_ ids: [Int], skipSpecial: Bool = true) -> String {
        String(decoding: ids.filter { !skipSpecial || !specialIDs.contains($0) }.flatMap(bytes), as: UTF8.self)
    }
}

/// Text from a stream of tokens: holds back a character's bytes until they are all there.
public struct TextStream {
    private var pending: [UInt8] = []
    public init() {}

    public mutating func push(_ bytes: [UInt8]) -> String {
        pending += bytes
        // the longest prefix that ends on a whole UTF-8 character
        var end = pending.count
        var back = 0
        while back < min(4, pending.count) {
            let b = pending[pending.count - 1 - back]
            if b & 0xC0 != 0x80 {
                let need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1
                if need > back + 1 { end = pending.count - 1 - back }
                break
            }
            back += 1
        }
        let out = String(decoding: pending[..<end], as: UTF8.self)
        pending.removeFirst(end)
        return out
    }

    public mutating func flush() -> String {
        defer { pending = [] }
        return String(decoding: pending, as: UTF8.self)
    }
}

public enum ChatError: Error, CustomStringConvertible {
    case badModel(String)
    case badRequest(String)
    case download(String)

    public var description: String {
        switch self {
        case .badModel(let s), .badRequest(let s), .download(let s): return s
        }
    }
}
