import Foundation

/// JSON with an object's members in the order they were written, which `JSONSerialization` does
/// not keep. The chat template writes the tools and a call's arguments in that order.
public enum JSON: Sendable {
    /// `literal` is null, true, false or a number, as `text` writes it.
    case literal(String), string(String), array([JSON]), object([(String, JSON)])

    public init(_ data: Data) throws {
        var p = Parser(b: Array(data))
        self = try p.value()
        p.space()
        guard p.i == p.b.count else { throw Parser.bad }
    }

    /// As Python's `json.dumps(_, ensure_ascii=False)`, the template's `tojson`.
    public var text: String {
        switch self {
        case .literal(let w): return w
        case .string(let s): return JSON.quote(s)
        case .array(let a): return "[" + a.map(\.text).joined(separator: ", ") + "]"
        case .object(let o): return "{" + o.map { JSON.quote($0.0) + ": " + $0.1.text }.joined(separator: ", ") + "}"
        }
    }

    static func quote(_ s: String) -> String {
        var out = "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case _ where u.value < 0x20: out += String(format: "\\u%04x", u.value)
            default: out.unicodeScalars.append(u)
            }
        }
        return out + "\""
    }

    subscript(key: String) -> JSON? {
        if case .object(let o) = self { return o.first { $0.0 == key }?.1 }
        return nil
    }

    var items: [JSON] {
        if case .array(let a) = self { return a }
        return []
    }

    var string: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    /// The types a parameter's schema names: its `type`, alone or in a list, those of its `anyOf` or
    /// `oneOf`, and those behind a `$ref` into `root`, the tool's parameters.
    func types(in root: JSON, depth: Int = 0) -> [String] {
        if let ref = self["$ref"]?.string {
            guard ref.hasPrefix("#/"), depth < 8 else { return [] }
            let target = ref.dropFirst(2).split(separator: "/").reduce(root as JSON?) { $0?[String($1)] }
            return target?.types(in: root, depth: depth + 1) ?? []
        }
        let type = self["type"]
        return (type?.string.map { [$0] } ?? type?.items.compactMap(\.string) ?? [])
            + ((self["anyOf"] ?? self["oneOf"])?.items.flatMap { $0.types(in: root, depth: depth + 1) } ?? [])
    }

    private struct Parser {
        static let bad = ChatError.badRequest("not JSON, or nested too deep")
        nonisolated(unsafe) static let number = /-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?/   // built once: a literal in place is built for every number
        let b: [UInt8]
        var i = 0, depth = 0

        mutating func space() {
            while i < b.count, b[i] == 0x20 || b[i] == 0x09 || b[i] == 0x0A || b[i] == 0x0D { i += 1 }
        }

        mutating func eat(_ c: Character) throws {
            space()
            guard i < b.count, b[i] == c.asciiValue else { throw Self.bad }
            i += 1
        }

        mutating func value() throws -> JSON {
            space()
            guard i < b.count else { throw Self.bad }
            switch b[i] {
            case UInt8(ascii: "{"), UInt8(ascii: "["):
                let object = b[i] == UInt8(ascii: "{"), close = UInt8(ascii: object ? "}" : "]")
                var members: [(String, JSON)] = [], keys = Set<String>()
                i += 1
                depth += 1   // the recursion is on a queue's small stack
                guard depth <= 128 else { throw Self.bad }
                while true {
                    space()
                    if i < b.count, b[i] == close { break }
                    if !members.isEmpty { try eat(",") }
                    var key = ""
                    if object {
                        guard case .string(let k) = try value() else { throw Self.bad }
                        key = k
                        try eat(":")
                    }
                    let member = try value()
                    // a repeated key keeps its place and takes the later value, as a Python dict does
                    if object, !keys.insert(key).inserted, let j = members.firstIndex(where: { $0.0 == key }) {
                        members[j].1 = member
                    } else {
                        members.append((key, member))
                    }
                }
                i += 1
                depth -= 1
                return object ? .object(members) : .array(members.map(\.1))
            case UInt8(ascii: "\""):
                let start = i
                i += 1
                while i < b.count, b[i] != UInt8(ascii: "\"") { i += b[i] == UInt8(ascii: "\\") ? 2 : 1 }
                guard i < b.count else { throw Self.bad }
                i += 1
                // the escapes are Foundation's to decode
                guard let s = try? JSONSerialization.jsonObject(with: Data(b[start ..< i]), options: .fragmentsAllowed) as? String else {
                    throw Self.bad
                }
                return .string(s)
            default:
                let start = i
                while i < b.count, !",]} \t\n\r".utf8.contains(b[i]) { i += 1 }
                let word = String(decoding: b[start ..< i], as: UTF8.self)
                if ["null", "true", "false"].contains(word) { return .literal(word) }
                guard let number = word.wholeMatch(of: Self.number) else { throw Self.bad }
                if number.2 == nil && number.3 == nil { return .literal(word == "-0" ? "0" : word) }
                // a float as Python writes it (1e3 is 1000.0): Swift's, but from 2^53 to 1e16 Swift has an exponent
                guard let d = Double(word), d.isFinite else { throw Self.bad }
                return .literal(d.magnitude >= 0x1p53 && d.magnitude < 1e16 ? String(format: "%.1f", d) : "\(d)")
            }
        }
    }
}

/// A function call: a name and its arguments, in order.
public struct ToolCall: Sendable {
    public var name: String
    public var arguments: [(String, JSON)]

    public init(name: String, arguments: [(String, JSON)]) {
        self.name = name
        self.arguments = arguments
    }

    /// A call in a client's message: OpenAI's `{"function": {"name", "arguments"}}`, the arguments
    /// a JSON string or an object.
    init(_ json: JSON) throws {
        let f = json["function"] ?? json
        guard let name = f["name"]?.string else { throw ChatError.badRequest("a tool call without a name") }
        var arguments = f["arguments"] ?? .object([])
        if let s = arguments.string { arguments = (try? JSON(Data((s.isEmpty ? "{}" : s).utf8))) ?? .literal("null") }
        guard case .object(let members) = arguments else {
            throw ChatError.badRequest("the arguments of the call to \(name) are not a JSON object")
        }
        self.init(name: name, arguments: members)
    }

    /// The calls in a reply's text after `<tool_call>`, in the format the template asks for. A
    /// value is JSON where the tool's schema names its types and a string is not one of them (a
    /// bare True, False or None being the model's true, false or null), else a string. Anything
    /// but that format ends the list: a call that does not close was cut off, and text after one
    /// is not another.
    static func parse(_ text: String, tools: [JSON]) -> [ToolCall] {
        var calls: [ToolCall] = []
        var rest = Substring(text)
        func skip(_ tag: String) -> Bool {
            rest = rest.drop(while: \.isWhitespace)
            guard rest.hasPrefix(tag) else { return false }
            rest = rest.dropFirst(tag.count).drop(while: \.isWhitespace)
            return true
        }
        // the first call's <tool_call> was the token; a later call has its own
        while skip("<tool_call>") || calls.isEmpty, let name = opening("function", in: &rest) {
            let parameters = tools.first { $0["function"]?["name"]?.string == name }?["function"]?["parameters"]
            var arguments: [(String, JSON)] = []
            while !skip("</function>") {
                // a value ends at its own closing tag, so it may hold a </function>
                guard let key = opening("parameter", in: &rest), let end = rest.firstRange(of: "</parameter>") else { return calls }
                var value = rest[..<end.lowerBound]
                rest = rest[end.upperBound...]
                if value.hasPrefix("\n") { value.removeFirst() }
                if value.hasSuffix("\n") { value.removeLast() }
                let types = parameters.flatMap { $0["properties"]?[key]?.types(in: $0) } ?? []
                let word = ["True": "true", "False": "false", "None": "null"][String(value)] ?? String(value)
                let json = types.isEmpty || types.contains("string")
                    ? (word == "null" && types.contains("null") ? JSON.literal("null") : nil) : try? JSON(Data(word.utf8))
                arguments.append((key, json ?? .string(String(value))))
            }
            calls.append(ToolCall(name: name, arguments: arguments))
            _ = skip("</tool_call>")
        }
        return calls
    }

    /// Takes a `<tag=name>` off the start of `rest` and returns the name.
    static func opening(_ tag: String, in rest: inout Substring) -> String? {
        guard rest.hasPrefix("<" + tag + "="), let end = rest.firstRange(of: ">") else { return nil }
        defer { rest = rest[end.upperBound...] }
        return String(rest[rest.index(rest.startIndex, offsetBy: tag.count + 2) ..< end.lowerBound])
    }
}
