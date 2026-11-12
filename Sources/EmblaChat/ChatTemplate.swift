import Foundation

public struct ChatMessage: Sendable {
    public var role: String
    public var content: String
    /// An assistant turn's reasoning, when the client sends it apart from the content.
    public var reasoning: String?

    public init(role: String, content: String, reasoning: String? = nil) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
    }
}

/// The bundle's chat template (`chat_template.jinja`) for text conversations, in Swift: the
/// system message, user and assistant turns, and the generation prompt that opens the reply's
/// `<think>` block. Earlier assistant turns keep only their answers, as the template does.
/// Tools and images are not supported.
public enum ChatTemplate {
    public static func render(_ messages: [ChatMessage]) throws -> String {
        guard !messages.isEmpty else { throw ChatError.badRequest("no messages") }
        var out = "<|endoftext|>"
        var msgs = messages
        if msgs[0].role == "system" || msgs[0].role == "developer" {
            out += "<|im_start|>system\n" + trim(msgs[0].content) + "<|im_end|>\n"
            msgs.removeFirst()
        }
        guard let lastUser = msgs.lastIndex(where: { $0.role == "user" }) else {
            throw ChatError.badRequest("no user message")
        }
        for (i, m) in msgs.enumerated() {
            var content = trim(m.content)
            switch m.role {
            case "user":
                out += "<|im_start|>user\n" + content + "<|im_end|>\n"
            case "assistant":
                var reasoning = m.reasoning ?? ""
                if m.reasoning == nil, let close = content.range(of: "</think>") {
                    var r = String(content[..<close.lowerBound])
                    while r.hasSuffix("\n") { r.removeLast() }
                    if let open = r.range(of: "<think>", options: .backwards) { r = String(r[open.upperBound...]) }
                    while r.hasPrefix("\n") { r.removeFirst() }
                    reasoning = r
                    let last = content.range(of: "</think>", options: .backwards)!
                    var c = String(content[last.upperBound...])
                    while c.hasPrefix("\n") { c.removeFirst() }
                    content = c
                }
                if i > lastUser {
                    // after the last user turn (a reply still being built): its reasoning stays
                    reasoning = trim(reasoning)
                    out += "<|im_start|>assistant\n<think>\n"
                    out += reasoning.isEmpty ? "\n</think>\n\n" + content : reasoning + "\n</think>\n\n" + content
                    out += "<|im_end|>\n"
                } else {
                    out += "<|im_start|>assistant\n" + content + "<|im_end|>\n"
                }
            case "system", "developer":
                throw ChatError.badRequest("a system message must come first")
            default:
                throw ChatError.badRequest("unsupported role \(m.role)")
            }
        }
        return out + "<|im_start|>assistant\n<think>\n"
    }

    /// Jinja's `trim`: leading and trailing whitespace.
    static func trim(_ s: String) -> String { s.trimmingCharacters(in: .whitespacesAndNewlines) }
}
