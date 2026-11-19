import Foundation

public struct ChatMessage: Sendable {
    public var role: String
    public var content: String
    /// An assistant turn's reasoning, when the client sends it apart from the content.
    public var reasoning: String?
    /// An assistant turn's tool calls.
    public var toolCalls: [ToolCall]

    public init(role: String, content: String, reasoning: String? = nil, toolCalls: [ToolCall] = []) {
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.toolCalls = toolCalls
    }
}

/// The bundle's chat template (`chat_template.jinja`) for text conversations, in Swift: the
/// tools and the system message, user and assistant turns, tool calls and their results, and the
/// generation prompt that opens the reply's `<think>` block. Earlier assistant turns keep only
/// their answers, as the template does. Images are not supported.
public enum ChatTemplate {
    public static func render(_ messages: [ChatMessage], tools: [JSON] = []) throws -> String {
        guard !messages.isEmpty else { throw ChatError.badRequest("no messages") }
        var out = "<|endoftext|>"
        var msgs = messages
        var system: String?
        if msgs[0].role == "system" || msgs[0].role == "developer" { system = trim(msgs.removeFirst().content) }
        if !tools.isEmpty {
            out += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
            out += tools.map { "\n" + $0.text }.joined() + "\n</tools>" + toolFormat
            if let system, !system.isEmpty { out += "\n\n" + system }
            out += "<|im_end|>\n"
        } else if let system {
            out += "<|im_start|>system\n" + system + "<|im_end|>\n"
        }
        // a user turn that is a tool's response is not the query the reply answers
        guard let lastUser = msgs.lastIndex(where: {
            guard $0.role == "user" else { return false }
            let c = trim($0.content)
            return !(c.hasPrefix("<tool_response>") && c.hasSuffix("</tool_response>"))
        }) else {
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
                let calls = render(m.toolCalls, after: content)
                if i > lastUser {
                    // after the last user turn (a reply still being built): its reasoning stays
                    reasoning = trim(reasoning)
                    out += "<|im_start|>assistant\n<think>\n"
                    out += reasoning.isEmpty ? "\n</think>\n\n" + content : reasoning + "\n</think>\n\n" + content
                    out += calls + "<|im_end|>\n"
                } else {
                    out += "<|im_start|>assistant\n" + content + calls + "<|im_end|>\n"
                }
            case "tool":
                // consecutive results share one user turn (the template's loop.previtem: a first message has none)
                if i == 0 ? system != nil : msgs[i - 1].role != "tool" { out += "<|im_start|>user" }
                out += "\n<tool_response>\n" + content + "\n</tool_response>"
                if i == msgs.count - 1 || msgs[i + 1].role != "tool" { out += "<|im_end|>\n" }
            case "system", "developer":
                throw ChatError.badRequest("a system message must come first")
            default:
                throw ChatError.badRequest("unsupported role \(m.role)")
            }
        }
        return out + "<|im_start|>assistant\n<think>\n"
    }

    /// The template's `render_tool_calls`: an assistant turn's calls, after its content.
    static func render(_ calls: [ToolCall], after content: String) -> String {
        var out = ""
        for (i, call) in calls.enumerated() {
            out += (i > 0 ? "\n" : content.isEmpty ? "" : "\n\n") + "<tool_call>\n<function=" + call.name + ">\n"
            for (name, value) in call.arguments {
                out += "<parameter=" + name + ">\n" + (value.string ?? value.text) + "\n</parameter>\n"
            }
            out += "</function>\n</tool_call>"
        }
        return out
    }

    /// What the template tells the model after the list of tools.
    static let toolFormat = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>"
        + "\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span"
        + "\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:"
        + "\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags"
        + "\n- Required parameters MUST be specified"
        + "\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after"
        + "\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls"
        + "\n</IMPORTANT>"

    /// Jinja's `trim`: leading and trailing whitespace.
    static func trim(_ s: String) -> String { s.trimmingCharacters(in: .whitespacesAndNewlines) }
}
