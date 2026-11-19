import EmblaMetal
import Foundation

/// The engine and tokenizer behind the server: one generation at a time (`run` is called on one
/// serial queue).
public final class ChatModel: @unchecked Sendable {
    public let engine: Engine
    public let tokenizer: Tokenizer
    public let directory: URL

    public init(directory: URL, context: Int) throws {
        let config = directory.appendingPathComponent("config.json")
        let type = ((try? JSONSerialization.jsonObject(with: Data(contentsOf: config))) as? [String: Any])?["model_type"] as? String
        guard let type, ["embla", "embla1"].contains(type) else {
            throw ChatError.badModel("\(directory.path) is not an Embla 1 model (config.json's model_type is \(type ?? "missing"))")
        }
        tokenizer = try Tokenizer(url: directory.appendingPathComponent("tokenizer.json"))
        engine = try Engine(modelDirectory: directory, maxContext: context)
        self.directory = directory
    }

    public struct Request: Sendable {
        public var prompt: [Int]
        public var maxTokens: Int
        public var sampling: Sampling
        public var stop: [String]
        /// The prompt opened a `<think>` block: text until `</think>` is reasoning.
        public var thinking: Bool
        /// The prompt offered tools: the reply after `<tool_call>` is its calls, not content.
        public var tools: Bool

        public init(prompt: [Int], maxTokens: Int, sampling: Sampling, stop: [String] = [], thinking: Bool, tools: Bool = false) {
            self.prompt = prompt
            self.maxTokens = maxTokens
            self.sampling = sampling
            self.stop = stop
            self.thinking = thinking
            self.tools = tools
        }
    }

    /// `call` is the reply after `<tool_call>`, for `ToolCall.parse`: an empty one as the tag comes, then its text.
    public enum Delta { case reasoning(String), content(String), call(String) }

    public struct Outcome {
        public var finishReason: String
        public var promptTokens: Int, completionTokens: Int
        /// Prompt tokens taken from the engine's prefix cache (an earlier request's) instead of computed.
        public var cachedTokens: Int
        public var promptMs: Double, decodeMs: Double
    }

    /// The tokens a conversation leaves for a reply.
    public func room(for prompt: [Int]) -> Int { engine.maxContext - prompt.count - 1 }

    /// Generates, handing text to `emit` as it comes; `emit` returning false stops (a client gone).
    public func run(_ r: Request, emit: (Delta) -> Bool) -> Outcome {
        let t0 = Date()
        var firstAt: Date?
        var cached: Int?
        let (finish, made) = Self.reply(r, tokenizer: tokenizer, generate: { [engine] prompt, count, onToken in
            _ = engine.generate(prompt: prompt, count: count, sampling: r.sampling) { id in
                if firstAt == nil { firstAt = Date() }
                return onToken(id)
            }
            if cached == nil { cached = engine.reusedTokens }
        }, emit: emit)
        let promptMs = (firstAt ?? Date()).timeIntervalSince(t0) * 1e3
        return Outcome(finishReason: finish, promptTokens: r.prompt.count, completionTokens: made,
                       cachedTokens: cached ?? 0, promptMs: promptMs, decodeMs: Date().timeIntervalSince(firstAt ?? Date()) * 1e3)
    }

    /// The reply's text from its tokens: `generate(prompt, count, onToken)` runs the model (the
    /// engine, or a script in the tests), `onToken` returning false stopping it. Returns the finish
    /// reason and the tokens made. (`generate` is escaping only to satisfy exclusivity checking: it
    /// is called before this returns.)
    static func reply(_ r: Request, tokenizer: Tokenizer,
                      generate: @escaping (_ prompt: [Int], _ count: Int, _ onToken: (Int) -> Bool) -> Void,
                      emit: (Delta) -> Bool) -> (finish: String, made: Int) {
        let endIDs = Set(["<|im_end|>", "<|endoftext|>"].compactMap(tokenizer.id(of:)))
        let thinkClose = tokenizer.id(of: "</think>"), toolCall = tokenizer.id(of: "<tool_call>")
        var text = TextStream()
        var thinking = r.thinking
        var reasonHeld = ""          // trailing newlines of the reasoning, dropped at </think>
        var contentStarted = false   // newlines after </think> are dropped
        var content = ""             // content, or after <tool_call> the call's text, not yet emitted (held for the stop strings)
        var calling = false          // after <tool_call>
        let hold = max(0, (r.stop.map(\.count).max() ?? 0) - 1)
        var finish = "length", made = 0, alive = true

        func emitContent(final: Bool) -> Bool {
            if let s = r.stop.compactMap({ content.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
                let before = String(content[..<s.lowerBound])
                content = ""
                if !before.isEmpty { alive = emit(calling ? .call(before) : .content(before)) && alive }
                finish = "stop"
                return false
            }
            // with tools, also the two newlines the template puts before a <tool_call>: they are not content
            let keep = final ? 0 : max(hold, r.tools ? content.suffix(2).reversed().prefix { $0 == "\n" }.count : 0)
            guard content.count > keep else { return true }
            let cut = content.index(content.endIndex, offsetBy: -keep)
            let out = String(content[..<cut])
            content = String(content[cut...])
            return emit(calling ? .call(out) : .content(out))
        }

        generate(r.prompt, r.maxTokens) { id in
            made += 1
            if endIDs.contains(id) { finish = "stop"; return false }
            if thinking && id == thinkClose {
                thinking = false
                reasonHeld = ""
                return true
            }
            // unless a stop string ends at the tag: then it is content, and the reply stops there
            if r.tools && !calling && id == toolCall && !r.stop.contains(where: { (content + "<tool_call>").contains($0) }) {
                thinking = false   // a call from inside the reasoning ends it, as </think> would
                while content.hasSuffix("\n") { content.removeLast() }
                alive = emitContent(final: true) && alive
                calling = true
                contentStarted = true   // the call's text is kept as it comes
                return emit(.call("")) && alive
            }
            var piece = text.push(tokenizer.specialIDs.contains(id) ? [] : tokenizer.bytes(id))
            guard !piece.isEmpty else { return true }
            if thinking {
                piece = reasonHeld + piece
                var tail = ""
                while piece.hasSuffix("\n") { piece.removeLast(); tail += "\n" }
                reasonHeld = tail
                if !piece.isEmpty { alive = emit(.reasoning(piece)) }
                return alive
            }
            if !contentStarted {
                while piece.hasPrefix("\n") { piece.removeFirst() }
                if piece.isEmpty { return true }
                contentStarted = true
            }
            content += piece
            alive = emitContent(final: false) && alive
            return alive && finish != "stop"
        }
        content += text.flush()
        if alive && finish != "stop" || !content.isEmpty { _ = emitContent(final: true) }
        return (finish, made)
    }
}
