import EmblaMetal
import Foundation

/// The engine and tokenizer behind the server: one generation at a time (`run` is called on one
/// serial queue).
public final class ChatModel: @unchecked Sendable {
    public let engine: Engine
    public let tokenizer: Tokenizer
    public let directory: URL
    let endIDs: Set<Int>
    let thinkClose: Int?

    public init(directory: URL, context: Int) throws {
        let config = directory.appendingPathComponent("config.json")
        let type = ((try? JSONSerialization.jsonObject(with: Data(contentsOf: config))) as? [String: Any])?["model_type"] as? String
        guard let type, ["embla", "embla1"].contains(type) else {
            throw ChatError.badModel("\(directory.path) is not an Embla 1 model (config.json's model_type is \(type ?? "missing"))")
        }
        tokenizer = try Tokenizer(url: directory.appendingPathComponent("tokenizer.json"))
        engine = try Engine(modelDirectory: directory, maxContext: context)
        self.directory = directory
        endIDs = Set(["<|im_end|>", "<|endoftext|>"].compactMap(tokenizer.id(of:)))
        thinkClose = tokenizer.id(of: "</think>")
    }

    public struct Request: Sendable {
        public var prompt: [Int]
        public var maxTokens: Int
        public var sampling: Sampling
        public var stop: [String]
        /// The prompt opened a `<think>` block: text until `</think>` is reasoning.
        public var thinking: Bool

        public init(prompt: [Int], maxTokens: Int, sampling: Sampling, stop: [String] = [], thinking: Bool) {
            self.prompt = prompt
            self.maxTokens = maxTokens
            self.sampling = sampling
            self.stop = stop
            self.thinking = thinking
        }
    }

    public enum Delta { case reasoning(String), content(String) }

    public struct Outcome {
        public var finishReason: String
        public var promptTokens: Int, completionTokens: Int
        public var promptMs: Double, decodeMs: Double
    }

    /// The tokens a conversation leaves for a reply.
    public func room(for prompt: [Int]) -> Int { engine.maxContext - prompt.count - 1 }

    /// Generates, handing text to `emit` as it comes; `emit` returning false stops (a client gone).
    public func run(_ r: Request, emit: (Delta) -> Bool) -> Outcome {
        var text = TextStream()
        var thinking = r.thinking
        var reasonHeld = ""          // trailing newlines of the reasoning, dropped at </think>
        var contentStarted = false   // newlines after </think> are dropped
        var content = ""             // content not yet emitted (held for the stop strings)
        let hold = max(0, (r.stop.map(\.count).max() ?? 0) - 1)
        var finish = "length", made = 0, alive = true
        let t0 = Date()
        var firstAt: Date?

        func emitContent(final: Bool) -> Bool {
            if let s = r.stop.compactMap({ content.range(of: $0) }).min(by: { $0.lowerBound < $1.lowerBound }) {
                let before = String(content[..<s.lowerBound])
                content = ""
                if !before.isEmpty { alive = emit(.content(before)) && alive }
                finish = "stop"
                return false
            }
            let keep = final ? 0 : hold
            guard content.count > keep else { return true }
            let cut = content.index(content.endIndex, offsetBy: -keep)
            let out = String(content[..<cut])
            content = String(content[cut...])
            return emit(.content(out))
        }

        _ = engine.generate(prompt: r.prompt, count: r.maxTokens, sampling: r.sampling) { [self] id in
            if firstAt == nil { firstAt = Date() }
            made += 1
            if endIDs.contains(id) { finish = "stop"; return false }
            if thinking && id == thinkClose {
                thinking = false
                reasonHeld = ""
                return true
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
        let promptMs = (firstAt ?? Date()).timeIntervalSince(t0) * 1e3
        return Outcome(finishReason: finish, promptTokens: r.prompt.count, completionTokens: made,
                       promptMs: promptMs, decodeMs: Date().timeIntervalSince(firstAt ?? Date()) * 1e3)
    }
}
