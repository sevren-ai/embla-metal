import Foundation
import XCTest

@testable import EmblaChat

final class ChatTests: XCTestCase {
    /// The template's output for text conversations, as transformers renders chat_template.jinja.
    func testTemplate() throws {
        XCTAssertEqual(try ChatTemplate.render([ChatMessage(role: "user", content: "Hi there")]),
                       "<|endoftext|><|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n")
        XCTAssertEqual(try ChatTemplate.render([
            ChatMessage(role: "system", content: "  Be terse.  "),
            ChatMessage(role: "user", content: "Q1"),
            ChatMessage(role: "assistant", content: "<think>\nreasoning\n</think>\n\nAnswer one"),
            ChatMessage(role: "user", content: "Q2"),
        ]), "<|endoftext|><|im_start|>system\nBe terse.<|im_end|>\n<|im_start|>user\nQ1<|im_end|>\n"
            + "<|im_start|>assistant\nAnswer one<|im_end|>\n<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>\n")
        // a reply after the last user turn keeps its reasoning
        XCTAssertEqual(try ChatTemplate.render([ChatMessage(role: "user", content: "Q1"), ChatMessage(role: "assistant", content: "partial")]),
                       "<|endoftext|><|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\npartial<|im_end|>\n<|im_start|>assistant\n<think>\n")
        XCTAssertThrowsError(try ChatTemplate.render([ChatMessage(role: "user", content: "a"), ChatMessage(role: "system", content: "b")]))
        XCTAssertThrowsError(try ChatTemplate.render([]))
    }

    /// OpenAI content parts are concatenated with nothing between them, like the jinja's render_content.
    func testContentParts() throws {
        XCTAssertEqual(try Server.text("plain"), "plain")
        XCTAssertEqual(try Server.text(nil), "")
        XCTAssertEqual(try Server.text([["type": "text", "text": "Always answer "], ["type": "text", "text": "in Danish."]]),
                       "Always answer in Danish.")
        XCTAssertThrowsError(try Server.text([["type": "image_url", "image_url": ["url": "x"]]]))
    }

    /// Multi-byte characters split across tokens come out whole.
    func testTextStream() {
        var s = TextStream()
        let bytes = Array("aé日🙂".utf8)
        var out = ""
        for b in bytes { out += s.push([b]) }
        out += s.flush()
        XCTAssertEqual(out, "aé日🙂")
        var t = TextStream()
        XCTAssertEqual(t.push(Array("🙂".utf8.prefix(2))), "")
        XCTAssertEqual(t.push(Array("🙂".utf8.suffix(2)) + [0x41]), "🙂A")
    }

    /// The tokenizer round-trips text and matches known ids (needs the model in the cache).
    func testTokenizer() throws {
        guard let dir = ModelStore.resolve(ModelStore.defaultRepo) else { throw XCTSkip("no \(ModelStore.defaultRepo) in the cache") }
        let tok = try Tokenizer(url: dir.appendingPathComponent("tokenizer.json"))
        // transformers: "The capital of Denmark is"
        XCTAssertEqual(tok.encode("The capital of Denmark is"), [758, 6426, 314, 29103, 368])
        for text in ["Hello, world!", "Café — 日本語 🙂\n\n  code:\tx = 1", "<|im_start|>user\nhi<|im_end|>"] {
            XCTAssertEqual(tok.decode(tok.encode(text), skipSpecial: false), text)
        }
        XCTAssertEqual(tok.encode("<think>"), [tok.id(of: "<think>")!])
    }
}
