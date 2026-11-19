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

    /// Tools, calls and their results, as transformers renders chat_template.jinja: the tools and
    /// the arguments in the client's order, written as Python's json.dumps writes them.
    func testToolTemplate() throws {
        let tool = #"{"type":"function","function":{"name":"get_weather","description":"Vädret, t.ex. \"Malmö\"","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}"#
        let tools = [try JSON(Data(tool.utf8))]
        let head = "<|endoftext|><|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
            + #"{"type": "function", "function": {"name": "get_weather", "description": "Vädret, t.ex. \"Malmö\"", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}"#
            + "\n</tools>" + ChatTemplate.toolFormat
        XCTAssertEqual(try ChatTemplate.render([ChatMessage(role: "system", content: " Be terse. "), ChatMessage(role: "user", content: "Q")], tools: tools),
                       head + "\n\nBe terse.<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n<think>\n")

        // a reply's calls and their results, after the last user turn: the reasoning stays
        let malmo = try ToolCall(JSON(Data(#"{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Malmö\", \"days\": 2}"}}"#.utf8)))
        let lund = try ToolCall(JSON(Data(#"{"function":{"name":"get_weather","arguments":{"city":"Lund","opts":{"wind":true,"units":["c"]}}}}"#.utf8)))
        let asked = "<|im_end|>\n<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n"
        let calls = "<tool_call>\n<function=get_weather>\n<parameter=city>\nMalmö\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>\n"
            + "<tool_call>\n<function=get_weather>\n<parameter=city>\nLund\n</parameter>\n<parameter=opts>\n{\"wind\": true, \"units\": [\"c\"]}\n</parameter>\n</function>\n</tool_call>"
        let results = "<|im_end|>\n<|im_start|>user\n<tool_response>\n6\n</tool_response>\n<tool_response>\n5\n</tool_response><|im_end|>\n"
        var messages = [
            ChatMessage(role: "user", content: "Weather?"),
            ChatMessage(role: "assistant", content: "Let me check.", reasoning: "Two cities.", toolCalls: [malmo, lund]),
            ChatMessage(role: "tool", content: "6"),
            ChatMessage(role: "tool", content: "5"),
        ]
        XCTAssertEqual(try ChatTemplate.render(messages, tools: tools),
                       head + asked + "<think>\nTwo cities.\n</think>\n\nLet me check.\n\n" + calls + results + "<|im_start|>assistant\n<think>\n")

        // before it: only the content and the calls, and no blank line without content
        messages[1].content = ""
        messages += [ChatMessage(role: "assistant", content: "6 and 5."), ChatMessage(role: "user", content: "Thanks")]
        XCTAssertEqual(try ChatTemplate.render(messages, tools: tools),
                       head + asked + calls + results + "<|im_start|>assistant\n6 and 5.<|im_end|>\n<|im_start|>user\nThanks<|im_end|>\n<|im_start|>assistant\n<think>\n")
    }

    /// The instructions after the tools are the template's, character for character (needs the model in the cache).
    func testToolFormat() throws {
        guard let dir = ModelStore.resolve(ModelStore.defaultRepo) else { throw XCTSkip("no \(ModelStore.defaultRepo) in the cache") }
        let jinja = try String(contentsOf: dir.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        XCTAssertTrue(jinja.contains("'" + ChatTemplate.toolFormat.replacingOccurrences(of: "\n", with: "\\n") + "'"))
    }

    /// A reply's calls: a value is JSON where the schema names types and a string is not one of them.
    func testToolCalls() throws {
        let tools = [try JSON(Data(#"{"type":"function","function":{"name":"find","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},"zip":{"type":"integer"},"tag":{"anyOf":[{"type":"string"},{"type":"null"}]},"filter":{"type":"object"}}}}}"#.utf8))]
        let reply = "<tool_call>\n<function=find>\n<parameter=query>\n42\n</parameter>\n<parameter=limit>\n42\n</parameter>\n"
            + "<parameter=zip>\n01234\n</parameter>\n<parameter=tag>\ntrue\n</parameter>\n"
            + "<parameter=filter>\n{\"lang\": \"da\"}\n</parameter>\n<parameter=note>\ntwo\nlines\n</parameter>\n</function>\n</tool_call>\n"
            + "<tool_call>\n<function=now>\n</function>\n</tool_call>"
        let calls = ToolCall.parse(reply, tools: tools)
        XCTAssertEqual(calls.map(\.name), ["find", "now"])
        // 01234 is not a JSON number
        XCTAssertEqual(JSON.object(calls[0].arguments).text,
                       #"{"query": "42", "limit": 42, "zip": "01234", "tag": "true", "filter": {"lang": "da"}, "note": "two\nlines"}"#)
        XCTAssertEqual(JSON.object(calls[1].arguments).text, "{}")
        // the template writes them back as the model wrote them
        XCTAssertEqual(ChatTemplate.render(calls, after: ""), reply)
        // without a schema a value stays a string, and a value may hold the format's own tags
        let html = ToolCall.parse("\n<function=write>\n<parameter=n>\n42\n</parameter>\n<parameter=text>\n<b></function>\n<parameter=x>\n</parameter>\n</function>", tools: tools)
        XCTAssertEqual(html.map { JSON.object($0.arguments).text }, [#"{"n": "42", "text": "<b></function>\n<parameter=x>"}"#])
        // a type behind a $ref, also one that only refers to itself
        let refs = [try JSON(Data(##"{"type":"function","function":{"name":"send","parameters":{"type":"object","properties":{"to":{"$ref":"#/$defs/Address"},"copies":{"anyOf":[{"$ref":"#/$defs/Count"},{"type":"null"}]},"loop":{"$ref":"#/$defs/Loop"}},"$defs":{"Address":{"type":"object"},"Count":{"type":"integer"},"Loop":{"$ref":"#/$defs/Loop"}}}}}"##.utf8))]
        XCTAssertEqual(JSON.object(ToolCall.parse("\n<function=send>\n<parameter=to>\n{\"city\": \"Malmö\"}\n</parameter>\n<parameter=copies>\n2\n</parameter>\n<parameter=loop>\n2\n</parameter>\n</function>", tools: refs)[0].arguments).text,
                       #"{"to": {"city": "Malmö"}, "copies": 2, "loop": "2"}"#)
        // only the format is a call: not what follows other text, nor a second one without its own <tool_call>
        XCTAssertEqual(ToolCall.parse("\n<function=a>\n</function>\n</tool_call> then <tool_call>\n<function=b>\n</function>", tools: tools).map(\.name), ["a"])
        XCTAssertEqual(ToolCall.parse("<function=a></function><function=b></function>", tools: tools).map(\.name), ["a"])
        XCTAssertTrue(ToolCall.parse("I would call <function=a></function>", tools: tools).isEmpty)
        // cut off inside a call: the calls before it
        XCTAssertTrue(ToolCall.parse("\n<function=find>\n<parameter=query>\n4", tools: tools).isEmpty)
        XCTAssertEqual(ToolCall.parse(reply + "\n<tool_call>\n<function=find>\n<parameter=query>\n4", tools: tools).count, 2)

        // a bare true, false or null in Python's spelling, and null where the schema allows it beside a string
        let flags = [try JSON(Data(#"{"type":"function","function":{"name":"set","parameters":{"type":"object","properties":{"on":{"type":"boolean"},"limit":{"type":["integer","null"]},"name":{"type":["string","null"]},"note":{"type":"string"}}}}}"#.utf8))]
        XCTAssertEqual(JSON.object(ToolCall.parse("\n<function=set>\n<parameter=on>\nTrue\n</parameter>\n<parameter=limit>\nNone\n</parameter>\n<parameter=name>\nnull\n</parameter>\n<parameter=note>\nNone\n</parameter>\n</function>", tools: flags)[0].arguments).text,
                       #"{"on": true, "limit": null, "name": null, "note": "None"}"#)

        // a client's call: the arguments a JSON string or an object, in its order
        XCTAssertEqual(try ToolCall(JSON(Data(#"{"function":{"name":"f","arguments":"{\"b\":1,\"a\":2}"}}"#.utf8))).arguments.map(\.0), ["b", "a"])
        XCTAssertTrue(try ToolCall(JSON(Data(#"{"function":{"name":"f","arguments":""}}"#.utf8))).arguments.isEmpty)
        for bad in [#"{"function":{"arguments":"{}"}}"#, #"{"function":{"name":"f","arguments":"{\"a\":"}}"#, #"{"function":{"name":"f","arguments":"[]"}}"#] {
            XCTAssertThrowsError(try ToolCall(JSON(Data(bad.utf8))), bad)
        }
    }

    /// Ordered JSON reads JSON (strictly: no trailing comma) and writes what Python's json.dumps writes.
    func testJSON() throws {
        let text = #"{"b": [1, -0.5, true, null, {}], "a": "é \" \\ \n \u0001 日本", "c": {"z": [], "y": "/"}}"#
        XCTAssertEqual(try JSON(Data(text.utf8)).text, text)
        XCTAssertEqual(try JSON(Data(" { \"k\" :\t\"\\u00e9\\/\" } ".utf8)).text, #"{"k": "é/"}"#)
        // a repeated key keeps its place and takes the later value, as a Python dict does
        XCTAssertEqual(try JSON(Data(#"{"x":1,"y":[{"a":1,"a":2}],"x":3}"#.utf8)).text, #"{"x": 3, "y": [{"a": 2}]}"#)
        // Python writes a float its own way and leaves an integer alone
        XCTAssertEqual(try JSON(Data("[1e3, -2.5E3, 0.10, 1e-7, 1E+16, 9007199254740994.0, 12345678901234567890, -0]".utf8)).text,
                       "[1000.0, -2500.0, 0.1, 1e-07, 1e+16, 9007199254740994.0, 12345678901234567890, 0]")
        for bad in ["", "{", "[1,]", "[,1]", #"{"a" 1}"#, "nul", "0x10", "01", "+5", ".5", "5.", "1e999", #""open"#, #""open\"#, "{} x",
                    String(repeating: "[", count: 5000) + String(repeating: "]", count: 5000)] {
            XCTAssertThrowsError(try JSON(Data(bad.utf8)), String(bad.prefix(20)))
        }
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
