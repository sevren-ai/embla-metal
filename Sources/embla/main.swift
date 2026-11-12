import EmblaChat
import Foundation

setvbuf(stdout, nil, _IOLBF, 0)
let usage = """
usage:
  embla download [repo]       fetch a model into the Hugging Face cache (default \(ModelStore.defaultRepo))
  embla serve [options]       an OpenAI-compatible server, with a chat page at /
      --model <repo or dir>   default \(ModelStore.defaultRepo), from the cache
      --host <addr>           default 127.0.0.1 (this Mac only)
      --port <n>              default 8080
      --context <tokens>      longest conversation, default 32768
      --open                  open the chat page in the browser
"""

@MainActor func run() {
var args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    defer { args.removeSubrange(i ... i + 1) }
    return args[i + 1]
}
func flag(_ name: String) -> Bool {
    guard let i = args.firstIndex(of: name) else { return false }
    args.remove(at: i)
    return true
}

switch args.first {
case "download":
    let repo = args.count > 1 ? args[1] : ModelStore.defaultRepo
    do {
        let dir = try ModelStore.download(repo)
        print("ready: \(dir.path)\nnow: swift run -c release embla serve --open")
    } catch { fail("download failed: \(error)") }
case "serve":
    args.removeFirst()
    let model = option("--model") ?? ModelStore.defaultRepo
    let host = option("--host") ?? "127.0.0.1"
    let port = Int(option("--port") ?? "8080") ?? 8080
    let context = Int(option("--context") ?? "32768") ?? 32768
    let open = flag("--open")
    if !args.isEmpty { fail("unknown arguments: \(args.joined(separator: " "))\n\(usage)") }
    guard let dir = ModelStore.resolve(model) else {
        fail("no model at \(model): run `swift run -c release embla download` first (or pass --model <dir>)")
    }
    do {
        try serve(modelDirectory: dir, name: model, host: host, port: port, context: context, open: open)
    } catch { fail("\(error)") }
case "check-tokenizer":
    // development: `embla check-tokenizer <model dir> <cases.json>`, a JSON list of strings
    // (tokenized) or of message lists (rendered with the chat template, then tokenized)
    do {
        let tok = try Tokenizer(url: URL(fileURLWithPath: args[1]).appendingPathComponent("tokenizer.json"))
        let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[2]))) as! [Any]
        let out: [[String: Any]] = try cases.map { c in
            var text = c as? String ?? ""
            if let ms = c as? [[String: String]] {
                text = try ChatTemplate.render(ms.map { ChatMessage(role: $0["role"]!, content: $0["content"]!, reasoning: $0["reasoning_content"]) })
            }
            let ids = tok.encode(text)
            return ["text": text, "ids": ids, "decoded": tok.decode(ids, skipSpecial: false)]
        }
        let d = try JSONSerialization.data(withJSONObject: out)
        FileHandle.standardOutput.write(d)
    } catch { fail("\(error)") }
default:
    print(usage)
    exit(args.isEmpty || args.first == "help" || args.first == "--help" || args.first == "-h" ? 0 : 2)
}
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

run()

func serve(modelDirectory: URL, name: String, host: String, port: Int, context: Int, open: Bool) throws {
    print("loading \(modelDirectory.path)")
    let t0 = Date()
    let model = try ChatModel(directory: modelDirectory, context: context)
    print(String(format: "ready in %.1f s, on %@ (context %d tokens)", Date().timeIntervalSince(t0), model.engine.ctx.device.name, context))
    let server = try Server(model: model, name: name.contains("/") ? String(name.split(separator: "/").last!) : "embla-1", host: host, port: port)
    let shown = host == "0.0.0.0" || host == "::" ? "localhost" : host
    let url = "http://\(shown):\(port)"
    server.start { error in
        if let error { fail("cannot listen on \(host):\(port): \(error) (another server there? try --port)") }
        print("""
        chat page:   \(url)/
        OpenAI API:  \(url)/v1  (model "\(server.modelName)", no authentication)
        Ctrl-C stops it.
        """)
        if open {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
            p.arguments = [url + "/"]
            try? p.run()
        }
    }
    dispatchMain()
}
