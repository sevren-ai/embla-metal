import EmblaMetal
import Foundation
import Network

/// An OpenAI-compatible HTTP server for a `ChatModel`: `/v1/chat/completions`, `/v1/completions`
/// and `/v1/models`, streaming or not, and the chat page at `/`. Requests queue for the one engine.
///
/// This server has no authentication or TLS and allows cross-origin requests. It is intended for
/// loopback use; network exposure requires an authenticated TLS reverse proxy.
public final class Server: @unchecked Sendable {
    let model: ChatModel
    let name: String
    public var modelName: String { name }
    let listener: NWListener
    let io = DispatchQueue(label: "embla.http", attributes: .concurrent)
    let gen = DispatchQueue(label: "embla.generate", qos: .userInitiated)
    public static let defaultTemperature: Float = 0.6, defaultTopP: Float = 0.95

    public init(model: ChatModel, name: String, host: String, port: Int) throws {
        self.model = model
        self.name = name
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        params.requiredLocalEndpoint = NWEndpoint.hostPort(host: NWEndpoint.Host(host), port: NWEndpoint.Port(integerLiteral: UInt16(port)))
        listener = try NWListener(using: params)
    }

    /// Starts listening; `ready` is called once the port is open (or with the error).
    public func start(ready: @escaping @Sendable (Error?) -> Void) {
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready: ready(nil)
            case .failed(let e): ready(e)
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] c in
            guard let self else { return }
            Connection(c, server: self).start()
        }
        listener.start(queue: io)
    }

    // ------------------------------------------------------------------ routes

    func handle(_ req: HTTPRequest, _ c: Connection) {
        switch (req.method, req.path) {
        case ("OPTIONS", _):
            c.respond(204, body: Data())
        case ("GET", "/"), ("GET", "/index.html"):
            guard let url = Bundle.module.url(forResource: "index", withExtension: "html", subdirectory: "Web"),
                  let page = try? Data(contentsOf: url) else { return c.error(500, "the chat page is missing") }
            c.respond(200, type: "text/html; charset=utf-8", body: page)
        case ("GET", "/health"):
            c.json(200, ["status": "ok"])
        case ("GET", "/v1/models"):
            c.json(200, ["object": "list", "data": [["id": name, "object": "model", "created": 0, "owned_by": "sevren-ai",
                                                       "context_length": model.engine.maxContext]]])
        case ("POST", "/v1/chat/completions"), ("POST", "/chat/completions"):
            completion(req, c, chat: true)
        case ("POST", "/v1/completions"), ("POST", "/completions"):
            completion(req, c, chat: false)
        default:
            c.error(404, "no route \(req.method) \(req.path)")
        }
    }

    func completion(_ req: HTTPRequest, _ c: Connection, chat: Bool) {
        guard let body = (try? JSONSerialization.jsonObject(with: req.body)) as? [String: Any] else {
            return c.error(400, "the body is not a JSON object")
        }
        let prompt: [Int]
        do {
            if chat {
                guard let raw = body["messages"] as? [[String: Any]] else { throw ChatError.badRequest("need messages") }
                let messages = try raw.map { m -> ChatMessage in
                    guard let role = m["role"] as? String else { throw ChatError.badRequest("a message without a role") }
                    return ChatMessage(role: role, content: try Self.text(m["content"]),
                                       reasoning: (m["reasoning_content"] ?? m["reasoning"]) as? String)
                }
                prompt = model.tokenizer.encode(try ChatTemplate.render(messages))
            } else {
                guard let p = body["prompt"] as? String else { throw ChatError.badRequest("need a prompt (a string)") }
                prompt = model.tokenizer.encode(p)
                guard !prompt.isEmpty else { throw ChatError.badRequest("an empty prompt") }
            }
        } catch {
            return c.error(400, "\(error)")
        }
        let room = model.room(for: prompt)
        guard room > 0 else {
            return c.error(400, "the conversation is \(prompt.count) tokens, over this server's context (\(model.engine.maxContext); see --context)")
        }
        let asked = (body["max_completion_tokens"] ?? body["max_tokens"]) as? Int
        let temperature = (body["temperature"] as? NSNumber)?.floatValue ?? Self.defaultTemperature
        let topP = (body["top_p"] as? NSNumber)?.floatValue ?? Self.defaultTopP
        let seed = (body["seed"] as? NSNumber).map { UInt32(truncatingIfNeeded: $0.int64Value) } ?? .random(in: 0 ... .max)
        let stop = (body["stop"] as? [String]) ?? (body["stop"] as? String).map { [$0] } ?? []
        let stream = body["stream"] as? Bool ?? false
        let usageInStream = (body["stream_options"] as? [String: Any])?["include_usage"] as? Bool ?? false
        let request = ChatModel.Request(prompt: prompt, maxTokens: max(1, min(asked ?? room, room)),
                                        sampling: Sampling(temperature: temperature, topP: topP, seed: seed),
                                        stop: stop.filter { !$0.isEmpty }, thinking: chat)
        let id = (chat ? "chatcmpl-" : "cmpl-") + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24).lowercased()
        let created = Int(Date().timeIntervalSince1970)
        let object = chat ? "chat.completion" : "text_completion"

        if stream { c.startEvents() }
        gen.async { [self] in
            guard c.isOpen else { return }
            var reasoning = "", content = ""
            if stream {
                c.event(chunk(id, created, chat, chat ? ["role": "assistant", "content": ""] : nil, text: chat ? nil : ""))
            }
            let out = model.run(request) { delta in
                switch delta {
                case .reasoning(let s):
                    reasoning += s
                    if stream && chat { c.event(chunk(id, created, chat, ["reasoning_content": s])) }
                case .content(let s):
                    content += s
                    if stream { c.event(chunk(id, created, chat, chat ? ["content": s] : nil, text: chat ? nil : s)) }
                }
                return c.isOpen
            }
            let usage: [String: Any] = ["prompt_tokens": out.promptTokens, "completion_tokens": out.completionTokens,
                                        "total_tokens": out.promptTokens + out.completionTokens]
            let timings: [String: Any] = ["prompt_n": out.promptTokens, "prompt_ms": out.promptMs,
                                          "predicted_n": out.completionTokens, "predicted_ms": out.decodeMs]
            log(String(format: "%@ %d prompt tokens (%.0f ms), %d new (%.1f tok/s), %@%@", chat ? "chat" : "completion",
                       out.promptTokens, out.promptMs, out.completionTokens,
                       Double(max(0, out.completionTokens - 1)) / max(1e-9, out.decodeMs / 1e3), out.finishReason,
                       c.isOpen ? "" : " (client left)"))
            if stream {
                var last = chunk(id, created, chat, chat ? [:] : nil, text: chat ? nil : "", finish: out.finishReason)
                last["timings"] = timings
                c.event(last)
                if usageInStream {
                    c.event(["id": id, "object": object + ".chunk", "created": created, "model": name, "choices": [], "usage": usage])
                }
                c.endEvents()
            } else {
                var choice: [String: Any] = ["index": 0, "finish_reason": out.finishReason]
                if chat {
                    var message: [String: Any] = ["role": "assistant", "content": content]
                    if !reasoning.isEmpty { message["reasoning_content"] = reasoning }
                    choice["message"] = message
                } else {
                    choice["text"] = content
                }
                c.json(200, ["id": id, "object": object, "created": created, "model": name, "choices": [choice],
                             "usage": usage, "timings": timings])
            }
        }
    }

    func chunk(_ id: String, _ created: Int, _ chat: Bool, _ delta: [String: Any]?, text: String? = nil,
               finish: String? = nil) -> [String: Any] {
        var choice: [String: Any] = ["index": 0, "finish_reason": finish as Any? ?? NSNull()]
        if let delta { choice["delta"] = delta }
        if let text { choice["text"] = text }
        return ["id": id, "object": chat ? "chat.completion.chunk" : "text_completion", "created": created,
                "model": name, "choices": [choice]]
    }

    /// A message's content: a string, or OpenAI's parts (text only).
    static func text(_ content: Any?) throws -> String {
        if let s = content as? String { return s }
        if content == nil || content is NSNull { return "" }
        guard let parts = content as? [[String: Any]] else { throw ChatError.badRequest("content is a string or a list of parts") }
        return try parts.map { p in
            guard (p["type"] as? String ?? "text") == "text", let t = p["text"] as? String else {
                throw ChatError.badRequest("only text content is supported")
            }
            return t
        }.joined(separator: "\n")
    }

    func log(_ s: String) {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        print("[\(f.string(from: Date()))] \(s)")
    }
}

struct HTTPRequest {
    let method: String, path: String
    let headers: [String: String]
    let body: Data
}

/// One HTTP/1.1 exchange on a TCP connection (closed after the response).
final class Connection: @unchecked Sendable {
    let c: NWConnection
    unowned let server: Server
    private var buffer = Data()
    private let lock = NSLock()
    private var open = true
    var isOpen: Bool { lock.lock(); defer { lock.unlock() }; return open }

    init(_ c: NWConnection, server: Server) {
        self.c = c
        self.server = server
    }

    func start() {
        c.stateUpdateHandler = { [self] state in
            switch state {
            case .failed, .cancelled: closed()
            default: break
            }
        }
        c.start(queue: server.io)
        read()
    }

    private func closed() {
        lock.lock()
        open = false
        lock.unlock()
    }

    private func read() {
        c.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [self] data, _, done, error in
            if let data { buffer += data }
            if let req = parse() {
                server.handle(req, self)
                watch()
            } else if done || error != nil || buffer.count > 64 << 20 {
                closed()
                c.cancel()
            } else {
                read()
            }
        }
    }

    /// After the request: the client closing its side means it is gone (a stopped stream).
    private func watch() {
        c.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [self] _, _, done, error in
            if done || error != nil { closed() } else if isOpen { watch() }
        }
    }

    private func parse() -> HTTPRequest? {
        guard let end = buffer.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let head = String(decoding: buffer[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
        let first = head[0].split(separator: " ")
        guard first.count >= 2 else { return nil }
        var headers: [String: String] = [:]
        for line in head.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[line[..<colon].lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        let length = Int(headers["content-length"] ?? "0") ?? 0
        guard buffer.count - end.upperBound >= length else { return nil }
        let body = buffer.subdata(in: end.upperBound ..< end.upperBound + length)
        let path = String(first[1]).split(separator: "?", maxSplits: 1).first.map(String.init) ?? "/"
        return HTTPRequest(method: String(first[0]), path: path, headers: headers, body: body)
    }

    // Browser clients may call the API cross-origin. This is not a security boundary: callers
    // must keep the unauthenticated server on loopback or protect it with a reverse proxy.
    private static let cors = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"

    private func send(_ d: Data, then close: Bool = false) {
        c.send(content: d, completion: .contentProcessed { [self] e in
            if e != nil { closed() }
            if close { c.cancel() }
        })
    }

    func respond(_ status: Int, type: String = "application/json", body: Data) {
        let reason = [200: "OK", 204: "No Content", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"][status] ?? "Error"
        let head = "HTTP/1.1 \(status) \(reason)\r\nContent-Type: \(type)\r\nContent-Length: \(body.count)\r\n\(Self.cors)Connection: close\r\n\r\n"
        send(Data(head.utf8) + body, then: true)
    }

    func json(_ status: Int, _ obj: Any) {
        respond(status, body: (try? JSONSerialization.data(withJSONObject: obj, options: [.withoutEscapingSlashes])) ?? Data())
    }

    func error(_ status: Int, _ message: String) {
        json(status, ["error": ["message": message, "type": status == 404 ? "not_found_error" : "invalid_request_error"]])
    }

    func startEvents() {
        send(Data("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\(Self.cors)Connection: close\r\n\r\n".utf8))
    }

    func event(_ obj: Any) {
        guard isOpen, let d = try? JSONSerialization.data(withJSONObject: obj, options: [.withoutEscapingSlashes]) else { return }
        send(Data("data: ".utf8) + d + Data("\n\n".utf8))
    }

    func endEvents() {
        send(Data("data: [DONE]\n\n".utf8), then: true)
    }
}
