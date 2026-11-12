import EmblaMetal
import Foundation
import Metal

/// `embla-bench check <model dir> <tokens.txt> <out dir> [mix steps] [from step]`: runs the tokens through the
/// engine and writes each step's logits (f32) and, for the given steps, every mix output (bf16), to
/// compare with a reference implementation's.
func runCheck(_ args: [String]) throws {
    let model = URL(fileURLWithPath: args[0])
    let tokens = try String(contentsOfFile: args[1], encoding: .utf8).split(whereSeparator: \.isWhitespace).map { Int($0)! }
    let out = URL(fileURLWithPath: args[2])
    let mixSteps = Set((args.count > 3 ? args[3] : "").split(separator: ",").compactMap { Int($0) })
    let from = args.count > 4 ? Int(args[4])! : 0   // write the logits of steps from here on
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let t0 = Date()
    let engine = try Engine(modelDirectory: model, maxContext: 4096)
    print(String(format: "engine ready in %.2f s", Date().timeIntervalSince(t0)))
    let dbg = engine.ctx.buffer(UInt16.self, count: 100 * Embla.d)
    var logits = Data()
    var produced: [Int] = []
    for (s, tok) in tokens.enumerated() {
        engine.debugMixes = mixSteps.contains(s) ? dbg : nil
        let (next, gpu) = engine.step(tok)
        produced.append(next)
        if s >= from { logits.append(Data(bytes: engine.logits.contents(), count: Embla.vocab * 4)) }
        if mixSteps.contains(s) {
            try Data(bytes: dbg.contents(), count: dbg.length).write(to: out.appendingPathComponent("mix_\(s).bf16"))
        }
        if s < 3 || s == tokens.count - 1 { print(String(format: "step %d: token %d -> %d, GPU %.3f ms", s, tok, next, gpu * 1e3)) }
    }
    if let failure = engine.verifySelection() { print("key selection: \(failure)") } else { print("key selection: matches a CPU sort") }
    try logits.write(to: out.appendingPathComponent("logits.f32"))
    try produced.map(String.init).joined(separator: " ").write(to: out.appendingPathComponent("next.txt"), atomically: true, encoding: .utf8)
}

/// `embla-bench gen <model dir> <tokens.txt> <prompt length> <count> <out.txt>`: greedy decoding of
/// `count` tokens after the first `prompt length` tokens of the file, through `generate` (prompt
/// tokens stop at layer 16), the new tokens written to out.txt.
func runGen(_ args: [String]) throws {
    let tokens = try String(contentsOfFile: args[1], encoding: .utf8).split(whereSeparator: \.isWhitespace).map { Int($0)! }
    let p = Int(args[2])!, n = Int(args[3])!
    let engine = try Engine(modelDirectory: URL(fileURLWithPath: args[0]), maxContext: max(4096, p + n + 64))
    let r = engine.generate(prompt: Array(tokens[..<p]), count: n)
    print(String(format: "prompt %d tokens in %.0f ms (%.0f tok/s; batched part: GPU %.0f ms, engram reads %.0f ms), then %.0f tok/s",
                 p, engine.prefillSeconds * 1e3, Double(p) / engine.prefillSeconds, engine.prefillGPU * 1e3, engine.prefillRead * 1e3, 1 / r.wallPerToken))
    try r.tokens.map(String.init).joined(separator: " ").write(toFile: args[4], atomically: true, encoding: .utf8)
    if let dump = ProcessInfo.processInfo.environment["EMBLA_DUMP_LOGITS"] {   // the last step's logits, f32
        try Data(bytes: engine.logits.contents(), count: Embla.vocab * 4).write(to: URL(fileURLWithPath: dump))
    }
    if ProcessInfo.processInfo.environment["EMBLA_TWICE"] != nil {
        _ = engine.generate(prompt: Array(tokens[..<p]), count: 2)   // warm: the engram rows read once
        let base = engine.prefillGPU
        for part in ["gemm", "experts", "mix", "attn", "router", "moe", "engram", "embed"] {
            engine.prefillTwice = [part]
            _ = engine.generate(prompt: Array(tokens[..<p]), count: 2)
            print(String(format: "  %-7@ %6.0f µs a token", part as NSString, (engine.prefillGPU - base) / Double(p - 1) * 1e6))
        }
        engine.prefillTwice = []
        print(String(format: "  all     %6.0f µs a token", base / Double(p - 1) * 1e6))
    }
}
