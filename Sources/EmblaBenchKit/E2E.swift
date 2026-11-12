import EmblaMetal
import Foundation

/// `embla-bench e2e <model dir> [prompt tokens] [decode steps] [reps]`: end-to-end latency without a
/// tokenizer: a prompt of `prompt tokens` ids (EMBLA_PROMPT_FILE's first ones, else a pattern),
/// prefilled, then `decode steps` greedy steps after the prompt's own token (1024 and 256 by
/// default). One warm-up and `reps` timed runs, wall clock from handing over the ids to the last
/// token.
func runE2E(_ args: [String]) throws {
    let p = args.count > 1 ? Int(args[1])! : 1024
    let decode = args.count > 2 ? Int(args[2])! : 256
    let reps = args.count > 3 ? Int(args[3])! : 5
    let t0 = Date()
    let engine = try Engine(modelDirectory: URL(fileURLWithPath: args[0]), maxContext: max(4096, p + decode + 64))
    print(String(format: "engine ready in %.2f s", Date().timeIntervalSince(t0)))
    var prompt: [Int] = []
    if let file = ProcessInfo.processInfo.environment["EMBLA_PROMPT_FILE"] {
        prompt = try String(contentsOfFile: file, encoding: .utf8).split(whereSeparator: \.isWhitespace).map { Int($0)! }
    }
    while prompt.count < p { prompt += [758, 6426, 314, 29103, 368, 13, 198, 46447] }
    prompt = Array(prompt.prefix(p))
    var times: [Double] = []
    for rep in 0 ... reps {
        let start = DispatchTime.now().uptimeNanoseconds
        // the prompt's last token's step makes the first new token; then `decode` more steps
        let r = engine.generate(prompt: prompt, count: decode + 1)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-6
        if rep > 0 { times.append(ms) }
        let sum = r.tokens.enumerated().reduce(UInt64(1469598103934665603)) { ($0 ^ UInt64($1.element &+ $1.offset)) &* 1099511628211 }
        print(String(format: "[%04x] %@ %.1f ms: prompt %.1f ms (%.0f tok/s; GPU %.1f ms, engram reads %.1f ms), decode %.3f ms a token (%.0f tok/s), GPU %.3f ms a step",
                     Int(sum & 0xffff), rep == 0 ? "warm-up" : "rep \(rep)", ms, engine.prefillSeconds * 1e3, Double(p) / engine.prefillSeconds,
                     engine.prefillGPU * 1e3, engine.prefillRead * 1e3, r.wallPerToken * 1e3, 1 / r.wallPerToken, r.gpuPerStep * 1e3))
        if rep == reps {
            print("  first tokens: \(r.tokens.prefix(16).map(String.init).joined(separator: " "))")
            print(String(format: "  GPU: first buffer %.0f µs, gap %.0f µs, second buffer %.0f µs, gap to next step %.0f µs",
                         engine.gaps.0, engine.gaps.1, engine.gaps.2, engine.gaps.3))
            print(String(format: "  round trip: GPU done -> CPU awake %.0f µs, engram read + signal %.0f µs, signal -> GPU resumes %.0f µs",
                         engine.latency.0, engine.latency.1, engine.latency.2))
        }
    }
    if ProcessInfo.processInfo.environment["EMBLA_PROFILE"] == "1" {
        _ = engine.generate(prompt: prompt, count: decode + 1)
        let r = try engine.profileSteps(8)
        print(String(format: "profile: a step %.0f µs inside encoders, %.0f µs between them (an encoder per dependent group)", r.total, r.gaps))
        for (name, us) in r.parts { print(String(format: "  %-14@ %7.1f µs", name as NSString, us)) }
    }
    if ProcessInfo.processInfo.environment["EMBLA_CLOCK"] == "1" {
        // the shader clock right after decoding, and after the GPU idled a while
        let probe = try ClockProbe(engine.ctx)
        _ = engine.generate(prompt: prompt, count: decode + 1)
        let busy = probe.run()
        usleep(2_000_000)
        let idle = (0 ..< 3).map { _ in probe.run() }
        var hot: [Double] = []
        for _ in 0 ..< 40 { hot.append(probe.run()) }
        print(String(format: "clock probe back to back (40): min %.0f, last %.0f µs", hot.min()!, hot.last!))
        print(String(format: "clock probe: %.0f µs right after decode, %@ µs after 2 s idle", busy, idle.map { String(format: "%.0f", $0) }.joined(separator: ", ")))
    }
    let mean = times.reduce(0, +) / Double(times.count)
    let sd = (times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(max(1, times.count - 1))).squareRoot()
    print(String(format: "e2e %d+%d: %.1f ms (sd %.1f) over %d reps", p, decode, mean, sd, times.count))
}
