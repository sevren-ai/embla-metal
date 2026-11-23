import EmblaMetal
import Foundation

/// `embla-bench cache <model dir> [prompt tokens] [new tokens]`: the prefix cache against computing
/// every prompt from scratch. Each request runs with the cache, then from scratch on the same
/// engine with the cache off, and the two are compared: the prompt tokens taken from the cache,
/// the prompt's time both ways, how many greedy tokens agree, and the largest difference in the
/// last step's logits.
///
/// With EMBLA_BATCH=0 (every prompt as decode steps) both ways compute every position the same
/// way, so the check is strict: every token and every logit must be equal. With the batched
/// prefill, positions taken from the cache may have been computed by a decode step (an earlier
/// reply, an earlier prompt's last token) where computing from scratch runs them through the
/// prefill; the two paths differ slightly (README), so a greedy continuation can part at a
/// near-tie. The check then needs the first new token equal, and reports the rest. Either way,
/// taking less from the cache than the case should fails it.
///
/// The live sequence (one conversation):
/// - turn: the next turn (the first prompt, its reply, then more)
/// - extend: the first prompt and more (its reply not taken)
/// - fork: the first half of the first prompt, then something else
/// - miss: nothing in common
///
/// The store of earlier sequences:
/// - a/b turns: two conversations with different instructions taking turns, each turn resuming
///   from the store after the other conversation ran
/// - shared: the first conversation's instructions with another question
/// - evict: three conversations in a store that holds two and a half: the one used again comes
///   back, the least recently used loses its newest blocks
///
/// The prompt is EMBLA_PROMPT_FILE's token ids, or shuffled sentences.
func runCache(_ args: [String]) throws {
    let p = args.count > 1 ? Int(args[1])! : 1024
    let n = args.count > 2 ? Int(args[2])! : 64
    let engine = try Engine(modelDirectory: URL(fileURLWithPath: args[0]), maxContext: max(4096, 4 * p + 8 * n + 256))
    var text: [Int] = []
    if let file = ProcessInfo.processInfo.environment["EMBLA_PROMPT_FILE"] {
        text = try String(contentsOfFile: file, encoding: .utf8).split(whereSeparator: \.isWhitespace).map { Int($0)! }
    }
    var random = SplitMix(seed: 7)
    let sentence = [758, 6426, 314, 29103, 368, 13, 198, 46447, 1307, 2201]
    while text.count < 3 * p + 4 * n { text += sentence.shuffled(using: &random) }
    // `more` backwards, so that it is not what the model writes after `first` (its reply)
    let first = Array(text[..<p]), more = Array(text[p ..< p + n].reversed())
    let other = Array(text[p + n ..< 2 * p + n]), third = Array(text[2 * p + n ..< 3 * p + n])

    struct Run { var tokens: [Int], logits: [Float], seconds: Double, reused: Int }
    func run(_ prompt: [Int]) -> Run {
        let r = engine.generate(prompt: prompt, count: n)
        let logits = Array(UnsafeBufferPointer(start: engine.logits.contents().assumingMemoryBound(to: Float.self), count: engine.config.vocab))
        return Run(tokens: r.tokens, logits: logits, seconds: engine.prefillSeconds, reused: engine.reusedTokens)
    }
    /// From scratch: no cache used, nothing stored. It leaves the same sequence in the live caches
    /// as the cached run before it (the same tokens).
    func scratch(_ prompt: [Int]) -> Run {
        engine.prefixCaching = false
        defer { engine.prefixCaching = true }
        return run(prompt)
    }
    let strict = ProcessInfo.processInfo.environment["EMBLA_BATCH"] == "0"
    var failed = false
    var lastReused = 0
    /// Runs `prompt` with the cache as things stand, then from scratch; prints and checks them.
    func compare(_ name: String, _ prompt: [Int], minReuse: Int) -> [Int] {
        let cached = run(prompt), fresh = scratch(prompt)
        lastReused = cached.reused
        let agree = zip(cached.tokens, fresh.tokens).prefix { $0 == $1 }.count
        let diff = zip(cached.logits, fresh.logits).map { abs($0 - $1) }.max()!
        let ok = (strict ? agree == n && diff == 0 : agree >= 1) && cached.reused >= minReuse
        if !ok { failed = true }
        print(String(format: "  %-9@ %5d tokens, %5d from the cache · prompt %6.1f ms (from scratch %6.1f) · %d of %d tokens agree · last logits max |Δ| %.4f%@",
                     name as NSString, prompt.count, cached.reused, cached.seconds * 1e3, fresh.seconds * 1e3, agree, n, diff,
                     ok ? "" : "  <- FAILED" as NSString))
        return cached.tokens
    }

    _ = scratch(first)   // warm: kernels and engram pages
    print("prompt \(p) tokens, then \(n) greedy; store limit \(engine.prefixCacheLimit >> 20) MB; " +
          (strict ? "strict: every token and logit equal" : "the first new token equal"))

    print("the live sequence, after the first prompt and its reply:")
    for name in ["turn", "extend", "fork", "miss"] {
        engine.clearPrefixCache()
        let reply = run(first).tokens
        switch name {
        case "turn": _ = compare(name, first + reply + more, minReuse: p)
        case "extend": _ = compare(name, first + more, minReuse: p)
        case "fork": _ = compare(name, Array(first[..<(p / 2)]) + other, minReuse: p / 2)
        default: _ = compare(name, Array(first.reversed()), minReuse: 0)
        }
    }

    print("the store of earlier sequences:")
    engine.clearPrefixCache()
    var a = first, b = other
    for turn in 1 ... 3 {
        // after the other conversation's turn, all but the last block or two of this one's
        // sequence comes back from the store
        a += compare("a turn \(turn)", a, minReuse: turn == 1 ? 0 : a.count - n - 2 * 128) + more.prefix(16 * turn)
        b += compare("b turn \(turn)", b, minReuse: turn == 1 ? 0 : b.count - n - 2 * 128) + more.suffix(16 * turn)
    }
    // the first conversation's first prompt is stored with a checkpoint just before its end
    _ = compare("shared", first + third, minReuse: (p - 16) / 128 * 128)
    print(String(format: "  store: %d blocks of 128 positions, %.0f MB", engine.prefixCacheBlocks, Double(engine.prefixCacheBytes) / 1e6))

    // a store that holds about two and a half conversations (measured on the first):
    // a, b, a, c, a, b: a comes back each time, c pushes out b's newest blocks (the least
    // recently used, leaves first), so b comes back with less
    engine.clearPrefixCache()
    let limit = engine.prefixCacheLimit
    var convs = [first, other, third]
    for (k, i) in [0, 1, 0, 2, 0, 1].enumerated() {
        if k == 1 { engine.prefixCacheLimit = engine.prefixCacheBytes * 5 / 2 }
        let hit = k == 2 || k == 4   // a's second and third turns
        convs[i] += compare("evict \(["a", "b", "c"][i])\(k + 1)", convs[i], minReuse: hit ? convs[i].count - n - 2 * 128 : 0) + more.prefix(8)
        if k == 5 && lastReused >= p { failed = true; print("  <- FAILED: none of b was evicted") }
    }
    print(String(format: "  store: %d blocks, %.1f MB of %.1f", engine.prefixCacheBlocks, Double(engine.prefixCacheBytes) / 1e6,
                 Double(engine.prefixCacheLimit) / 1e6))
    if engine.prefixCacheBytes > engine.prefixCacheLimit { failed = true; print("  <- FAILED: over the limit") }
    engine.prefixCacheLimit = limit

    // the engine's own determinism, for scale: the same request from scratch twice
    let x = scratch(first + more), y = scratch(first + more)
    print(String(format: "scratch twice: last logits max |Δ| %.4f", zip(x.logits, y.logits).map { abs($0 - $1) }.max()!))
    print(failed ? "FAILED" : "ok")
    fflush(stdout)   // the report, before an error ends the program
    if failed { throw CacheCheckFailed() }
}

struct CacheCheckFailed: Error, CustomStringConvertible {
    var description: String { "the prefix cache changed the tokens, or reused less than it should" }
}

/// The same shuffles every run.
private struct SplitMix: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
