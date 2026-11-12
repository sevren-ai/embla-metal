import EmblaMetal
import Foundation

/// `embla-bench det <model dir> [steps]`: the prompt's first `steps` tokens (EMBLA_PROMPT_FILE) one
/// step at a time, twice, comparing every step's logits: the first step that differs and by how
/// much. The kernels are deterministic, so any difference is a race.
func runDet(_ args: [String]) throws {
    let n = args.count > 1 ? Int(args[1])! : 64
    let engine = try Engine(modelDirectory: URL(fileURLWithPath: args[0]), maxContext: 4096)
    if let parts = ProcessInfo.processInfo.environment["EMBLA_SKIP"] { engine.skip = Set(parts.split(separator: ",").map(String.init)) }
    var ids: [Int] = []
    if let file = ProcessInfo.processInfo.environment["EMBLA_PROMPT_FILE"] {
        ids = try String(contentsOfFile: file, encoding: .utf8).split(whereSeparator: \.isWhitespace).map { Int($0)! }
    }
    while ids.count < n { ids += [758, 6426, 314, 29103, 368, 13, 198, 46447] }
    func run() -> [[Float]] {
        engine.reset()
        return ids.prefix(n).map { t in
            engine.step(t)
            let p = engine.logits.contents().assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: p, count: Embla.vocab))
        }
    }
    let a = run()
    for trial in 1 ... 3 {
        let b = run()
        var first = -1, worst: Float = 0
        for s in 0 ..< n {
            let d = zip(a[s], b[s]).map { abs($0 - $1) }.max()!
            if d > 0 && first < 0 { first = s }
            worst = max(worst, d)
        }
        print(first < 0 ? "trial \(trial): identical over \(n) steps" : "trial \(trial): first difference at step \(first), largest \(worst)")
    }
}
