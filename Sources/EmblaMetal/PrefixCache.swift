import Foundation
import Metal

/// Keys and values of earlier sequences, kept for prompts that start the same way: a
/// conversation's next turn, another conversation with the same instructions, a second question
/// about the same document. `Engine.generate` uses it by itself.
///
/// Sequences are cut into blocks of 128 positions (a multiple of the sliding window, `Engine.check`). A block is known by its
/// tokens and every token before it, so equal prefixes share their blocks however many sequences
/// they start. Every stored block keeps the full layers' keys and values of its positions (2.6 KB
/// a token). Some also keep a checkpoint: the sliding layers' keys and values of its positions
/// and the engram conv's last inputs (4 MB), which is what decoding needs from the positions
/// before a block's end besides the full layers'. A prompt resumes at the end of the last stored
/// checkpoint block it starts with and computes the rest. Checkpoints are only at turns: for each
/// request, the last block end at least `turnMargin` positions before its prompt's end and before
/// its sequence's end (often the same block). That is where the next prompt goes its own way:
/// a conversation's next turn renders the reply it follows a little differently (a chat template
/// drops the reply's empty reasoning), or asks another question after the same instructions.
/// Resuming there computes the new tokens and less than a block more. A prompt that leaves an
/// earlier sequence elsewhere, inside a long document, resumes at the turn checkpoint before that,
/// or computes it all; a 10k-token document costs one or two checkpoints, not one a block.
///
/// The least recently used sequences go first once the store is over `limit` bytes.
final class PrefixCache {
    static let block = 128
    /// The engram conv's reach back (4 taps dilated by 3), rounded up: rows of it a checkpoint keeps.
    static let histRows = 16
    /// How far before a turn's end its checkpoint is at least: how many tokens before the end of an
    /// earlier prompt the next one may go its own way and still resume there.
    static let turnMargin = 16

    final class Block {
        let hash: UInt64
        let parent: Block?
        /// Positions index · 128 ..< (index + 1) · 128.
        let index: Int
        let tokens: [Int]
        /// The full layers' keys and values (`Engine.fullSpans`).
        let full: MTLBuffer
        /// The checkpoint (`Engine.windowSpans`), if this block has one.
        var window: MTLBuffer?
        var children = 0
        var lastUse: UInt64 = 0

        init(hash: UInt64, parent: Block?, index: Int, tokens: [Int], full: MTLBuffer) {
            self.hash = hash
            self.parent = parent
            self.index = index
            self.tokens = tokens
            self.full = full
        }

        var bytes: Int { full.length + (window?.length ?? 0) }
        var end: Int { (index + 1) * PrefixCache.block }
    }

    private var blocks: [UInt64: Block] = [:]
    private var clock: UInt64 = 0
    /// Bytes the stored blocks hold.
    private(set) var bytes = 0
    /// Most bytes to hold; the least recently used blocks go past it. 0: store nothing.
    var limit: Int

    init(limit: Int) {
        self.limit = limit
    }

    var count: Int { blocks.count }

    /// FNV-1a over the block's tokens, started from its parent's hash.
    static func hash(_ parent: UInt64, _ tokens: ArraySlice<Int>) -> UInt64 {
        var h = parent ^ 0xcbf2_9ce4_8422_2325
        for t in tokens {
            var v = UInt32(truncatingIfNeeded: t)
            for _ in 0 ..< 4 {
                h = (h ^ UInt64(v & 0xFF)) &* 0x100_0000_01b3
                v >>= 8
            }
        }
        return h
    }

    /// The stored blocks `tokens` starts with, in order, covering at most `limit` positions.
    func chain(_ tokens: [Int], upTo limit: Int) -> [Block] {
        var out: [Block] = []
        var parent: Block?
        var j = 0
        while (j + 1) * Self.block <= limit {
            let slice = tokens[j * Self.block ..< (j + 1) * Self.block]
            let h = Self.hash(parent?.hash ?? 0, slice)
            // equal hashes are checked: the same parent and the same tokens
            guard let b = blocks[h], b.parent === parent, b.index == j, b.tokens.elementsEqual(slice) else { break }
            out.append(b)
            parent = b
            j += 1
        }
        return out
    }

    /// Marks `chain`'s blocks as used now (a block is never used more recently than its parent).
    func touch(_ chain: [Block]) {
        clock += 1
        for b in chain { b.lastUse = clock }
    }

    /// A new block after `parent` (nil if one with its hash is stored for other tokens).
    func insert(parent: Block?, index: Int, tokens: ArraySlice<Int>, full: MTLBuffer) -> Block? {
        let h = Self.hash(parent?.hash ?? 0, tokens)
        guard blocks[h] == nil else { return nil }
        let b = Block(hash: h, parent: parent, index: index, tokens: Array(tokens), full: full)
        b.lastUse = clock
        blocks[h] = b
        parent?.children += 1
        bytes += b.bytes
        return b
    }

    func addWindow(_ b: Block, _ window: MTLBuffer) {
        b.window = window
        bytes += window.length
    }

    /// Drops the least recently used blocks with nothing after them until within `limit`, never
    /// one of `keep`. A block left with nothing after it and no checkpoint goes too: no prompt can
    /// resume past it.
    func evict(keeping keep: Set<UInt64> = []) {
        while bytes > limit {
            guard let lru = blocks.values.filter({ $0.children == 0 && !keep.contains($0.hash) }).min(by: { $0.lastUse < $1.lastUse }) else { return }
            remove(lru)
        }
    }

    func removeAll() {
        blocks = [:]
        bytes = 0
    }

    private func remove(_ b: Block) {
        blocks[b.hash] = nil
        bytes -= b.bytes
        guard let parent = b.parent else { return }
        parent.children -= 1
        if parent.children == 0 && parent.window == nil { remove(parent) }
    }
}

extension Engine {
    /// A copy between the engine's caches and a block's buffer: `bytes` at `offset` in `buffer`.
    struct Span { let buffer: MTLBuffer, offset: Int, bytes: Int }

    /// The full layers' keys and values of positions p0 ..< p0 + 128, per layer, k then v, per
    /// kv head: the rows, then (int8) their scales.
    func fullSpans(_ p0: Int) -> [Span] {
        let n = PrefixCache.block, hd = config.headDim, row = Self.kv8 ? hd : 2 * hd
        var out: [Span] = []
        for (k, v) in fullRegions {
            for base in [k, v] {
                for h in 0 ..< config.kvHeads {
                    out.append(Span(buffer: cache, offset: base + (h * maxContext + p0) * row, bytes: n * row))
                }
                if Self.kv8 {
                    for h in 0 ..< config.kvHeads {
                        out.append(Span(buffer: cache, offset: base + config.kvHeads * maxContext * hd + (h * maxContext + p0) * 4, bytes: n * 4))
                    }
                }
            }
        }
        return out
    }

    /// A checkpoint at p0 + 128: the sliding layers' keys and values of positions p0 ..< p0 + 128
    /// (their ring slots, contiguous: p0 and the ring are multiples of 128) and the engram conv's
    /// inputs of the last `PrefixCache.histRows` of them.
    func windowSpans(_ p0: Int) -> [Span] {
        let n = PrefixCache.block, slot = p0 % Self.ring, hd = config.headDim
        var out: [Span] = []
        for (k, v) in slidingRegions {
            for base in [k, v] {
                for h in 0 ..< config.slidingKVHeads {
                    out.append(Span(buffer: cache, offset: base + ((h * Self.ring + slot) * hd) * 2, bytes: n * hd * 2))
                }
            }
        }
        let h0 = (p0 + n - PrefixCache.histRows) % Self.ring
        out.append(Span(buffer: hist, offset: h0 * config.hidden * 2, bytes: PrefixCache.histRows * config.hidden * 2))
        return out
    }

    /// Copies `spans` into `stored` (save) or back (restore), in order, in `blit`.
    func copy(_ spans: [Span], _ stored: MTLBuffer, save: Bool, _ blit: MTLBlitCommandEncoder) {
        var at = 0
        for s in spans {
            if save {
                blit.copy(from: s.buffer, sourceOffset: s.offset, to: stored, destinationOffset: at, size: s.bytes)
            } else {
                blit.copy(from: stored, sourceOffset: at, to: s.buffer, destinationOffset: s.offset, size: s.bytes)
            }
            at += s.bytes
        }
    }

    static func spanBytes(_ spans: [Span]) -> Int { spans.reduce(0) { $0 + $1.bytes } }

    /// Encodes a chained command buffer of copies, committed: later GPU work waits for it.
    func chainedBlit(_ body: (MTLBlitCommandEncoder) -> Void) -> MTLCommandBuffer {
        let cb = chainedCommandBuffer()
        let blit = cb.makeBlitCommandEncoder()!
        body(blit)
        blit.endEncoding()
        endChained(cb)
        cb.commit()
        return cb
    }

    /// The store's start for `prompt` if it beats `reuse` (where the live caches can resume): the
    /// end of the last checkpoint block the prompt starts with. Restores that checkpoint and the
    /// blocks before it that the live caches lack (they hold the prompt's first `live` positions),
    /// and returns its position; nil if it is no further.
    func restoreFromStore(_ prompt: [Int], live: Int, beyond reuse: Int) -> Int? {
        let store = prefixStore
        guard store.limit > 0 else { return nil }
        let chain = store.chain(prompt, upTo: prompt.count - 1)
        guard let k = chain.lastIndex(where: { $0.window != nil }), chain[k].end > reuse else {
            store.touch(chain)
            return nil
        }
        let used = Array(chain[...k])
        store.touch(used)
        let start = used[k].end
        resume(at: start, tokens: prompt)
        // the rings hold this sequence's positions from the checkpoint's on
        ringValidFrom = start - PrefixCache.block
        // later GPU work waits for it (chained); the CPU writes nothing these copies touch
        _ = chainedBlit { blit in
            // the blocks the live caches lack (they hold positions before `live`)
            for b in used where b.end > live { copy(fullSpans(b.index * PrefixCache.block), b.full, save: false, blit) }
            copy(windowSpans(used[k].index * PrefixCache.block), used[k].window!, save: false, blit)
        }
        return start
    }

    /// Stores the blocks of the sequence now in the caches (positions 0 ..< position) the store
    /// lacks, and the checkpoints it wants that the rings still hold: the last block end at least
    /// `PrefixCache.turnMargin` before each of `turnEnds` (the prompt's end and the
    /// sequence's).
    func saveToStore(turnEnds: [Int]) {
        let store = prefixStore
        guard store.limit > 0 else { return }
        let n = PrefixCache.block
        let tokens = cachedTokens
        var wanted = Set<Int>()   // block indices to checkpoint
        for e in turnEnds where e - PrefixCache.turnMargin >= n {
            wanted.insert((e - PrefixCache.turnMargin) / n - 1)
        }
        let fullBytes = Self.spanBytes(fullSpans(0)), windowBytes = Self.spanBytes(windowSpans(0))
        let dev = ctx.device
        var chain = store.chain(tokens, upTo: position)
        var saves: [(spans: [Span], buffer: MTLBuffer)] = []
        var j = chain.count
        while (j + 1) * n <= position {
            guard let buffer = dev.makeBuffer(length: fullBytes, options: .storageModePrivate),
                  let b = store.insert(parent: chain.last, index: j, tokens: tokens[j * n ..< (j + 1) * n], full: buffer) else { break }
            saves.append((fullSpans(j * n), buffer))
            chain.append(b)
            j += 1
        }
        for b in chain where b.window == nil {
            let p0 = b.index * n
            // a checkpoint we keep, whose ring slots hold this sequence's positions (not overwritten
            // since, nor left from another sequence by a restore)
            guard p0 >= ringValidFrom, position - p0 <= Self.ring,
                  wanted.contains(b.index),
                  let buffer = dev.makeBuffer(length: windowBytes, options: .storageModePrivate) else { continue }
            store.addWindow(b, buffer)
            saves.append((windowSpans(p0), buffer))
        }
        store.touch(chain)
        if !saves.isEmpty {
            // Waited for: the engram conv's inputs are in a shared buffer the CPU clears (`reset`) as
            // soon as the next prompt shares nothing, which must not happen before they are copied.
            chainedBlit { blit in
                for s in saves { copy(s.spans, s.buffer, save: true, blit) }
            }.waitUntilCompleted()
        }
        store.evict(keeping: Set(chain.map(\.hash)))
        if store.bytes > store.limit { store.evict() }
    }
}
