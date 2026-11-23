import Foundation

/// A model's shape, read from its `config.json` (the Hugging Face config the checkpoint ships with).
/// The engine derives its plans, buffers and kernel instantiations from it instead of from constants,
/// so a model of another shape, or another model line, is a config away where the kernels allow it;
/// `Engine.check(_:)` says what they do not.
public struct ModelConfig: Sendable, Equatable {
    public enum Attention: String, Sendable { case sliding = "sliding_attention", full = "full_attention", sparse = "sparse_attention" }
    /// A layer's MLP: dense, or mixture of experts ("sparse" in the config).
    public enum MLP: String, Sendable { case dense, moe = "sparse" }

    public var hidden: Int
    public var layers: Int
    public var headDim: Int
    public var vocab: Int
    public var heads: Int, kvHeads: Int
    public var slidingHeads: Int, slidingKVHeads: Int
    /// Keys a sliding layer attends.
    public var window: Int
    /// Past `sparseTopK + sparseRecent` keys, a sparse layer attends the `sparseTopK` highest-scoring
    /// older keys (selected by its block's full layer) and the `sparseRecent` latest.
    public var sparseTopK: Int, sparseRecent: Int
    /// Layers an AttnRes block: each completed block is a source of every later mix.
    public var resBlock: Int
    /// Layers [0, prefillLayers) and layer prefillLayers' K/V: what a prompt token whose logits are not
    /// needed runs (every K/V the later layers read projects from them).
    public var prefillLayers: Int
    public var experts: Int, expertsPerToken: Int, expertDim: Int, sharedDim: Int
    /// A dense MLP's hidden width.
    public var denseDim: Int
    public var rmsEps: Double
    public var logitSoftcap: Double
    public var ropeTheta: Double, ropeFraction: Double
    public var attention: [Attention]
    public var mlp: [MLP]
    /// Full layer -> the layer whose (normed, mixed) input its K/V project.
    public var kvSources: [Int: Int]
    public var engram: Engram

    public struct Engram: Sendable, Equatable {
        /// The layer after whose MLP the engram's output becomes a source.
        public var layer: Int
        public var conv: Int, heads: Int, headDim: Int, pageHeads: Int, pages: Int, pageBytes: Int
        public var orders: [Int]
    }

    /// Options the config sets that change the model's function and that `ModelConfig` does not
    /// describe, by config key. An engine that does not run them refuses the model.
    public var unknownOptions: [String]

    // ------------------------------------------------------------------ derived

    public var kinds: [Attention] { attention }

    /// K/V source layer -> the later full layers that project its input, in layer order.
    public var bridges: [Int: [Int]] {
        var b: [Int: [Int]] = [:]
        for (f, s) in kvSources.sorted(by: { $0.key < $1.key }) where f != s { b[s, default: []].append(f) }
        return b
    }

    /// The last full layer before layer i (the one a sparse layer reads its keys and selection from).
    public func blockFull(_ i: Int) -> Int? { (0 ..< i).last { attention[$0] == .full } }

    /// Full layers whose keys the sparse layers after them select from.
    public var selects: Set<Int> {
        Set((0 ..< layers).filter { i in
            attention[i] == .full && (i + 1 ..< layers).contains { attention[$0] == .sparse && blockFull($0) == i }
        })
    }

    public func blockStart(_ i: Int) -> Bool { i % resBlock == 0 }

    /// Sources before layer i's first sub-layer: the embedding; after the engram's layer, the engram;
    /// and every completed block.
    public func sources(_ i: Int) -> Int { 1 + (i > engram.layer ? 1 : 0) + i / resBlock }

    /// Sources of the final mix (the last block is the partial sum).
    public var finalSources: Int { sources(layers - 1) }

    /// Source slots a token's state holds: the most any mix reads.
    public var sourceSlots: Int { finalSources }

    /// The mixes layer i's attention input takes: its query's, and one for each later full layer
    /// that projects its K/V from this layer's input.
    public func attentionMixes(_ i: Int) -> Int { 1 + (bridges[i]?.count ?? 0) }

    /// Mixes a token's mix output holds (the most any sub-layer takes).
    public var mixSlots: Int { (0 ..< layers).map(attentionMixes).max() ?? 1 }

    /// MoE layers, in order.
    public var moeLayers: [Int] { (0 ..< layers).filter { mlp[$0] == .moe } }

    // ------------------------------------------------------------------ loading

    /// From a model directory's config.json.
    public static func load(directory: URL) throws -> ModelConfig {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url) else { throw EngineError.badFile("\(url.path): the model's config.json is needed") }
        return try ModelConfig(json: data, source: url.path)
    }

    /// Keys the parser reads, or knows to be bookkeeping (tokens, dtypes, quantization, training).
    static let known: Set<String> = [
        "hidden_size", "num_hidden_layers", "head_dim", "vocab_size", "num_attention_heads", "num_key_value_heads",
        "sliding_num_attention_heads", "sliding_num_key_value_heads", "sliding_window", "sparse_topk", "sparse_recent",
        "attn_res_block", "num_prefill_layers", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
        "shared_expert_intermediate_size", "intermediate_size", "rms_norm_eps", "final_logit_softcapping",
        "rope_parameters", "layer_types", "mlp_layer_types", "kv_sources", "engram_layer", "engram_conv", "engram_heads",
        "engram_head_dim", "engram_page_heads", "engram_pages", "engram_page_bytes", "engram_orders", "engram_table",
        "engram_hash", "use_qk_norm", "attn_res_weights",
        // bookkeeping: no effect on the forward the engine runs
        "architectures", "model_type", "dtype", "torch_dtype", "bos_token_id", "eos_token_id", "pad_token_id",
        "initializer_range", "engram_init_std", "max_position_embeddings", "tie_word_embeddings", "transformers_version",
        "use_cache", "quantization", "quantization_config", "auto_map",
    ]

    public init(json data: Data, source: String = "config.json") throws {
        guard let c = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw EngineError.badFile(source) }
        func bad(_ key: String) -> EngineError { .badFile("\(source): \(key) missing or malformed") }
        func int(_ key: String) throws -> Int {
            guard let n = c[key] as? NSNumber, n.doubleValue == n.doubleValue.rounded() else { throw bad(key) }
            return n.intValue
        }
        func double(_ key: String) throws -> Double {
            guard let n = c[key] as? NSNumber else { throw bad(key) }
            return n.doubleValue
        }
        func list<T: RawRepresentable>(_ key: String, _: T.Type) throws -> [T] where T.RawValue == String {
            guard let a = c[key] as? [String] else { throw bad(key) }
            return try a.map { guard let v = T(rawValue: $0) else { throw EngineError.unsupported("\(key): \($0)") }; return v }
        }

        hidden = try int("hidden_size")
        layers = try int("num_hidden_layers")
        headDim = try int("head_dim")
        vocab = try int("vocab_size")
        heads = try int("num_attention_heads")
        kvHeads = try int("num_key_value_heads")
        slidingHeads = try int("sliding_num_attention_heads")
        slidingKVHeads = try int("sliding_num_key_value_heads")
        window = try int("sliding_window")
        sparseTopK = try int("sparse_topk")
        sparseRecent = try int("sparse_recent")
        resBlock = try int("attn_res_block")
        prefillLayers = try int("num_prefill_layers")
        experts = try int("num_experts")
        expertsPerToken = try int("num_experts_per_tok")
        expertDim = try int("moe_intermediate_size")
        sharedDim = try int("shared_expert_intermediate_size")
        denseDim = try int("intermediate_size")
        rmsEps = try double("rms_norm_eps")
        logitSoftcap = try double("final_logit_softcapping")
        let rope = c["rope_parameters"] as? [String: Any] ?? [:]
        guard let theta = rope["rope_theta"] as? NSNumber else { throw bad("rope_parameters.rope_theta") }
        ropeTheta = theta.doubleValue
        ropeFraction = (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 1
        attention = try list("layer_types", Attention.self)
        mlp = try list("mlp_layer_types", MLP.self)
        guard let kv = c["kv_sources"] as? [String: Any] else { throw bad("kv_sources") }
        var sources: [Int: Int] = [:]
        for (k, v) in kv {
            guard let f = Int(k), let s = (v as? NSNumber)?.intValue else { throw bad("kv_sources") }
            sources[f] = s
        }
        kvSources = sources
        guard let orders = c["engram_orders"] as? [NSNumber] else { throw bad("engram_orders") }
        engram = Engram(layer: try int("engram_layer"), conv: try int("engram_conv"), heads: try int("engram_heads"),
                        headDim: try int("engram_head_dim"), pageHeads: try int("engram_page_heads"),
                        pages: try int("engram_pages"), pageBytes: try int("engram_page_bytes"), orders: orders.map(\.intValue))

        var unknown: [String] = []
        for k in c.keys where !Self.known.contains(k) { unknown.append(k) }
        // values of read keys this parser describes only in their default form
        if let qk = c["use_qk_norm"] as? Bool, !qk { unknown.append("use_qk_norm") }
        if let w = c["attn_res_weights"] as? String, w != "softmax" { unknown.append("attn_res_weights") }
        if let t = c["engram_table"] as? String, t != "sparse" { unknown.append("engram_table") }
        if let h = c["engram_hash"] as? String, h != "page" { unknown.append("engram_hash") }
        unknownOptions = unknown.sorted()

        guard attention.count == layers else { throw bad("layer_types (\(attention.count) for \(layers) layers)") }
        guard mlp.count == layers else { throw bad("mlp_layer_types (\(mlp.count) for \(layers) layers)") }
        guard resBlock > 0 else { throw bad("attn_res_block") }
        for (f, s) in kvSources {
            guard (0 ..< layers).contains(f), (0 ..< layers).contains(s), s <= f else { throw bad("kv_sources \(f): \(s)") }
        }
    }
}
