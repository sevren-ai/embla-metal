import Foundation
import Testing
@testable import EmblaMetal

/// The released Embla 1 config.json (sevren-ai/Embla1-4bit), without its quantization entries and token ids.
func embla1Config() -> [String: Any] {
    let kinds = (0 ..< 32).map { i in
        i < 16 ? (i == 7 ? "full_attention" : "sliding_attention") : ((i - 16) % 4 == 0 ? "full_attention" : "sparse_attention")
    }
    return [
        "architectures": ["EmblaForCausalLM"], "model_type": "embla", "attn_res_block": 4, "dtype": "bfloat16",
        "engram_conv": 4, "engram_head_dim": 128, "engram_heads": 8, "engram_init_std": 1.0, "engram_layer": 0,
        "engram_orders": [2, 3], "engram_page_bytes": 4096, "engram_page_heads": 4, "engram_pages": 209716,
        "engram_table": "sparse", "final_logit_softcapping": 15.0, "head_dim": 128, "hidden_size": 2048,
        "intermediate_size": 2048, "kv_sources": ["7": 7, "16": 16, "20": 16, "24": 7, "28": 7], "layer_types": kinds,
        "mlp_layer_types": ["dense"] + [String](repeating: "sparse", count: 31), "moe_intermediate_size": 512,
        "num_attention_heads": 16, "num_experts": 96, "num_experts_per_tok": 8, "num_hidden_layers": 32,
        "num_key_value_heads": 2, "num_prefill_layers": 16, "rms_norm_eps": 0.0078125,
        "rope_parameters": ["partial_rotary_factor": 0.5, "rope_theta": 10000.0, "rope_type": "default"],
        "shared_expert_intermediate_size": 1024, "sliding_num_attention_heads": 16, "sliding_num_key_value_heads": 4,
        "sliding_window": 128, "sparse_recent": 128, "sparse_topk": 1024, "tie_word_embeddings": false,
        "use_cache": true, "use_qk_norm": true, "vocab_size": 65536,
        "quantization": ["bits": 4, "group_size": 64, "mode": "affine"],
    ]
}

func parse(_ c: [String: Any]) throws -> ModelConfig {
    try ModelConfig(json: JSONSerialization.data(withJSONObject: c))
}

@Test func embla1DerivesTheEnginesFormerConstants() throws {
    let c = try parse(embla1Config())
    #expect(c.unknownOptions.isEmpty)
    try Engine.check(c)
    #expect(c.bridges == [7: [24, 28], 16: [20]])
    #expect(c.selects == [16, 20, 24, 28])
    #expect(c.blockFull(17) == 16 && c.blockFull(31) == 28)
    #expect((0 ..< 32).map(c.sources) == (0 ..< 32).map { $0 == 0 ? 1 : 2 + $0 / 4 })
    #expect(c.finalSources == 9 && c.sourceSlots == 9)
    #expect(c.mixSlots == 3)
    #expect(Engine.queryColumns(c) == 69)
    #expect(Engine.moeSlices(c) == 80)
    #expect(Engine.neuronMajorBytes(c) == 96 * 512 * 256 * 4 + 2 * 96 * 8 * 2048 * 2 + 1024 * 256 * 4 + 2 * 16 * 2048 * 2)
    #expect(c.moeLayers == Array(1 ..< 32))
}

@Test func anUnknownOptionIsNotIgnored() throws {
    var j = embla1Config()
    j["some_new_option"] = 3
    let c = try parse(j)
    #expect(c.unknownOptions == ["some_new_option"])
    #expect(throws: EngineError.self) { try Engine.check(c) }
}

@Test func anUnknownLayerKindIsRefused() throws {
    var j = embla1Config()
    j["mlp_layer_types"] = ["dense", "some_new_kind"] + [String](repeating: "sparse", count: 30)
    #expect(throws: EngineError.self) { try parse(j) }
    j = embla1Config()
    j["layer_types"] = ["some_new_kind"] + [String](repeating: "sliding_attention", count: 31)
    #expect(throws: EngineError.self) { try parse(j) }
}

@Test func aShapeTheKernelsDoNotTakeIsRefused() throws {
    var j = embla1Config()
    j["hidden_size"] = 4096
    #expect(throws: EngineError.self) { try Engine.check(parse(j)) }
    j = embla1Config()
    j.removeValue(forKey: "kv_sources")
    #expect(throws: EngineError.self) { try parse(j) }
}
