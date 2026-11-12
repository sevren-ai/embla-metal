// Prefill: the per-token pieces of the prefill half (layers 0 to 15, layer 16's K/V) for a chunk
// of T prompt tokens at positions p0 .. p0 + T - 1, a threadgroup row per token. The matrices go
// through gemm.metal; these kernels are batched counterparts of the decode ones.

constant constexpr int RING = 4096, HRING = 4096, E = 96, TOPK = 8, F = 512;
#ifndef TILE_ENTRIES
#define TILE_ENTRIES 32   // entries of an expert tile: the GEMM's BM
#endif

// The tokens' embedding rows, RMS-normed, into each token's source 0. Grid (1, T), 256 threads.
kernel void embed_b(
    device const uchar*  W     [[buffer(0)]],
    constant ulong3&     off   [[buffer(1)]],
    device const uint*   token [[buffer(2)]],
    device bfloat*       state [[buffer(3)]],   // [T][9][D]
    constant int&        p0    [[buffer(4)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float buf[8];
    const size_t row = token[p0 + int(tg.y)];
    const uint word = ((device const uint*)(W + off.x))[row * (D / 8) + t];
    const float sc = float(((device const bfloat*)(W + off.y))[row * (D / 64) + t / 8]);
    const float bs = float(((device const bfloat*)(W + off.z))[row * (D / 64) + t / 8]);
    float v[8], ss = 0.0f;
    for (int n = 0; n < 8; n++) {
        v[n] = bf(float((word >> (4 * n)) & 0xFu) * sc + bs);
        ss += v[n] * v[n];
    }
    const float r = precise::rsqrt(block_sum<256>(ss, buf, lane, sg) / D + EPS);
    for (int n = 0; n < 8; n++) state[size_t(tg.y) * 9 * D + t * 8 + n] = bfloat(v[n] * r);
}

// Engram, first half: the gate and g·W_v e (to gated) and its RMS norm (to the history ring at the
// token's position). Grid (1, T), 1024 threads.
kernel void engram_gate_b(
    device const bfloat* kv    [[buffer(0)]],   // [T][2][D]: W_k e, W_v e
    device const bfloat* he    [[buffer(1)]],   // [T][3][D]: the engram mix (not normed)
    device bfloat*       gated [[buffer(2)]],   // [T][D]
    device bfloat*       hist  [[buffer(3)]],   // [HRING][D]
    constant int&        p0    [[buffer(4)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = 1024, E2 = D / TG;
    threadgroup float buf[32];
    const int tok = int(tg.y), pos = p0 + tok;
    kv += size_t(tok) * 2 * D;
    he += size_t(tok) * 3 * D;
    float key[E2], val[E2], kk = 0.0f, xx = 0.0f, kx = 0.0f;
    for (int e = 0; e < E2; e++) {
        const int d = int(t) + e * TG;
        key[e] = float(kv[d]);
        val[e] = float(kv[D + d]);
        const float h = float(he[d]);
        kk += key[e] * key[e];
        xx += h * h;
        kx += key[e] * h;
    }
    kk = block_sum<TG>(kk, buf, lane, sg);
    xx = block_sum<TG>(xx, buf, lane, sg);
    kx = block_sum<TG>(kx, buf, lane, sg);
    const float z = kx * precise::rsqrt(kk / D + EPS32) * precise::rsqrt(xx / D + EPS32) / precise::sqrt(float(D));
    const float a = sign(z) * precise::sqrt(fabs(z) + 1e-6f);
    const float g = bf(1.0f / (1.0f + precise::exp(-a)));
    float gv[E2], ss = 0.0f;
    for (int e = 0; e < E2; e++) {
        gv[e] = bf(g * val[e]);
        ss += gv[e] * gv[e];
    }
    const float r = precise::rsqrt(block_sum<TG>(ss, buf, lane, sg) / D + EPS32);
    for (int e = 0; e < E2; e++) {
        const int d = int(t) + e * TG;
        gated[size_t(tok) * D + d] = bfloat(gv[e]);
        hist[size_t(pos % HRING) * D + d] = bfloat(bf(gv[e] * r));
    }
}

// Engram, second half: gated + silu(conv) over the history ring, into source 1. Grid (D / 256, T).
kernel void engram_conv_b(
    device const bfloat* gated [[buffer(0)]],
    device const bfloat* hist  [[buffer(1)]],
    device const bfloat* convw [[buffer(2)]],   // [D, 4]
    device bfloat*       state [[buffer(3)]],   // [T][9][D]
    constant int&        p0    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int d = int(gid.x), tok = int(gid.y), pos = p0 + tok;
    float conv = 0.0f;
    for (int i = 0; i < 4; i++) {
        const int tp = pos - (3 - i) * 3;
        const float v = tp >= 0 ? float(hist[size_t(tp % HRING) * D + d]) : 0.0f;
        conv = conv + v * float(convw[d * 4 + i]);
    }
    const float silu = bf(conv / (1.0f + precise::exp(-conv)));
    state[size_t(tok) * 9 * D + D + d] = bfloat(float(gated[size_t(tok) * D + d]) + silu);
}

// The sliding layers' new keys (normed with the k gains, rotated) and values into the ring. Grid
// (4 kv heads, T), 32 threads.
kernel void kv_sliding_b(
    device const bfloat* qkv [[buffer(0)]],   // [T][4096]: q, k (at 2048), v (at 2560)
    device const bfloat* kg  [[buffer(1)]],
    device bfloat*       kc  [[buffer(2)]],   // [4][RING][128]
    device bfloat*       vc  [[buffer(3)]],
    constant int&        p0  [[buffer(4)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int kvh = int(tg.x), tok = int(tg.y), pos = p0 + tok, slot = pos % RING;
    device const bfloat* row = qkv + size_t(tok) * 4096;
    float k[4];
    for (int i = 0; i < 4; i++) k[i] = float(row[16 * HD + kvh * HD + lane + 32 * i]);
    head_norm(k, kg, kg, 1, lane);
    rope(k, pos, lane);
    for (int i = 0; i < 4; i++) {
        const int d = int(lane) + 32 * i;
        kc[(kvh * RING + slot) * HD + d] = bfloat(k[i]);
        vc[(kvh * RING + slot) * HD + d] = row[20 * HD + kvh * HD + d];
    }
}

// Sliding attention of each token over its last 128 positions (all in the ring by now). Grid
// (16 heads, T), 128 threads: as attn_sliding.
kernel void attn_sliding_b(
    device const bfloat* qkv [[buffer(0)]],
    device const bfloat* qg  [[buffer(1)]],
    device const bfloat* kc  [[buffer(2)]],
    device const bfloat* vc  [[buffer(3)]],
    device bfloat*       o   [[buffer(4)]],   // [T][D]
    constant int&        p0  [[buffer(5)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr int W = 128, NS = 4;
    const int h = int(tg.x), tok = int(tg.y), pos = p0 + tok, kvh = h / 4;
    float q[4];
    for (int i = 0; i < 4; i++) q[i] = float(qkv[size_t(tok) * 4096 + h * HD + lane + 32 * i]);
    head_norm(q, qg, qg, 1, lane);
    rope(q, pos, lane);
    for (int i = 0; i < 4; i++) q[i] = bf(q[i]);
    const int n = min(pos + 1, W);
    const float scale = rsqrt(float(HD));
    float m = NEG, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = int(sg); i < n; i += NS) {
        const int slot = (pos - i) % RING;
        float dot = 0.0f, vv[4];
        for (int e = 0; e < 4; e++) {
            const int d = int(lane) + 32 * e;
            dot += q[e] * float(kc[(kvh * RING + slot) * HD + d]);
            vv[e] = float(vc[(kvh * RING + slot) * HD + d]);
        }
        const float s = simd_sum(dot) * scale;
        const float mn = max(m, s), c = exp(m - mn), p = exp(s - mn);
        l = l * c + p;
        for (int e = 0; e < 4; e++) acc[e] = acc[e] * c + p * vv[e];
        m = mn;
    }
    threadgroup float tm[NS], tl[NS], ta[NS][HD];
    if (lane == 0) { tm[sg] = m; tl[sg] = l; }
    for (int e = 0; e < 4; e++) ta[sg][lane + 32 * e] = acc[e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float M = NEG;
        for (int s = 0; s < NS; s++) M = max(M, tm[s]);
        float L = 0.0f, f[NS];
        for (int s = 0; s < NS; s++) { f[s] = exp(tm[s] - M); L += tl[s] * f[s]; }
        for (int e = 0; e < 4; e++) {
            const int d = int(lane) + 32 * e;
            float r = 0.0f;
            for (int s = 0; s < NS; s++) r += ta[s][d] * f[s];
            o[size_t(tok) * D + h * HD + d] = bfloat(r / L);
        }
    }
}

// The full layers' new keys (normed, no gains) and values. Grid (2 · jobs, T), 32 threads.
struct KVJob { uint k, v; uint kc, vc; };
kernel void kv_append_b(
    device const bfloat* qkv   [[buffer(0)]],
    device kv_t*         cache [[buffer(1)]],
    constant KVJob*      jobs  [[buffer(2)]],
    constant int2&       pm    [[buffer(3)]],   // p0, maxS
    uint2 tg  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const KVJob job = jobs[tg.x / 2];
    const int head = int(tg.x % 2), tok = int(tg.y), pos = pm.x + tok;
    device const bfloat* row = qkv + size_t(tok) * 4096;
    float k[4], v[4];
    for (int i = 0; i < 4; i++) {
        k[i] = float(row[job.k + head * HD + lane + 32 * i]);
        v[i] = float(row[job.v + head * HD + lane + 32 * i]);
    }
    head_norm(k, nullptr, nullptr, 0, lane);
    const size_t r = size_t(head) * pm.y + pos;
#if KV8
    write_row8(cache + job.kc, r, pm.y, k, lane);
    write_row8(cache + job.vc, r, pm.y, v, lane);
#else
    for (int i = 0; i < 4; i++) {
        const int d = int(lane) + 32 * i;
        cache[job.kc + r * HD + d] = bfloat(k[i]);
        cache[job.vc + r * HD + d] = bfloat(v[i]);
    }
#endif
}

// ReLU² attention of each token over positions 0 .. its own (layer 7). Grid (C, 2, T), 256
// threads: a threadgroup is (chunk of CH keys, kv head, token), its 8 simdgroups the query heads;
// each (token, kv head)'s last threadgroup adds the chunks. Chunks past a token's position add 0.
struct AttnB { int p0, CH, C, maxS; };
kernel void attn_full_b(
    device const bfloat* qkv     [[buffer(0)]],
    device const bfloat* g1      [[buffer(1)]],
    device const bfloat* g2      [[buffer(2)]],
    device const kv_t*   kc      [[buffer(3)]],
    device const kv_t*   vc      [[buffer(4)]],
    device float*        part    [[buffer(5)]],   // [T][C][16][HD + 1]
    device atomic_uint*  counter [[buffer(6)]],   // [T][2]
    device bfloat*       o       [[buffer(7)]],   // [T][D]
    constant AttnB&      p       [[buffer(8)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  t    [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int c = int(tg.x), kvh = int(tg.y), tok = int(tg.z), h = kvh * 8 + int(sg);
    const int S = p.p0 + tok + 1;
    float q[4];
    for (int i = 0; i < 4; i++) q[i] = float(qkv[size_t(tok) * 4096 + h * HD + lane + 32 * i]);
    head_norm(q, g1, g2, 2, lane);
    for (int i = 0; i < 4; i++) q[i] = bf(q[i]);
    device const kv_t* kb = kc + size_t(kvh) * p.maxS * HD;
    device const kv_t* vb = vc + size_t(kvh) * p.maxS * HD;
    const int j0 = c * p.CH, j1 = min(S, j0 + p.CH);
    const float scale = rsqrt(float(HD));
    float psum = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#if KV8
    device const float* ks = kv_scales(kc, p.maxS) + kvh * p.maxS;
    device const float* vs = kv_scales(vc, p.maxS) + kvh * p.maxS;
    for (int j = j0; j < j1; j++) {
        float dot = 0.0f;
        for (int i = 0; i < 4; i++) dot += q[i] * float(kb[size_t(j) * HD + lane + 32 * i]);
        const float a = max(simd_sum(dot) * ks[j] * scale, 0.0f), w = a * a;
        if (w != 0.0f) {
            psum += w;
            const float wv = w * vs[j];
            for (int i = 0; i < 4; i++) acc[i] += wv * float(vb[size_t(j) * HD + lane + 32 * i]);
        }
    }
#else
    for (int j = j0; j < j1; j++) {
        float dot = 0.0f;
        for (int i = 0; i < 4; i++) dot += q[i] * float(kb[size_t(j) * HD + lane + 32 * i]);
        const float a = max(simd_sum(dot) * scale, 0.0f), w = a * a;
        if (w != 0.0f) {
            psum += w;
            for (int i = 0; i < 4; i++) acc[i] += w * float(vb[size_t(j) * HD + lane + 32 * i]);
        }
    }
#endif
    device float* pp = part + ((size_t(tok) * p.C + c) * 16 + h) * (HD + 1);
    for (int i = 0; i < 4; i++) pp[lane + 32 * i] = acc[i];
    if (lane == 0) pp[HD] = psum;

    threadgroup bool last;
    if (!arrive_last(counter + tok * 2 + kvh, uint(p.C), t, &last)) return;
    const int hh = kvh * 8 + int(t) / 32, d0 = (int(t) % 32) * 4;
    float den = sqrt(float(HD)), num[4] = {0, 0, 0, 0};
    for (int cc = 0; cc < p.C; cc++) {
        device const float* q2 = part + ((size_t(tok) * p.C + cc) * 16 + hh) * (HD + 1);
        den += q2[HD];
        for (int n = 0; n < 4; n++) num[n] += q2[d0 + n];
    }
    for (int n = 0; n < 4; n++) o[size_t(tok) * D + hh * HD + d0 + n] = bfloat(num[n] / den);
}

// Router logits of each token (bf16 weights, f32 sums). Grid (12, T), 256 threads.
kernel void router_b(
    device const bfloat* w      [[buffer(0)]],   // [E][D]
    device const bfloat* x      [[buffer(1)]],   // [T][3][D]: the MLP input mix
    device float*        logits [[buffer(2)]],   // [T][E]
    uint2 tg  [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int row = int(tg.x) * 8 + int(sg), tok = int(tg.y);
    device const bfloat4* wp = (device const bfloat4*)(w + row * D);
    device const bfloat4* xp = (device const bfloat4*)(x + size_t(tok) * 3 * D);
    float acc = 0.0f;
    for (int i = 0; i < 16; i++) acc += dot(float4(wp[i * 32 + lane]), float4(xp[i * 32 + lane]));
    acc = simd_sum(acc);
    if (lane == 0) logits[tok * E + row] = acc;
}

// Routing of each token (as moe_router's last threadgroup): route [T][8] experts, gates [T][8].
// Grid T simdgroups (32 threads each).
kernel void route_b(
    device const float*  logits [[buffer(0)]],
    device const bfloat* bias   [[buffer(1)]],
    device int*          experts [[buffer(2)]],
    device float*        gates  [[buffer(3)]],
    uint tok  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr int RR = E / 32;
    float score[RR], biased[RR];
    for (int i = 0; i < RR; i++) {
        const int e = int(lane) + 32 * i;
        const float l = logits[tok * E + e];
        const float y = 1.0f / (1.0f + precise::exp(fabs(l)));
        score[i] = l < 0 ? y : 1.0f - y;
        biased[i] = score[i] + float(bias[e]);
    }
    float wk[TOPK], total = 0.0f;
    for (int k = 0; k < TOPK; k++) {
        float best = NEG;
        int best_e = E;
        for (int i = 0; i < RR; i++) {
            const int e = int(lane) + 32 * i;
            if (biased[i] > best || (biased[i] == best && e < best_e)) { best = biased[i]; best_e = e; }
        }
        const float top = simd_max(best);
        const int winner = simd_min(best == top ? best_e : E);
        float sc = 0.0f;
        for (int i = 0; i < RR; i++) {
            if (int(lane) + 32 * i == winner) { sc = score[i]; biased[i] = NEG; }
        }
        wk[k] = simd_sum(sc);
        if (lane == 0) experts[tok * TOPK + k] = winner;
    }
    for (int k = 0; k < TOPK; k++) total = wk[k] + total;
    if (lane == 0) {
        for (int k = 0; k < TOPK; k++) gates[tok * TOPK + k] = bf(wk[k] / total);
    }
}

// The routed (token, slot) entries grouped by expert, in token order within an expert: entry_tok
// [T·8] (the X row of each entry), slot_entry [T·8] (where (token, slot) went), tiles of up to
// TILE_ENTRIES entries of one expert (for gemm EXPERTS) and their count. One threadgroup of 128 threads; thread
// e < 96 owns expert e.
kernel void group_b(
    device const int* experts    [[buffer(0)]],   // [T][8]
    device int*       entry_tok  [[buffer(1)]],
    device int*       slot_entry [[buffer(2)]],
    device int4*      tiles      [[buffer(3)]],
    device int*       ntiles     [[buffer(4)]],
    constant int&     T          [[buffer(5)]],
    uint t [[thread_index_in_threadgroup]])
{
    threadgroup int count[E], start[E], tstart[E];
    const int n = T * TOPK;
    if (int(t) < E) {
        int c = 0;
        for (int i = 0; i < n; i++) c += experts[i] == int(t);
        count[t] = c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        int s = 0, ts = 0;
        for (int e = 0; e < E; e++) {
            start[e] = s;
            tstart[e] = ts;
            s += count[e];
            ts += (count[e] + TILE_ENTRIES - 1) / TILE_ENTRIES;
        }
        ntiles[0] = ts;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) < E) {
        int at = start[t];
        for (int i = 0; i < n; i++) {
            if (experts[i] == int(t)) {
                entry_tok[at] = i / TOPK;
                slot_entry[i] = at;
                at++;
            }
        }
        for (int j = 0; j < (count[t] + TILE_ENTRIES - 1) / TILE_ENTRIES; j++)
            tiles[tstart[t] + j] = int4(int(t), start[t] + TILE_ENTRIES * j, min(TILE_ENTRIES, count[t] - TILE_ENTRIES * j), 0);
    }
}

// x += bf16(shared + Σ gate · expert output) for each token, the experts' outputs in bf16 as the reference
// rounds them. Grid (D / 256, T).
kernel void moe_combine_b(
    device const bfloat* ysh       [[buffer(0)]],   // [T][D] shared expert
    device const bfloat* yex       [[buffer(1)]],   // [T·8][D] by entry
    device const float* gates      [[buffer(2)]],   // [T][8]
    device const int*   slot_entry [[buffer(3)]],
    device float*       x          [[buffer(4)]],   // [T][D]
    uint2 gid [[thread_position_in_grid]])
{
    const int d = int(gid.x), tok = int(gid.y);
    float s = float(ysh[size_t(tok) * D + d]);
    for (int k = 0; k < TOPK; k++) s += gates[tok * TOPK + k] * float(yex[size_t(slot_entry[tok * TOPK + k]) * D + d]);
    x[size_t(tok) * D + d] += bf(s);
}

// The full layer's queries for attn_nax: normed with both gains, rounded to bf16, head-major
// [16][T][128]. Grid (16, T), 32 threads.
kernel void q_prep_b(
    device const bfloat* qkv [[buffer(0)]],
    device const bfloat* g1  [[buffer(1)]],
    device const bfloat* g2  [[buffer(2)]],
    device bfloat*       qn  [[buffer(3)]],
    constant int&        T   [[buffer(4)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int h = int(tg.x), tok = int(tg.y);
    float q[4];
    for (int i = 0; i < 4; i++) q[i] = float(qkv[size_t(tok) * 4096 + h * HD + lane + 32 * i]);
    head_norm(q, g1, g2, 2, lane);
#if QK8   // measurement (attn_nax.metal): int8 rows [16][T][128], then their scales [16][T]
    write_row8((device char*)qn, size_t(h) * T + tok, 8 * T, q, lane);
#else
    for (int i = 0; i < 4; i++) qn[(size_t(h) * T + tok) * HD + lane + 32 * i] = bfloat(q[i]);
#endif
}

// attn_sliding_b on simdgroup matrices: a threadgroup takes one kv head and two consecutive tokens,
// so its 8 query rows (row r: head 4 · kvh + r % 4, token tok0 + r / 4) share the 129-key window,
// read once (one threadgroup a query head and token read it 8 times over). As attn_sparse8: Q's
// head dims permuted (qperm) so a lane loads its share of 8 keys as contiguous runs, S = Q Kᵀ and
// O += W V over 8 keys at a time with an online softmax in the fragments, each row masked to its own
// window, the simdgroups' (max, sum, O) merged at the end. Grid (4, ceil(T / 2)), 128 threads.
inline short2 frag_coord_b(uint lane) {
    const short qid = short(lane / 4);
    return short2((qid & 4) + short(lane / 2) % 4, (qid & 2) * 2 + short(lane % 2) * 2);
}

kernel void attn_sliding8_b(
    device const bfloat* qkv [[buffer(0)]],
    device const bfloat* qg  [[buffer(1)]],
    device const bfloat* kc  [[buffer(2)]],   // [4][RING][128]
    device const bfloat* vc  [[buffer(3)]],
    device bfloat*       o   [[buffer(4)]],   // [T][D]
    constant int&        p0  [[buffer(5)]],
    constant int&        T   [[buffer(6)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  t    [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr int W = 128, NSG = 4;
    const int kvh = int(tg.x), tok0 = 2 * int(tg.y), pos0 = p0 + tok0;
    threadgroup float qs[8][HD];
    threadgroup float ot[NSG][8][HD];
    threadgroup float msg[NSG][8], lsg[NSG][8];
    for (int r = int(sg); r < 8; r += NSG) {
        const int tok = min(tok0 + r / 4, T - 1), h = kvh * 4 + r % 4;
        float q[4];
        for (int i = 0; i < 4; i++) q[i] = float(qkv[size_t(tok) * 4096 + h * HD + lane + 32 * i]);
        head_norm(q, qg, qg, 1, lane);
        rope(q, p0 + tok, lane);
        for (int i = 0; i < 4; i++) qs[r][qperm(int(lane) + 32 * i)] = bf(q[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Q[16], O[16];
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 16; k++) {
        simdgroup_load(Q[k], &qs[0][8 * k], HD);
        O[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const short2 fc = frag_coord_b(lane);
    const int fm = fc.x, fn = fc.y;
    // the window of the pair: keys lo ..= pos0 + 1; row fm's own: its position - 127 ..= its position
    const int lo = max(0, pos0 - (W - 1)), hi = pos0 + 1, nk = hi - lo + 1;
    const int rpos = pos0 + fm / 4;
    device const bfloat* kb = kc + size_t(kvh) * RING * HD;
    device const bfloat* vb = vc + size_t(kvh) * RING * HD;
    const float scale = rsqrt(float(HD));
    float m = NEG, l = 0.0f;
    for (int b = 8 * int(sg); b < nk; b += 8 * NSG) {
        const int pa = lo + b + fn, pb = pa + 1;
        const bool va_ = pa <= rpos && pa > rpos - W && b + fn < nk;
        const bool vb_ = pb <= rpos && pb > rpos - W && b + fn + 1 < nk;
        // K rows of keys pa and pb, dims 16 fm .. 16 fm + 15 (Q's slice k holds dims 16 fm + k)
        device const uint4* ka = (device const uint4*)(kb + size_t(min(pa, hi) % RING) * HD + 16 * fm);
        device const uint4* kbp = (device const uint4*)(kb + size_t(min(pb, hi) % RING) * HD + 16 * fm);
        const uint4 a0 = ka[0], a1 = ka[1], b0 = kbp[0], b1 = kbp[1];
        // V row of key b + fm, dims (fn / 2) · 32 .. + 31 (O's slice n column c: dim (c / 2) · 32 + 2n + c % 2)
        const int pv = min(lo + b + fm, hi);
        device const uint4* vr = (device const uint4*)(vb + size_t(pv % RING) * HD + (fn / 2) * 32);
        const uint4 v0 = vr[0], v1 = vr[1], v2 = vr[2], v3 = vr[3];
        simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        _Pragma("clang loop unroll(full)")
        for (int k = 0; k < 16; k++) {
            const uint4 qa = k < 8 ? a0 : a1, qb = k < 8 ? b0 : b1;
            const uint wa = qa[(k % 8) / 2], wb = qb[(k % 8) / 2];
            simdgroup_float8x8 KT;
            KT.thread_elements()[0] = float(as_type<bfloat2>(wa)[k % 2]);
            KT.thread_elements()[1] = float(as_type<bfloat2>(wb)[k % 2]);
            simdgroup_multiply_accumulate(S, Q[k], KT, S);
        }
        const float s0 = va_ ? S.thread_elements()[0] * scale : NEG;
        const float s1 = vb_ ? S.thread_elements()[1] * scale : NEG;
        float bm = max(s0, s1);
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        const float mn = max(m, bm), cf = exp(m - mn);
        const float w0 = va_ ? exp(s0 - mn) : 0.0f, w1 = vb_ ? exp(s1 - mn) : 0.0f;
        l = l * cf + w0 + w1;
        m = mn;
        simdgroup_float8x8 Wm;
        Wm.thread_elements()[0] = w0;
        Wm.thread_elements()[1] = w1;
        const uint vw[16] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w, v2.x, v2.y, v2.z, v2.w, v3.x, v3.y, v3.z, v3.w};
        _Pragma("clang loop unroll(full)")
        for (int n = 0; n < 16; n++) {
            O[n].thread_elements()[0] *= cf;
            O[n].thread_elements()[1] *= cf;
            const bfloat2 b2 = as_type<bfloat2>(vw[n]);
            simdgroup_float8x8 V;
            V.thread_elements()[0] = float(b2[0]);
            V.thread_elements()[1] = float(b2[1]);
            simdgroup_multiply_accumulate(O[n], Wm, V, O[n]);
        }
    }
    l += simd_shuffle_xor(l, 1);
    l += simd_shuffle_xor(l, 8);
    if ((lane & 9) == 0) {
        msg[sg][fm] = m;
        lsg[sg][fm] = l;
    }
    _Pragma("clang loop unroll(full)")
    for (int n = 0; n < 16; n++) simdgroup_store(O[n], &ot[sg][0][8 * n], HD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // merged: thread t takes row t / 16, dims (t % 16) · 8 ..
    const int r = int(t) / 16, d0 = (int(t) % 16) * 8, tok = tok0 + r / 4, h = kvh * 4 + r % 4;
    if (tok >= T) return;
    float M = NEG;
    for (int g = 0; g < NSG; g++) M = max(M, msg[g][r]);
    float L = 0.0f, num[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < NSG; g++) {
        const float f = exp(msg[g][r] - M);
        L += lsg[g][r] * f;
        for (int e = 0; e < 8; e++) num[e] += ot[g][r][ocol(d0 + e)] * f;
    }
    for (int e = 0; e < 8; e++) o[size_t(tok) * D + h * HD + d0 + e] = bfloat(num[e] / L);
}
