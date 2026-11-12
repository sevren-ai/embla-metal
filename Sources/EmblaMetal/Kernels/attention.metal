// Decode attention, one query token. Heads are held 4 dims a lane (dims lane + 32 i) and a query
// head is one simdgroup. Caches are bf16 [kv head][position][128].

// The sliding layers' caches are rings of RING positions (a prefill chunk's keys must fit beside
// the window before its attention runs); the window is 128.
constant constexpr int RING = 4096;

struct AttnParams {
    int S;      // keys (the current one included), or selected rows
    int CH;     // keys per chunk (threadgroup)
    int C;      // chunks
    int maxS;   // cache rows per kv head
    int pos;    // position of the current token
    int ng;     // query gains (1 or 2)
    int pad0, pad1;
};

// --------------------------------------------------------------------------- sliding
// Softmax over the last 128 positions (the current one included), RoPE on q and k, 16 query heads
// on 4 kv heads. One threadgroup (4 simdgroups splitting the keys) per query head; the first
// threadgroup of each kv head writes the new k and v into the ring of 128.

#ifndef NS
#define NS 16
#endif
kernel void attn_sliding(
    device const bfloat* qkv [[buffer(0)]],   // q [16·128], k [4·128], v [4·128]
    device const bfloat* qg  [[buffer(1)]],
    device const bfloat* kg  [[buffer(2)]],
    device bfloat*       kc  [[buffer(3)]],   // [4][RING][128]
    device bfloat*       vc  [[buffer(4)]],
    device bfloat*       o   [[buffer(5)]],
    constant int&        pos [[buffer(6)]],
    uint h    [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    constexpr int W = 128;
    const int kvh = int(h) / 4;
    float q[4], k[4], v[4];
    for (int i = 0; i < 4; i++) {
        const int d = int(lane) + 32 * i;
        q[i] = float(qkv[h * HD + d]);
        k[i] = float(qkv[16 * HD + kvh * HD + d]);
        v[i] = float(qkv[20 * HD + kvh * HD + d]);
    }
    head_norm(q, qg, qg, 1, lane);
    rope(q, pos, lane);
    head_norm(k, kg, kg, 1, lane);
    rope(k, pos, lane);
    for (int i = 0; i < 4; i++) {
        q[i] = bf(q[i]);
        k[i] = bf(k[i]);
    }
    if (h % 4 == 0 && sg == 0) {
        const int slot = pos % RING;
        for (int i = 0; i < 4; i++) {
            const int d = int(lane) + 32 * i;
            kc[(kvh * RING + slot) * HD + d] = bfloat(k[i]);
            vc[(kvh * RING + slot) * HD + d] = bfloat(v[i]);
        }
    }
    const int n = min(pos + 1, W);
    const float scale = rsqrt(float(HD));
    float m = NEG, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = int(sg); i < n; i += NS) {
        float kk[4], vv[4];
        if (i == 0) {
            for (int e = 0; e < 4; e++) { kk[e] = k[e]; vv[e] = v[e]; }
        } else {
            const int slot = (pos - i) % RING;
            for (int e = 0; e < 4; e++) {
                const int d = int(lane) + 32 * e;
                kk[e] = float(kc[(kvh * RING + slot) * HD + d]);
                vv[e] = float(vc[(kvh * RING + slot) * HD + d]);
            }
        }
        float dot = 0.0f;
        for (int e = 0; e < 4; e++) dot += q[e] * kk[e];
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
            o[h * HD + d] = bfloat(r / L);
        }
    }
}

// --------------------------------------------------------------------------- k/v append
// The full layers' new keys (RMS-normed per head, no gains) and values, for the layer itself and
// the later layers whose K/V project the same layer's input. One simdgroup per (job, kv head).

struct KVJob {
    uint k, v;     // element offsets of the raw k and v projections in qkv
    uint kc, vc;   // offsets of the layer's k and v caches in `cache` (elements, bytes with KV8)
};

kernel void kv_append(
    device const bfloat* qkv   [[buffer(0)]],
    device kv_t*         cache [[buffer(1)]],
    constant KVJob*      jobs  [[buffer(2)]],
    constant int2&       pm    [[buffer(3)]],   // pos, maxS
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const KVJob job = jobs[tg / 2];
    const int head = int(tg % 2);
    float k[4], v[4];
    for (int i = 0; i < 4; i++) {
        k[i] = float(qkv[job.k + head * HD + lane + 32 * i]);
        v[i] = float(qkv[job.v + head * HD + lane + 32 * i]);
    }
    head_norm(k, nullptr, nullptr, 0, lane);
    const size_t row = size_t(head) * pm.y + pm.x;
#if KV8
    write_row8(cache + job.kc, row, pm.y, k, lane);
    write_row8(cache + job.vc, row, pm.y, v, lane);
#else
    for (int i = 0; i < 4; i++) {
        const int d = int(lane) + 32 * i;
        cache[job.kc + row * HD + d] = bfloat(k[i]);
        cache[job.vc + row * HD + d] = bfloat(v[i]);
    }
#endif
}

// --------------------------------------------------------------------------- full (ReLU²)
// relu(q·k/√D)² over √D + Σ, no position encoding, split over chunks of keys: a threadgroup is
// (chunk, kv head), its 8 simdgroups the 8 query heads. ReLU² attention is linear in its weights,
// so the chunks' sums just add; the last threadgroup to finish divides and writes o.
// SCORES: also write each key's max over the 8 heads of q·k (the sparse layers' selection).

kernel void attn_full(
    device const bfloat* qraw    [[buffer(0)]],
    device const bfloat* g1      [[buffer(1)]],
    device const bfloat* g2      [[buffer(2)]],
    device const bfloat* kc      [[buffer(3)]],
    device const bfloat* vc      [[buffer(4)]],
    device float*        part    [[buffer(5)]],   // [C][16][HD + 1]
    device float*        scores  [[buffer(6)]],   // [2][maxS]
    device atomic_uint*  counter [[buffer(7)]],
    device bfloat*       o       [[buffer(8)]],
    constant AttnParams& p       [[buffer(9)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  t    [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr int MAXCH = 512, U = 4;
    const int c = int(tg.x), kvh = int(tg.y), h = kvh * 8 + int(sg);
    float q[4];
    for (int i = 0; i < 4; i++) q[i] = float(qraw[h * HD + lane + 32 * i]);
    head_norm(q, g1, g2, p.ng, lane);
    for (int i = 0; i < 4; i++) q[i] = bf(q[i]);

    device const bfloat* kb = kc + size_t(kvh) * p.maxS * HD;
    device const bfloat* vb = vc + size_t(kvh) * p.maxS * HD;
    const int j0 = c * p.CH, j1 = min(p.S, j0 + p.CH);
    const float scale = rsqrt(float(HD));
    threadgroup float smax[8][MAXCH];
    float psum = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // U keys at a time, their loads first; a value row is read only where the weight is not zero
    // (V-skip: at long contexts nearly all ReLU² weights are)
    for (int jb = j0; jb < j1; jb += U) {
        float dot[U];
        for (int u = 0; u < U; u++) {
            const int j = min(jb + u, j1 - 1);
            dot[u] = 0.0f;
            for (int i = 0; i < 4; i++) dot[u] += q[i] * float(kb[size_t(j) * HD + lane + 32 * i]);
        }
        for (int u = 0; u < U && jb + u < j1; u++) {
            const int j = jb + u;
            const float s = simd_sum(dot[u]);
            if (SCORES && lane == 0) smax[sg][j - j0] = s;
            const float a = max(s * scale, 0.0f), w = a * a;
            if (w != 0.0f) {
                psum += w;
                for (int i = 0; i < 4; i++) acc[i] += w * float(vb[size_t(j) * HD + lane + 32 * i]);
            }
        }
    }
    device float* pp = part + (size_t(c) * 16 + h) * (HD + 1);
    for (int i = 0; i < 4; i++) pp[lane + 32 * i] = acc[i];
    if (lane == 0) pp[HD] = psum;
    if (SCORES) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = int(t); j < j1 - j0; j += 256) {
            float m = smax[0][j];
            for (int s = 1; s < 8; s++) m = max(m, smax[s][j]);
            scores[size_t(kvh) * p.maxS + j0 + j] = m;
        }
    }

    // the last threadgroup of each kv head adds its chunks for its 8 heads
    threadgroup bool last;
    if (!arrive_last(counter + kvh, uint(p.C), t, &last)) return;
    const int hh = kvh * 8 + int(t) / 32, d0 = (int(t) % 32) * 4;
    float den = sqrt(float(HD)), num[4] = {0, 0, 0, 0};
    for (int cc = 0; cc < p.C; cc++) {
        device const float* q2 = part + (size_t(cc) * 16 + hh) * (HD + 1);
        den += q2[HD];
        for (int n = 0; n < 4; n++) num[n] += q2[d0 + n];
    }
    for (int n = 0; n < 4; n++) o[hh * HD + d0 + n] = bfloat(num[n] / den);
}

// KV8: the same on simdgroup matrices, the int8 rows loaded straight into the 8x8 fragments (a
// lane's bytes are contiguous runs; Q and O take the head dims in that order, qperm / ocol). A
// threadgroup is (chunk of CH keys, kv head),
// its 4 simdgroups take 8 keys at a time: S = Q K^T for the 8 query heads (16 multiplies over the
// head dim), w = relu(S·scale_k/√D)², and unless all 64 are zero (V-skip) O += (w·scale_v) V.

/// The (row, column) of the two elements [0], [1] (columns fn, fn + 1) a lane holds of an 8x8
/// simdgroup matrix.
inline short2 frag_coord(uint lane) {
    const short qid = short(lane / 4);
    return short2((qid & 4) + short(lane / 2) % 4, (qid & 2) * 2 + short(lane % 2) * 2);
}

kernel void attn_full8(
    device const bfloat* qraw    [[buffer(0)]],
    device const bfloat* g1      [[buffer(1)]],
    device const bfloat* g2      [[buffer(2)]],
    device const char*   kc      [[buffer(3)]],
    device const char*   vc      [[buffer(4)]],
    device float*        part    [[buffer(5)]],   // [C][16][HD + 1]
    device float*        scores  [[buffer(6)]],   // [2][maxS]
    device atomic_uint*  counter [[buffer(7)]],
    device bfloat*       o       [[buffer(8)]],
    constant AttnParams& p       [[buffer(9)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  t    [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr int NSG = 4;
    const int c = int(tg.x), kvh = int(tg.y);
    threadgroup float qs[8][HD];
    threadgroup float ot[NSG][8][HD];
    threadgroup float sw[NSG][8][8];
    threadgroup float pl[NSG][32];
    for (int hh = int(sg); hh < 8; hh += NSG) {
        float q[4];
        for (int i = 0; i < 4; i++) q[i] = float(qraw[(kvh * 8 + hh) * HD + lane + 32 * i]);
        head_norm(q, g1, g2, p.ng, lane);
        for (int i = 0; i < 4; i++) qs[hh][qperm(int(lane) + 32 * i)] = bf(q[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Q[16], O[16];
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 16; k++) {
        simdgroup_load(Q[k], &qs[0][8 * k], HD);
        O[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const short2 fc = frag_coord(lane);
    const int fm = fc.x, fn = fc.y;
    device const char* kb = kc + size_t(kvh) * p.maxS * HD;
    device const char* vb = vc + size_t(kvh) * p.maxS * HD;
    device const float* ks = kv_scales(kc, p.maxS) + kvh * p.maxS;
    device const float* vs = kv_scales(vc, p.maxS) + kvh * p.maxS;
    const int j0 = c * p.CH, j1 = min(p.S, j0 + p.CH);
    const float scale = rsqrt(float(HD));
    float psum = 0.0f;
    for (int b = j0 + 8 * int(sg); b < j1; b += 8 * NSG) {
        // K^T fragments: element (row fm of slice k, key fn + u) is byte 16 fm + k of key b + fn + u
        const int ja = b + fn, jb = ja + 1;
        const uint4 ra = ja < j1 ? *(device const uint4*)(kb + size_t(ja) * HD + fm * 16) : uint4(0);
        const uint4 rb = jb < j1 ? *(device const uint4*)(kb + size_t(jb) * HD + fm * 16) : uint4(0);
        const float sa = ja < j1 ? ks[ja] : 0.0f, sb = jb < j1 ? ks[jb] : 0.0f;
        simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        _Pragma("clang loop unroll(full)")
        for (int k = 0; k < 16; k++) {
            simdgroup_float8x8 KT;
            KT.thread_elements()[0] = float(as_type<char4>(ra[k / 4])[k % 4]);
            KT.thread_elements()[1] = float(as_type<char4>(rb[k / 4])[k % 4]);
            simdgroup_multiply_accumulate(S, Q[k], KT, S);
        }
        // S's elements here: head fm, keys fn and fn + 1
        const float s0 = S.thread_elements()[0] * sa, s1 = S.thread_elements()[1] * sb;
        if (SCORES) {
            sw[sg][fm][fn] = s0;
            sw[sg][fm][fn + 1] = s1;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 8 && b + int(lane) < j1) {
                float m = sw[sg][0][lane];
                for (int r = 1; r < 8; r++) m = max(m, sw[sg][r][lane]);
                scores[size_t(kvh) * p.maxS + b + lane] = m;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float a0 = max(s0 * scale, 0.0f), a1 = max(s1 * scale, 0.0f);
        const float w0 = a0 * a0, w1 = a1 * a1;
        psum += w0 + w1;
        if (!simd_any(w0 != 0.0f || w1 != 0.0f)) continue;
        simdgroup_float8x8 W;
        W.thread_elements()[0] = w0 * (ja < j1 ? vs[ja] : 0.0f);
        W.thread_elements()[1] = w1 * (jb < j1 ? vs[jb] : 0.0f);
        // V fragments: element (key fm, column fn + u of slice n) is byte (fn / 2) · 32 + 2n + u
        const int jv = b + fm;
        uint4 v0 = uint4(0), v1 = uint4(0);
        if (jv < j1) {
            device const uint4* vr = (device const uint4*)(vb + size_t(jv) * HD + (fn / 2) * 32);
            v0 = vr[0];
            v1 = vr[1];
        }
        _Pragma("clang loop unroll(full)")
        for (int n = 0; n < 16; n++) {
            const uint wd = n < 8 ? v0[n / 2] : v1[(n - 8) / 2];
            const char4 b4 = as_type<char4>(wd);
            simdgroup_float8x8 V;
            V.thread_elements()[0] = float(b4[(n % 2) * 2]);
            V.thread_elements()[1] = float(b4[(n % 2) * 2 + 1]);
            simdgroup_multiply_accumulate(O[n], W, V, O[n]);
        }
    }
    pl[sg][lane] = psum;
    _Pragma("clang loop unroll(full)")
    for (int n = 0; n < 16; n++) simdgroup_store(O[n], &ot[sg][0][8 * n], HD);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the simdgroups' sums into this chunk's partial, thread t: head t / 16, dims (t % 16) · 8 ..
    const int th = int(t) / 16, d0 = (int(t) % 16) * 8;
    device float* pp = part + (size_t(c) * 16 + kvh * 8 + th) * (HD + 1);
    for (int e = 0; e < 8; e++) {
        float v = 0.0f;
        for (int g = 0; g < NSG; g++) v += ot[g][th][ocol(d0 + e)];
        pp[d0 + e] = v;
    }
    if (t % 16 == 0) {
        float v = 0.0f;
        for (int g = 0; g < NSG; g++) {
            for (uint l = 0; l < 32; l++) v += frag_coord(l).x == th ? pl[g][l] : 0.0f;
        }
        pp[HD] = v;
    }

    // the last threadgroup of each kv head adds its chunks for its 8 heads
    threadgroup bool last;
    if (!arrive_last(counter + kvh, uint(p.C), t, &last)) return;
    float den = sqrt(float(HD)), num[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int cc = 0; cc < p.C; cc++) {
        device const float* q2 = part + (size_t(cc) * 16 + kvh * 8 + th) * (HD + 1);
        den += q2[HD];
        for (int e = 0; e < 8; e++) num[e] += q2[d0 + e];
    }
    for (int e = 0; e < 8; e++) o[(kvh * 8 + th) * HD + d0 + e] = bfloat(num[e] / den);
}

// --------------------------------------------------------------------------- sparse (softmax)
// Softmax over the keys of the block's full layer: every key, or with SEL the selected rows
// idx[kv head][S]. Split over chunks like attn_full; the last threadgroup merges the chunks'
// running maxima and sums.

kernel void attn_sparse(
    device const bfloat* qraw    [[buffer(0)]],
    device const bfloat* g1      [[buffer(1)]],
    device const bfloat* g2      [[buffer(2)]],
    device const kv_t*   kc      [[buffer(3)]],
    device const kv_t*   vc      [[buffer(4)]],
    device float*        part    [[buffer(5)]],   // [C][16][HD + 2]
    device const uint*   idx     [[buffer(6)]],   // [2][S]
    device atomic_uint*  counter [[buffer(7)]],
    device bfloat*       o       [[buffer(8)]],
    constant AttnParams& p       [[buffer(9)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  t    [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int c = int(tg.x), kvh = int(tg.y), h = kvh * 8 + int(sg);
    float q[4];
    for (int i = 0; i < 4; i++) q[i] = float(qraw[h * HD + lane + 32 * i]);
    head_norm(q, g1, g2, p.ng, lane);
    for (int i = 0; i < 4; i++) q[i] = bf(q[i]);

    device const kv_t* kb = kc + size_t(kvh) * p.maxS * HD;
    device const kv_t* vb = vc + size_t(kvh) * p.maxS * HD;
#if KV8
    device const float* ks = kv_scales(kc, p.maxS) + kvh * p.maxS;
    device const float* vs = kv_scales(vc, p.maxS) + kvh * p.maxS;
#endif
    const int j0 = c * p.CH, j1 = min(p.S, j0 + p.CH);
    const float scale = rsqrt(float(HD));
    float m = NEG, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = j0; j < j1; j++) {
        const size_t row = SEL ? size_t(idx[kvh * p.S + j]) : size_t(j);
        float dot = 0.0f, vv[4];
        for (int i = 0; i < 4; i++) {
            const int d = int(lane) + 32 * i;
#if KV8
            dot += q[i] * float(kb[row * HD + d]);
            vv[i] = float(vb[row * HD + d]) * vs[row];
#else
            dot += q[i] * float(kb[row * HD + d]);
            vv[i] = float(vb[row * HD + d]);
#endif
        }
#if KV8
        const float s = simd_sum(dot) * ks[row] * scale;
#else
        const float s = simd_sum(dot) * scale;
#endif
        const float mn = max(m, s), cf = exp(m - mn), w = exp(s - mn);
        l = l * cf + w;
        for (int i = 0; i < 4; i++) acc[i] = acc[i] * cf + w * vv[i];
        m = mn;
    }
    device float* pp = part + (size_t(c) * 16 + h) * (HD + 2);
    for (int i = 0; i < 4; i++) pp[lane + 32 * i] = acc[i];
    if (lane == 0) { pp[HD] = m; pp[HD + 1] = l; }

    threadgroup bool last;
    if (!arrive_last(counter + kvh, uint(p.C), t, &last)) return;
    const int hh = kvh * 8 + int(t) / 32, d0 = (int(t) % 32) * 4;
    float M = NEG;
    for (int cc = 0; cc < p.C; cc++) M = max(M, part[(size_t(cc) * 16 + hh) * (HD + 2) + HD]);
    float L = 0.0f, num[4] = {0, 0, 0, 0};
    for (int cc = 0; cc < p.C; cc++) {
        device const float* q2 = part + (size_t(cc) * 16 + hh) * (HD + 2);
        const float f = exp(q2[HD] - M);
        L += q2[HD + 1] * f;
        for (int n = 0; n < 4; n++) num[n] += q2[d0 + n] * f;
    }
    for (int n = 0; n < 4; n++) o[hh * HD + d0 + n] = bfloat(num[n] / L);
}

// KV8: the same on simdgroup matrices (as attn_full8), in one pass with an online softmax: a lane
// holds one head's elements of S (keys fn, fn + 1) and of O (row fm), so it rescales its own O when
// its head's running maximum grows; the 4 lanes of a head (lane bits 0 and 3) share the maximum.
// With one chunk (C = 1) the threadgroup writes o itself.

kernel void attn_sparse8(
    device const bfloat* qraw    [[buffer(0)]],
    device const bfloat* g1      [[buffer(1)]],
    device const bfloat* g2      [[buffer(2)]],
    device const char*   kc      [[buffer(3)]],
    device const char*   vc      [[buffer(4)]],
    device float*        part    [[buffer(5)]],   // [C][16][HD + 2]
    device const uint*   idx     [[buffer(6)]],   // [2][S]
    device atomic_uint*  counter [[buffer(7)]],
    device bfloat*       o       [[buffer(8)]],
    constant AttnParams& p       [[buffer(9)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  t    [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    constexpr int NSG = 4;
    const int c = int(tg.x), kvh = int(tg.y);
    threadgroup float qs[8][HD];
    threadgroup float ot[NSG][8][HD];
    threadgroup float msg[NSG][8], lsg[NSG][8];
    for (int hh = int(sg); hh < 8; hh += NSG) {
        float q[4];
        for (int i = 0; i < 4; i++) q[i] = float(qraw[(kvh * 8 + hh) * HD + lane + 32 * i]);
        head_norm(q, g1, g2, p.ng, lane);
        for (int i = 0; i < 4; i++) qs[hh][qperm(int(lane) + 32 * i)] = bf(q[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Q[16], O[16];
    _Pragma("clang loop unroll(full)")
    for (int k = 0; k < 16; k++) {
        simdgroup_load(Q[k], &qs[0][8 * k], HD);
        O[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    const short2 fc = frag_coord(lane);
    const int fm = fc.x, fn = fc.y;
    device const char* kb = kc + size_t(kvh) * p.maxS * HD;
    device const char* vb = vc + size_t(kvh) * p.maxS * HD;
    device const float* ks = kv_scales(kc, p.maxS) + kvh * p.maxS;
    device const float* vs = kv_scales(vc, p.maxS) + kvh * p.maxS;
    device const uint* rows = idx + kvh * p.S;
    const int j0 = c * p.CH, j1 = min(p.S, j0 + p.CH);
    const float scale = rsqrt(float(HD));
    float m = NEG, l = 0.0f;
    for (int b = j0 + 8 * int(sg); b < j1; b += 8 * NSG) {
        const int ja = b + fn, jb = ja + 1;
        const int na = ja < j1 ? (SEL ? int(rows[ja]) : ja) : -1, nb = jb < j1 ? (SEL ? int(rows[jb]) : jb) : -1;
        const uint4 ra = na >= 0 ? *(device const uint4*)(kb + size_t(na) * HD + fm * 16) : uint4(0);
        const uint4 rb = nb >= 0 ? *(device const uint4*)(kb + size_t(nb) * HD + fm * 16) : uint4(0);
        const float ka = na >= 0 ? ks[na] * scale : 0.0f, kbs = nb >= 0 ? ks[nb] * scale : 0.0f;
        const float va = na >= 0 ? vs[na] : 0.0f, vbs = nb >= 0 ? vs[nb] : 0.0f;
        const int jv = b + fm;
        const int nv = jv < j1 ? (SEL ? int(rows[jv]) : jv) : -1;
        uint4 v0 = uint4(0), v1 = uint4(0);
        if (nv >= 0) {
            device const uint4* vr = (device const uint4*)(vb + size_t(nv) * HD + (fn / 2) * 32);
            v0 = vr[0];
            v1 = vr[1];
        }
        simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        _Pragma("clang loop unroll(full)")
        for (int k = 0; k < 16; k++) {
            simdgroup_float8x8 KT;
            KT.thread_elements()[0] = float(as_type<char4>(ra[k / 4])[k % 4]);
            KT.thread_elements()[1] = float(as_type<char4>(rb[k / 4])[k % 4]);
            simdgroup_multiply_accumulate(S, Q[k], KT, S);
        }
        const float s0 = na >= 0 ? S.thread_elements()[0] * ka : NEG;
        const float s1 = nb >= 0 ? S.thread_elements()[1] * kbs : NEG;
        float bm = max(s0, s1);
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        const float mn = max(m, bm), cf = exp(m - mn);
        const float w0 = na >= 0 ? exp(s0 - mn) : 0.0f, w1 = nb >= 0 ? exp(s1 - mn) : 0.0f;
        l = l * cf + w0 + w1;
        m = mn;
        simdgroup_float8x8 W;
        W.thread_elements()[0] = w0 * va;
        W.thread_elements()[1] = w1 * vbs;
        _Pragma("clang loop unroll(full)")
        for (int n = 0; n < 16; n++) {
            O[n].thread_elements()[0] *= cf;
            O[n].thread_elements()[1] *= cf;
            const uint wd = n < 8 ? v0[n / 2] : v1[(n - 8) / 2];
            const char4 b4 = as_type<char4>(wd);
            simdgroup_float8x8 V;
            V.thread_elements()[0] = float(b4[(n % 2) * 2]);
            V.thread_elements()[1] = float(b4[(n % 2) * 2 + 1]);
            simdgroup_multiply_accumulate(O[n], W, V, O[n]);
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

    // the simdgroups' (max, sum, O) merged: thread t takes head t / 16, dims (t % 16) · 8 ..
    const int th = int(t) / 16, d0 = (int(t) % 16) * 8, hh = kvh * 8 + th;
    float M = NEG;
    for (int g = 0; g < NSG; g++) M = max(M, msg[g][th]);
    float L = 0.0f, num[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < NSG; g++) {
        const float f = exp(msg[g][th] - M);
        L += lsg[g][th] * f;
        for (int e = 0; e < 8; e++) num[e] += ot[g][th][ocol(d0 + e)] * f;
    }
    if (p.C == 1) {
        for (int e = 0; e < 8; e++) o[hh * HD + d0 + e] = bfloat(num[e] / L);
        return;
    }
    device float* pp = part + (size_t(c) * 16 + hh) * (HD + 2);
    for (int e = 0; e < 8; e++) pp[d0 + e] = num[e];
    if (t % 16 == 0) {
        pp[HD] = M;
        pp[HD + 1] = L;
    }

    threadgroup bool last;
    if (!arrive_last(counter + kvh, uint(p.C), t, &last)) return;
    M = NEG;
    for (int cc = 0; cc < p.C; cc++) M = max(M, part[(size_t(cc) * 16 + hh) * (HD + 2) + HD]);
    L = 0.0f;
    for (int e = 0; e < 8; e++) num[e] = 0.0f;
    for (int cc = 0; cc < p.C; cc++) {
        device const float* q2 = part + (size_t(cc) * 16 + hh) * (HD + 2);
        const float f = exp(q2[HD] - M);
        L += q2[HD + 1] * f;
        for (int e = 0; e < 8; e++) num[e] += q2[d0 + e] * f;
    }
    for (int e = 0; e < 8; e++) o[hh * HD + d0 + e] = bfloat(num[e] / L);
}

// KV8, few keys (no selection, up to a few hundred): one threadgroup per query head as in the
// sliding layers, NS simdgroups taking every NS-th key (4 dims a lane, a simd_sum a key), merged
// through threadgroup memory; no chunks to merge across threadgroups.

#ifndef NS
#define NS 16
#endif
kernel void attn_sparse_short(
    device const bfloat* qraw    [[buffer(0)]],
    device const bfloat* g1      [[buffer(1)]],
    device const bfloat* g2      [[buffer(2)]],
    device const char*   kc      [[buffer(3)]],
    device const char*   vc      [[buffer(4)]],
    device bfloat*       o       [[buffer(8)]],
    constant AttnParams& p       [[buffer(9)]],
    uint h    [[threadgroup_position_in_grid]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int kvh = int(h) / 8;
    float q[4];
    for (int i = 0; i < 4; i++) q[i] = float(qraw[h * HD + lane + 32 * i]);
    head_norm(q, g1, g2, p.ng, lane);
    for (int i = 0; i < 4; i++) q[i] = bf(q[i]);
    device const char* kb = kc + size_t(kvh) * p.maxS * HD;
    device const char* vb = vc + size_t(kvh) * p.maxS * HD;
    device const float* ks = kv_scales(kc, p.maxS) + kvh * p.maxS;
    device const float* vs = kv_scales(vc, p.maxS) + kvh * p.maxS;
    const float scale = rsqrt(float(HD));
    float m = NEG, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = int(sg); j < p.S; j += NS) {
        float dot = 0.0f, vv[4];
        for (int i = 0; i < 4; i++) {
            dot += q[i] * float(kb[size_t(j) * HD + lane + 32 * i]);
            vv[i] = float(vb[size_t(j) * HD + lane + 32 * i]);
        }
        const float s = simd_sum(dot) * ks[j] * scale;
        const float mn = max(m, s), cf = exp(m - mn), w = exp(s - mn);
        l = l * cf + w;
        const float wv = w * vs[j];
        for (int i = 0; i < 4; i++) acc[i] = acc[i] * cf + wv * vv[i];
        m = mn;
    }
    threadgroup float tm[NS], tl[NS], ta[NS][HD];
    if (lane == 0) { tm[sg] = m; tl[sg] = l; }
    for (int i = 0; i < 4; i++) ta[sg][lane + 32 * i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float M = NEG;
        for (int g = 0; g < NS; g++) M = max(M, tm[g]);
        float L = 0.0f, f[NS];
        for (int g = 0; g < NS; g++) { f[g] = exp(tm[g] - M); L += tl[g] * f[g]; }
        for (int i = 0; i < 4; i++) {
            const int d = int(lane) + 32 * i;
            float r = 0.0f;
            for (int g = 0; g < NS; g++) r += ta[g][d] * f[g];
            o[h * HD + d] = bfloat(r / L);
        }
    }
}

// --------------------------------------------------------------------------- key selection
// The sparse layers' keys once there are more than TOPK + RECENT: per kv head, the TOPK older keys
// (all but the RECENT latest) with the highest score (max over the full layer's query heads of q·k,
// from attn_full with SCORES), then the RECENT latest, as row indices idx[kv head][TOPK + RECENT].
// One threadgroup of 1024 threads per kv head: a radix select on the scores' order-preserving bits
// finds the TOPK-th largest, then an ordered compaction (by key index, so the result does not
// depend on timing) takes every larger key and the first equal ones.

constant constexpr int TOPK = 1024, RECENT = 128, SELT = 1024;

inline uint order_bits(float f) {
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// Exclusive prefix sum over the threadgroup's SELT values, and the total.
inline uint scan_excl(uint v, threadgroup uint* buf, uint t, uint lane, uint sg, thread uint& total) {
    const uint incl = simd_prefix_inclusive_sum(v);
    if (lane == 31) buf[sg] = incl;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint s = buf[lane];
        const uint p = simd_prefix_inclusive_sum(s);
        buf[lane] = p - s;
        if (lane == 31) buf[32] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r = buf[sg] + incl - v;
    total = buf[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

kernel void select_keys(
    device const float* scores [[buffer(0)]],   // [2][maxS]
    device uint*        idx    [[buffer(1)]],   // [2][TOPK + RECENT]
    constant int2&      sm     [[buffer(2)]],   // S (keys), maxS
    uint kvh  [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup atomic_uint hist[256];
    threadgroup uint cum[256];
    threadgroup uint buf[33];
    threadgroup uint sel[2];   // digit, count above it
    const int S = sm.x, n = S - RECENT;
    device const float* sc = scores + size_t(kvh) * sm.y;
    device uint* out = idx + kvh * (TOPK + RECENT);

    uint prefix = 0, mask = 0, k = TOPK;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = int(t); j < n; j += SELT) {
            const uint b = order_bits(sc[j]);
            if ((b & mask) == prefix) atomic_fetch_add_explicit(&hist[(b >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // counts at or above each digit, from the top: the digit whose bucket holds the k-th largest
        if (sg == 0) {
            uint c[8], s = 0;
            for (int i = 0; i < 8; i++) { c[i] = atomic_load_explicit(&hist[255 - (lane * 8 + i)], memory_order_relaxed); s += c[i]; }
            uint above = simd_prefix_exclusive_sum(s);
            for (int i = 0; i < 8; i++) {
                const uint d = 255 - (lane * 8 + i);
                if (above < k && above + c[i] >= k) { sel[0] = d; sel[1] = above; }
                above += c[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= sel[0] << shift;
        mask |= 255u << shift;
        k -= sel[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // prefix is the TOPK-th largest; take every key above it and the first k equal to it
    const int per = (n + SELT - 1) / SELT, j0 = int(t) * per, j1 = min(n, j0 + per);
    uint gt = 0, eq = 0;
    for (int j = j0; j < j1; j++) {
        const uint b = order_bits(sc[j]);
        gt += b > prefix;
        eq += b == prefix;
    }
    uint ngt, neq;
    const uint gpos = scan_excl(gt, buf, t, lane, sg, ngt);
    uint epos = scan_excl(eq, buf, t, lane, sg, neq);
    uint g = gpos;
    for (int j = j0; j < j1; j++) {
        const uint b = order_bits(sc[j]);
        if (b > prefix) out[g++] = uint(j);
        else if (b == prefix) {
            if (epos < k) out[ngt + epos] = uint(j);
            epos++;
        }
    }
    for (int j = int(t); j < RECENT; j += SELT) out[TOPK + j] = uint(n + j);
}
