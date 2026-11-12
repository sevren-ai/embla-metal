// Token embedding, the engram gate, greedy sampling and the load-time layout kernels.

// The token's embedding row (4-bit), RMS-normed (no gains) into source 0.
kernel void embed(
    device const uchar*  W     [[buffer(0)]],
    constant ulong3&     off   [[buffer(1)]],   // words, scales, biases of transformer.wte
    device const uint*   token [[buffer(2)]],   // the token ids by position
    device bfloat*       state [[buffer(3)]],
    constant int&        pos   [[buffer(4)]],
    uint t    [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float buf[8];
    const size_t row = token[pos];
    const uint word = ((device const uint*)(W + off.x))[row * (D / 8) + t];
    const float sc = float(((device const bfloat*)(W + off.y))[row * (D / 64) + t / 8]);
    const float bs = float(((device const bfloat*)(W + off.z))[row * (D / 64) + t / 8]);
    float v[8], ss = 0.0f;
    for (int n = 0; n < 8; n++) {
        v[n] = bf(float((word >> (4 * n)) & 0xFu) * sc + bs);
        ss += v[n] * v[n];
    }
    const float r = precise::rsqrt(block_sum<256>(ss, buf, lane, sg) / D + EPS);
    for (int n = 0; n < 8; n++) state[t * 8 + n] = bfloat(v[n] * r);
}

// The engram read: gate g = σ(sign(z)·√|z|), z = <rms(W_k e), rms(he)>/√D, then
// g·W_v e + silu(conv(rms(g·W_v e))), a depthwise causal conv of 4 taps dilated by 3 over a ring
// of the normed values by position (HRING of them, so a prefill chunk fits). Written to source 1.
constant constexpr int HRING = 4096;
kernel void engram_gate(
    device const bfloat* kv    [[buffer(0)]],   // W_k e [D], W_v e [D]
    device const bfloat* he    [[buffer(1)]],
    device const bfloat* convw [[buffer(2)]],   // [D, 4]
    device bfloat*       hist  [[buffer(3)]],   // [HRING, D]
    device bfloat*       state [[buffer(4)]],
    constant int&        tpos  [[buffer(5)]],
    uint t    [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = 1024, E2 = D / TG;
    threadgroup float buf[32];
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
    float gated[E2], ss = 0.0f;
    for (int e = 0; e < E2; e++) {
        gated[e] = bf(g * val[e]);
        ss += gated[e] * gated[e];
    }
    const float r = precise::rsqrt(block_sum<TG>(ss, buf, lane, sg) / D + EPS32);
    for (int e = 0; e < E2; e++) {
        const int d = int(t) + e * TG;
        const float normed = bf(gated[e] * r);
        float conv = 0.0f;
        for (int i = 0; i < 3; i++) {
            const int tp = tpos - (3 - i) * 3;
            const float past = tp >= 0 ? float(hist[(tp % HRING) * D + d]) : 0.0f;
            conv = conv + past * float(convw[d * 4 + i]);
        }
        conv = conv + normed * float(convw[d * 4 + 3]);
        hist[(tpos % HRING) * D + d] = bfloat(normed);
        const float silu = bf(conv / (1.0f + precise::exp(-conv)));
        state[D + d] = bfloat(gated[e] + silu);
    }
}

// Greedy token over the logits as the reference samples them: bf16 logits, softcapped in bf16
// (tanh(l / 15) · 15), the first index of the maximum. Written to greedy[pos] and, past the
// prompt, as the next position's token.
kernel void argmax(
    device const float* logits [[buffer(0)]],
    device uint*        token  [[buffer(1)]],   // the token ids by position
    constant int&       V      [[buffer(2)]],
    device uint*        greedy [[buffer(3)]],
    constant int2&      pp     [[buffer(4)]],   // pos, prompt length
    uint t    [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float bv[32];
    threadgroup uint bi[32];
    float best = NEG;
    uint idx = 0;
    for (int i = int(t); i < V; i += 1024) {
        const float c = bf(bf(precise::tanh(bf(bf(logits[i]) / 15.0f))) * 15.0f);
        if (c > best) { best = c; idx = uint(i); }
    }
    for (int off = 16; off > 0; off /= 2) {
        const float ob = simd_shuffle_down(best, ushort(off));
        const uint oi = simd_shuffle_down(idx, ushort(off));
        if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
    }
    if (lane == 0) { bv[sg] = best; bi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        best = bv[lane];
        idx = bi[lane];
        for (int off = 16; off > 0; off /= 2) {
            const float ob = simd_shuffle_down(best, ushort(off));
            const uint oi = simd_shuffle_down(idx, ushort(off));
            if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
        }
        if (lane == 0) {
            greedy[pp.x] = idx;
            if (pp.x + 1 >= pp.y) token[pp.x + 1] = idx;
            // tell a CPU that polls: greedy[-1] (the word before) = (pos + 1) << 16 | token, the
            // token in the word itself: the CPU can see this word before it sees the stores above
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            atomic_store_explicit((device atomic_uint*)(greedy - 1), (uint(pp.x + 1) << 16) | idx, memory_order_relaxed);
        }
    }
}

// Load time: [E, R, C/8] words (8 nibbles of a row, low first) -> [E, C, R/8], out[e, c, w] packing
// W[e, 8w + n, c]. Grid (R/8, C, E).
kernel void nibble_transpose(
    device const uint* in   [[buffer(0)]],
    device uint*       out  [[buffer(1)]],
    constant uint2&    rc   [[buffer(2)]],   // R, C
    uint3 gid [[thread_position_in_grid]])
{
    const uint R = rc.x, C = rc.y, w = gid.x, c = gid.y, e = gid.z;
    uint v = 0;
    for (uint n = 0; n < 8; n++) {
        const uint word = in[(size_t(e) * R + 8 * w + n) * (C / 8) + c / 8];
        v |= ((word >> (4 * (c % 8))) & 0xFu) << (4 * n);
    }
    out[(size_t(e) * C + c) * (R / 8) + w] = v;
}

// Load time: [E, R, G] 16-bit -> [E, G, R]. Grid (R, G, E).
kernel void transpose16(
    device const ushort* in  [[buffer(0)]],
    device ushort*       out [[buffer(1)]],
    constant uint2&      rg  [[buffer(2)]],   // R, G
    uint3 gid [[thread_position_in_grid]])
{
    const uint R = rg.x, G = rg.y;
    out[(size_t(gid.z) * G + gid.y) * R + gid.x] = in[(size_t(gid.z) * R + gid.x) * G + gid.y];
}

// Debug: copy n bf16 values.
kernel void copy16(device const ushort* in [[buffer(0)]], device ushort* out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    out[i] = in[i];
}


// Sampling instead of the argmax (same outputs): softmax(logits / T) truncated to its top-p
// nucleus, by the Gumbel-max trick, so the token stays on the GPU and steps stay pipelined.
// The nucleus is every token with e = exp((c - max) / T) at or above a threshold found by
// bisection (the largest that keeps at least top-p of the mass); the draw is the argmax of
// (c - max) / T + Gumbel noise over it, the noise a hash of (seed, position, token).
struct SampleParams { float invT; float topP; uint seed; };

inline float sample_max(float v, threadgroup float* buf, uint lane, uint sg) {
    v = simd_max(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = simd_max(buf[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

inline float sample_sum(float v, threadgroup float* buf, uint lane, uint sg) {
    v = simd_sum(v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = simd_sum(buf[lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}

inline uint sample_hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

kernel void sample(
    device const float*    logits [[buffer(0)]],
    device uint*           token  [[buffer(1)]],
    constant int&          V      [[buffer(2)]],
    device uint*           greedy [[buffer(3)]],
    constant int2&         pp     [[buffer(4)]],   // pos, prompt length
    constant SampleParams& sp     [[buffer(5)]],
    device float*          ex     [[buffer(6)]],   // V floats of scratch: (c - max) / T
    uint t    [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float buf[32];
    threadgroup uint bi[32];
    float mx = NEG;
    for (int i = int(t); i < V; i += 1024) {
        const float c = bf(bf(precise::tanh(bf(bf(logits[i]) / 15.0f))) * 15.0f) * sp.invT;
        ex[i] = c;
        mx = max(mx, c);
    }
    mx = sample_max(mx, buf, lane, sg);
    float z = 0.0f;
    for (int i = int(t); i < V; i += 1024) {
        const float x = ex[i] - mx;
        ex[i] = x;
        z += exp(x);
    }
    // the nucleus' threshold on exp(x), in [lo, hi): lo always keeps at least top-p of the mass
    // The bisection runs over the tokens with e >= CMIN, gathered into threadgroup memory, when
    // they hold the nucleus (then the threshold is at least CMIN); else over all of them.
    float lo = 0.0f;
    if (sp.topP < 1.0f) {
        constexpr float CMIN = 1.0f / 16384.0f;
        threadgroup float cand[2048];
        threadgroup atomic_uint ncand;
        if (t == 0) atomic_store_explicit(&ncand, 0u, memory_order_relaxed);
        const float want = sp.topP * sample_sum(z, buf, lane, sg);
        float cm = 0.0f;
        for (int i = int(t); i < V; i += 1024) {
            const float e = exp(ex[i]);
            if (e >= CMIN) {
                const uint k = atomic_fetch_add_explicit(&ncand, 1u, memory_order_relaxed);
                if (k < 2048) cand[k] = e;
                cm += e;
            }
        }
        cm = sample_sum(cm, buf, lane, sg);   // its barrier also publishes cand and ncand
        const uint nc = atomic_load_explicit(&ncand, memory_order_relaxed);
        float hi = 1.0f;
        if (nc <= 2048 && cm >= want) {
            lo = CMIN;
            for (int it = 0; it < 20; it++) {
                const float mid = 0.5f * (lo + hi);
                float kept = 0.0f;
                for (uint i = t; i < nc; i += 1024) kept += cand[i] >= mid ? cand[i] : 0.0f;
                if (sample_sum(kept, buf, lane, sg) >= want) lo = mid; else hi = mid;
            }
        } else {
            for (int it = 0; it < 20; it++) {
                const float mid = 0.5f * (lo + hi);
                float kept = 0.0f;
                for (int i = int(t); i < V; i += 1024) {
                    const float e = exp(ex[i]);
                    kept += e >= mid ? e : 0.0f;
                }
                if (sample_sum(kept, buf, lane, sg) >= want) lo = mid; else hi = mid;
            }
        }
    }
    const float floorX = lo > 0.0f ? log(lo) : NEG;
    float best = NEG;
    uint idx = 0;
    const uint key = sample_hash(sp.seed ^ sample_hash(uint(pp.x) * 0x9e3779b9u));
    for (int i = int(t); i < V; i += 1024) {
        const float x = ex[i];
        if (x < floorX) continue;
        const float u = (float(sample_hash(key ^ uint(i)) >> 8) + 0.5f) * (1.0f / 16777216.0f);
        const float s = x - log(-log(u));
        if (s > best) { best = s; idx = uint(i); }
    }
    for (int off = 16; off > 0; off /= 2) {
        const float ob = simd_shuffle_down(best, ushort(off));
        const uint oi = simd_shuffle_down(idx, ushort(off));
        if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
    }
    if (lane == 0) { buf[sg] = best; bi[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        best = buf[lane];
        idx = bi[lane];
        for (int off = 16; off > 0; off /= 2) {
            const float ob = simd_shuffle_down(best, ushort(off));
            const uint oi = simd_shuffle_down(idx, ushort(off));
            if (ob > best || (ob == best && oi < idx)) { best = ob; idx = oi; }
        }
        if (lane == 0) {
            greedy[pp.x] = idx;
            if (pp.x + 1 >= pp.y) token[pp.x + 1] = idx;
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            atomic_store_explicit((device atomic_uint*)(greedy - 1), (uint(pp.x + 1) << 16) | idx, memory_order_relaxed);
        }
    }
}
