// The MoE of one token: moe_router (logits and routing), moe_fused (every expert's up projection and
// sparse down projection, one launch), moe_reduce (the partial outputs into x). The input hm is
// the MLP input mix (mix.metal).

#ifndef UNR
#define UNR 8
#endif
#ifndef MR
#define MR 1
#endif

constant constexpr int E = 96, TOPK = 8, F = 512, FS = 1024, KW = D / 8;
#ifndef RSG
#define RSG 8     // router rows (simdgroups) a threadgroup
#endif
constant constexpr int ROUTER_TGS = E / RSG;

struct MoEWeights {
    ulong router;             // [E, D] bf16, in W
    ulong fc_w, fc_s, fc_b;   // routed up, in W: [E, F, D/8] u32, [E, F, D/64] bf16
    ulong sh_w, sh_s, sh_b;   // shared up: [FS, D/8], [FS, D/64]
    ulong bias;               // expert_bias [E] bf16, in W
    ulong pw, ps, pb;         // routed down, neuron-major, in NM: [E, F, D/8], [E, F/64, D]
    ulong spw, sps, spb;      // shared down, neuron-major: [FS, D/8], [FS/64, D]
};

// The router logits (bf16 weights, f32 sums), a simdgroup a row; the last threadgroup routes: top-8
// of sigmoid(logit) + bias, the chosen sigmoids renormalized as gates (rounded to bf16).
#ifndef HAS_X
#define HAS_X 1
#endif

// The router's input is the MLP input mix, which every threadgroup computes (mix_input) and
// threadgroup 0 writes to hm for moe_fused.
kernel void moe_router(
    device const uchar*  W       [[buffer(0)]],
    constant MoEWeights& mw      [[buffer(1)]],
    device const float*  queries [[buffer(2)]],
    device const bfloat* m0      [[buffer(3)]],
    device const float4* stats   [[buffer(4)]],
    device const float*  xres    [[buffer(5)]],
    device bfloat*       hm      [[buffer(6)]],
    device float*        logits  [[buffer(7)]],
    device atomic_uint*  counter [[buffer(8)]],
    device uint*         route   [[buffer(9)]],   // [TOPK] experts, then [TOPK] gates (f32 bits)
    constant int&        col     [[buffer(10)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int row = int(tg) * RSG + int(sg);
    bfloat4 rw[16];
    device const bfloat4* wp = (device const bfloat4*)(W + mw.router) + row * (D / 4);
    for (int i = 0; i < 16; i++) rw[i] = wp[i * 32 + lane];
    threadgroup bfloat xm[D];
    threadgroup float red[RSG * 3 + 2];
    mix_input<1, HAS_X, RSG * 32>(queries, m0, stats, col, xres, xm, hm, tg == 0, red, t, lane, sg);
    float acc = 0.0f;
    threadgroup const bfloat4* x4 = (threadgroup const bfloat4*)xm;
    for (int i = 0; i < 16; i++) acc += dot(float4(rw[i]), float4(x4[i * 32 + lane]));
    acc = simd_sum(acc);
    if (lane == 0) logits[row] = acc;

    threadgroup bool last;
    if (!arrive_last(counter, ROUTER_TGS, t, &last) || sg != 0) return;
    constexpr int RR = E / 32;
    device const bfloat* bias = (device const bfloat*)(W + mw.bias);
    float score[RR], biased[RR];
    for (int i = 0; i < RR; i++) {
        const int e = int(lane) + 32 * i;
        const float l = logits[e];
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
        if (lane == 0) route[k] = uint(winner);
    }
    for (int k = 0; k < TOPK; k++) total = wk[k] + total;
    if (lane == 0) {
        for (int k = 0; k < TOPK; k++) route[TOPK + k] = as_type<uint>(bf(wk[k] / total));
    }
}

// The routed and shared experts in one launch (a fused MoE): a threadgroup takes a slice
// of SL neurons of one expert slot (the 8 routed experts, then the shared expert), computes their
// ReLU² activations (up projection, MLX layout), compacts the live ones, and multiplies only those
// into its gated partial output through the neuron-major down projection (one load a thread a live
// neuron, all in flight together). moe_reduce adds the NS partials into x.
constant constexpr int SL = 64, NSLICE = (TOPK * F + FS) / SL, RSLICE = TOPK * F / SL;

// PART 0: every slice (a threadgroup each). PART 1: the routed slices only. PART 2: the shared
// expert's slices only, which need no routing: they run beside the router, each computing the MLP
// input mix itself (as the router does) instead of reading the router's copy.
#ifndef PART
#define PART 0
#endif

kernel void moe_fused(
    device const uchar*  W       [[buffer(0)]],
    device const uchar*  NM      [[buffer(1)]],
    device const bfloat* x       [[buffer(2)]],   // the MLP input mix (moe_router writes it)
    device const uint*   route   [[buffer(3)]],
    device float*        part    [[buffer(4)]],   // [NSLICE, D]
    constant MoEWeights& mw      [[buffer(5)]],
    device const float*  queries [[buffer(6)]],
    device const bfloat* m0      [[buffer(7)]],
    device const float4* stats   [[buffer(8)]],
    device const float*  xres    [[buffer(9)]],
    constant int&        col     [[buffer(10)]],
    uint tgi  [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgi  [[simdgroup_index_in_threadgroup]])
{
#if PART == 2
    const uint s = tgi + RSLICE;
    threadgroup bfloat xm[D];
    threadgroup float red[8 * 3 + 2];
    mix_input<1, true, 256>(queries, m0, stats, col, xres, xm, (device bfloat*)nullptr, false, red, tid, lane, sgi);
    threadgroup const bfloat* xin = xm;
#else
    const uint s = tgi;
    device const bfloat* xin = x;
#endif
    const bool routed = int(s) < RSLICE;
    threadgroup float hs[SL];
    threadgroup uchar live[SL];
    threadgroup int nlive;

    device const uint* uw; device const bfloat* us; device const bfloat* ub;
    device const uint* dw; device const bfloat* ds; device const bfloat* db;
    float gain;
    if (routed) {
        const int ei = int(s) / (F / SL), r0 = (int(s) % (F / SL)) * SL;
        const size_t e = route[ei], row = e * F + r0;
        uw = (device const uint*)(W + mw.fc_w) + row * KW;
        us = (device const bfloat*)(W + mw.fc_s) + row * (D / 64);
        ub = (device const bfloat*)(W + mw.fc_b) + row * (D / 64);
        dw = (device const uint*)(NM + mw.pw) + row * KW;
        ds = (device const bfloat*)(NM + mw.ps) + (e * (F / 64) + r0 / 64) * D;
        db = (device const bfloat*)(NM + mw.pb) + (e * (F / 64) + r0 / 64) * D;
        gain = as_type<float>(route[TOPK + ei]);
    } else {
        const int r0 = (int(s) - RSLICE) * SL;
        uw = (device const uint*)(W + mw.sh_w) + size_t(r0) * KW;
        us = (device const bfloat*)(W + mw.sh_s) + size_t(r0) * (D / 64);
        ub = (device const bfloat*)(W + mw.sh_b) + size_t(r0) * (D / 64);
        dw = (device const uint*)(NM + mw.spw) + size_t(r0) * KW;
        ds = (device const bfloat*)(NM + mw.sps) + size_t(r0 / 64) * D;
        db = (device const bfloat*)(NM + mw.spb) + size_t(r0 / 64) * D;
        gain = 1.0f;
    }

    // ---- up: a simdgroup's 8 rows in rounds of MR (GemvRows), ReLU² as bf16
    for (int r0 = int(sgi) * MR; r0 < SL; r0 += 8 * MR) {
        GemvRows<MR> rows;
        rows.load((device const uint4*)uw, us, ub, r0, lane);
        float acc[MR];
#if PART == 2
        rows.dot((threadgroup const bfloat4*)xin, lane, acc);
#else
        rows.dot((device const bfloat4*)xin, lane, acc);
#endif
        if (lane == 0) {
            for (int u = 0; u < MR; u++) {
                const float v = max(bf(acc[u]), 0.0f);
                hs[r0 + u] = bf(v * v);
            }
        }
    }
    const float4 s0 = float4(*(device const bfloat4*)(ds + 8 * tid)), s1 = float4(*(device const bfloat4*)(ds + 8 * tid + 4));
    const float4 c0 = float4(*(device const bfloat4*)(db + 8 * tid)), c1 = float4(*(device const bfloat4*)(db + 8 * tid + 4));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- the live neurons, compacted by simdgroup 0
    if (sgi == 0) {
        const bool a = hs[lane] != 0.0f, c = hs[32 + lane] != 0.0f;
        const uint m0 = uint((simd_vote::vote_t)simd_ballot(a)), m1 = uint((simd_vote::vote_t)simd_ballot(c));
        const uint below = (1u << lane) - 1u;
        if (a) live[popcount(m0 & below)] = uchar(lane);
        if (c) live[popcount(m0) + popcount(m1 & below)] = uchar(32 + lane);
        if (lane == 0) nlive = popcount(m0) + popcount(m1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- sparse down: thread tid owns word tid (outputs 8·tid ..), UNR live neurons in flight
    const int n = nlive;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}, hsum = 0.0f;
    for (int j0 = 0; j0 < n; j0 += UNR) {
        uint qw[UNR];
        float hv[UNR];
        _Pragma("clang loop unroll(full)")
        for (int u = 0; u < UNR; u++) {
            const bool v = j0 + u < n;
            const int i = v ? int(live[j0 + u]) : 0;
            qw[u] = v ? dw[size_t(i) * KW + tid] : 0u;
            hv[u] = v ? hs[i] : 0.0f;
        }
        _Pragma("clang loop unroll(full)")
        for (int u = 0; u < UNR; u++) {
            hsum += hv[u];
            for (int k = 0; k < 8; k++) acc[k] += hv[u] * float((qw[u] >> (4 * k)) & 0xFu);
        }
    }
    const float sc[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    device float4* op = (device float4*)(part + size_t(s) * D + 8 * tid);
    float o[8];
    for (int k = 0; k < 8; k++) o[k] = n == 0 ? 0.0f : gain * (sc[k] * acc[k] + bs[k] * hsum);
    op[0] = float4(o[0], o[1], o[2], o[3]);
    op[1] = float4(o[4], o[5], o[6], o[7]);
}

#ifndef RG
#define RG 1   // groups of slices a column's sum is split over (threads 256 / RG columns a threadgroup)
#endif
#ifndef MIXK
#define MIXK 0   // > 0: also the next layer's MIXK attention-input mixes (mix_input, MERGE), into hs
#endif
kernel void moe_reduce(
    device const float*  part    [[buffer(0)]],
    device float*        x       [[buffer(1)]],
    device const float*  queries [[buffer(2)]],
    device const bfloat* m0      [[buffer(3)]],
    device const float4* stats   [[buffer(4)]],
    constant int&        col     [[buffer(5)]],
    device atomic_uint*  counter [[buffer(6)]],
    device float*        dots    [[buffer(7)]],   // [threadgroups][2 MIXK + 1]
    device bfloat*       hs      [[buffer(8)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    constexpr int C = 256 / RG, PER = NSLICE / RG;
    const uint g = t / C, d = tg * C + t % C;
    float xn = 0.0f;
#if RG == 1
    float s = 0.0f;
    for (int i = 0; i < NSLICE; i++) s += part[i * D + d];
    xn = x[d] + bf(s);
    x[d] = xn;
#else
    float s = 0.0f;
    for (int i = int(g) * PER; i < int(g + 1) * PER; i++) s += part[i * D + d];
    threadgroup float sums[RG][C];
    sums[g][t % C] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (g == 0) {
        float r = sums[0][t];
        for (int k = 1; k < RG; k++) r += sums[k][t];
        xn = x[d] + bf(r);
        x[d] = xn;
    }
#endif
#if MIXK
    // this threadgroup's share of q·x, M0·x (each mix) and |x|²
    constexpr int NP = 2 * MIXK + 1, NTG = D * RG / 256;
    float acc[NP];
    for (int p = 0; p < NP; p++) acc[p] = 0.0f;
    if (g == 0) {
        acc[2 * MIXK] = xn * xn;
        for (int j = 0; j < MIXK; j++) {
            acc[2 * j] = queries[(col + j) * D + d] * xn;
            acc[2 * j + 1] = float(m0[(col + j) * D + d]) * xn;
        }
    }
    threadgroup float red[8][NP];
    for (int p = 0; p < NP; p++) {
        const float v = simd_sum(acc[p]);
        if (lane == 0) red[sg][p] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) < NP) {
        float v = 0.0f;
        for (int i = 0; i < 8; i++) v += red[i][t];
        dots[tg * NP + t] = v;
    }
    // the last threadgroup to finish: the mixes, from every threadgroup's sums and the new x
    threadgroup bool last;
    if (!arrive_last(counter, uint(NTG), t, &last)) return;
    float tot[NP];
    for (int p = 0; p < NP; p++) {
        float v = 0.0f;
        for (int i = 0; i < NTG; i++) v += dots[i * NP + p];
        tot[p] = v;
    }
    const float xx = tot[2 * MIXK], rx = precise::rsqrt(xx / D + EPS);
    float w[MIXK], scale[MIXK];
    for (int j = 0; j < MIXK; j++) {
        const float4 st = stats[col + j];
        w[j] = 1.0f / (1.0f + precise::exp(st.x - tot[2 * j] * rx));
        const float u = 1.0f - w[j];
        const float ss = u * u * st.y + 2.0f * w[j] * u * tot[2 * j + 1] + w[j] * w[j] * xx;
        scale[j] = precise::rsqrt(ss / D + EPS);
    }
    for (int e = int(t); e < D; e += 256) {
        const float xv = x[e];
        for (int j = 0; j < MIXK; j++) {
            float m = float(m0[(col + j) * D + e]);
            m += w[j] * (xv - m);
            hs[j * D + e] = bfloat(m * scale[j]);
        }
    }
#endif
}
