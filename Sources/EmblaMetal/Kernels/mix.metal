// Block Attention Residuals: the K mixes of one sub-layer input.
// One threadgroup of 1024 threads; each source is read once into registers.
//
// N sources in `state` (bf16), K queries from column `col` of `queries` (f32), HAS_X: merge with
// the current block's partial sum x (f32), NORM: RMS-norm the mixes, APPEND: first write bf16(x)
// to source N - 1 (a block was completed) and read it from there (then HAS_X is 0).

kernel void mix(
    device bfloat*       state   [[buffer(0)]],   // [tokens][9][D]
    device const float*  queries [[buffer(1)]],
    device const float*  x       [[buffer(2)]],   // [tokens][D]
    device bfloat*       out     [[buffer(3)]],   // [tokens][3][D]
    constant int&        col     [[buffer(4)]],
    uint tok  [[threadgroup_position_in_grid]],   // prefill: one threadgroup a token
    uint t    [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = 1024, E = D / TG, SG = TG / 32;
    constexpr int NP = (N + HAS_X) * (K + 1);
    state += tok * 9 * D;
    x += tok * D;
    out += tok * 3 * D;
    threadgroup float buf[SG][NP];
    threadgroup float tot[NP];

    float v[N][E], xv[E], acc[NP];
    for (int p = 0; p < NP; p++) acc[p] = 0.0f;
    for (int e = 0; e < E; e++) {
        const int d = int(t) + e * TG;
        float q[K];
        for (int j = 0; j < K; j++) q[j] = queries[(col + j) * D + d];
        for (int n = 0; n < N; n++) {
            if (APPEND && n == N - 1) {
                v[n][e] = bf(x[d]);
                state[n * D + d] = bfloat(v[n][e]);
            } else {
                v[n][e] = float(state[n * D + d]);
            }
            acc[n * (K + 1)] += v[n][e] * v[n][e];
            for (int j = 0; j < K; j++) acc[n * (K + 1) + 1 + j] += q[j] * v[n][e];
        }
        if (HAS_X) {
            xv[e] = x[d];
            acc[N * (K + 1)] += xv[e] * xv[e];
            for (int j = 0; j < K; j++) acc[N * (K + 1) + 1 + j] += q[j] * xv[e];
        }
    }
    for (int p = 0; p < NP; p++) {
        const float s = simd_sum(acc[p]);
        if (lane == 0) buf[sg][p] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = int(t); p < NP; p += TG) {
        float s = 0.0f;
        for (int i = 0; i < SG; i++) s += buf[i][p];
        tot[p] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float o[K][E], sum_sq[K];
    for (int j = 0; j < K; j++) {
        float lg[N], mx = NEG;
        for (int n = 0; n < N; n++) {
            const float rstd = precise::rsqrt(tot[n * (K + 1)] / D + EPS);
            lg[n] = tot[n * (K + 1) + 1 + j] * rstd;
            mx = max(mx, lg[n]);
        }
        float se = 0.0f;
        for (int n = 0; n < N; n++) se += precise::exp(lg[n] - mx);
        const float lse = mx + precise::log(se);
        float w = 0.0f;
        if (HAS_X) {
            const float rstd = precise::rsqrt(tot[N * (K + 1)] / D + EPS);
            const float lx = tot[N * (K + 1) + 1 + j] * rstd;
            w = 1.0f / (1.0f + precise::exp(lse - lx));
        }
        // The source weights and their mix are rounded to bf16, as in the reference.
        float pw[N];
        for (int n = 0; n < N; n++) pw[n] = bf(precise::exp(lg[n] - lse));
        sum_sq[j] = 0.0f;
        for (int e = 0; e < E; e++) {
            float m = 0.0f;
            for (int n = 0; n < N; n++) m += pw[n] * v[n][e];
            m = bf(m);
            if (HAS_X) m += w * (xv[e] - m);
            o[j][e] = m;
            sum_sq[j] += m * m;
        }
    }

    float scale[K];
    for (int j = 0; j < K; j++) scale[j] = 1.0f;
    if (NORM) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < K; j++) {
            const float s = simd_sum(sum_sq[j]);
            if (lane == 0) buf[sg][j] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < K; j++) {
            float s = 0.0f;
            for (int i = 0; i < SG; i++) s += buf[i][j];
            scale[j] = precise::rsqrt(s / D + EPS);
        }
    }
    for (int j = 0; j < K; j++) {
        for (int e = 0; e < E; e++) out[j * D + int(t) + e * TG] = bfloat(o[j][e] * scale[j]);
    }
}

// ---------------------------------------------------------------------------------------------
// Within a block the sources do not change, so neither do a query's softmax weights over them nor
// their mix M0 = bf16(Σ_n p_n V_n): only the merge with the partial sum x changes from one
// sub-layer to the next. `prep` computes, when the sources change, every coming query's M0 and its
// statistics; a consumer then needs only q·x, |x|² and M0·x (mix_input below).

#ifndef KOUT
#define KOUT 0
#endif

// One threadgroup (256 threads) per query col0 + tg: M0 [col] (bf16) and stats [col] = (lse,
// |M0|², 0, 0). With APPEND, bf16(x) is first source N - 1 (threadgroup 0 writes it to state).
kernel void prep(
    device bfloat*       state   [[buffer(0)]],
    device const float*  queries [[buffer(1)]],
    device const float*  x       [[buffer(2)]],
    device bfloat*       m0      [[buffer(3)]],
    device float4*       stats   [[buffer(4)]],
    constant int&        col0    [[buffer(5)]],
    device bfloat*       mixout  [[buffer(6)]],   // KOUT: the first KOUT columns' mixes, [KOUT][D]
    uint tg   [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    constexpr int TG = 256, E = D / TG, SG = TG / 32;
    threadgroup float buf[SG][2 * N];
    float mv[E];
    threadgroup float tot[2 * N];
    const int col = col0 + int(tg);
    float v[N][E], acc[2 * N];
    for (int p = 0; p < 2 * N; p++) acc[p] = 0.0f;
    for (int e = 0; e < E; e++) {
        const int d = int(t) + e * TG;
        const float q = queries[col * D + d];
        for (int n = 0; n < N; n++) {
            if (APPEND && n == N - 1) {
                v[n][e] = bf(x[d]);
                if (tg == 0) state[n * D + d] = bfloat(v[n][e]);
            } else {
                v[n][e] = float(state[n * D + d]);
            }
            acc[2 * n] += v[n][e] * v[n][e];
            acc[2 * n + 1] += q * v[n][e];
        }
    }
    for (int p = 0; p < 2 * N; p++) {
        const float s = simd_sum(acc[p]);
        if (lane == 0) buf[sg][p] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) < 2 * N) {
        float s = 0.0f;
        for (int i = 0; i < SG; i++) s += buf[i][t];
        tot[t] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lg[N], mx = NEG;
    for (int n = 0; n < N; n++) {
        lg[n] = tot[2 * n + 1] * precise::rsqrt(tot[2 * n] / D + EPS);
        mx = max(mx, lg[n]);
    }
    float se = 0.0f;
    for (int n = 0; n < N; n++) se += precise::exp(lg[n] - mx);
    const float lse = mx + precise::log(se);
    float pw[N];
    for (int n = 0; n < N; n++) pw[n] = bf(precise::exp(lg[n] - lse));
    float ss = 0.0f;
    for (int e = 0; e < E; e++) {
        float m = 0.0f;
        for (int n = 0; n < N; n++) m += pw[n] * v[n][e];
        m = bf(m);
        m0[col * D + int(t) + e * TG] = bfloat(m);
        mv[e] = m;
        ss += m * m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = simd_sum(ss);
    if (lane == 0) buf[sg][0] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float total;
    if (t == 0) {
        float s = 0.0f;
        for (int i = 0; i < SG; i++) s += buf[i][0];
        stats[col] = float4(lse, s, 0.0f, 0.0f);
        total = s;
    }
#if KOUT
    // a block's first sub-layer input has no partial sum yet: its mix is M0, normed (as mix_input
    // without MERGE), written once here instead of in every threadgroup of the GEMV that reads it
    if (int(tg) < KOUT) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float scale = precise::rsqrt(total / D + EPS);
        for (int e = 0; e < E; e++) mixout[int(tg) * D + int(t) + e * TG] = bfloat(mv[e] * scale);
    }
#endif
}
