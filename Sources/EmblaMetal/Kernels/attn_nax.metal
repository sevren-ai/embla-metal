// ReLU² prefill attention on the M5's neural accelerators (layer 7 of the prefill half): the body of
// an earlier ReLU² prefill kernel (relu2_prefill_nax), unchanged but for its output, which goes token-major into o [Tq][2048]. q [16][Tq][128] normed
// with its gains (q_prep), the keys normed in the cache: p = relu(q·k/√D)², out = Σ p·v / (√D + Σ p).
// KV8: the cache is int8 (common.metal): a key block's rows are converted once a threadgroup into
// threadgroup memory as their integers (exact in bf16), and the tiles load from there; each key's
// scale multiplies its column of S, and each value row's scale its column of P before P·V.

#ifndef T
#define T bfloat
#endif
constant constexpr int BQ = 64, BK = 32, WM = 4, PHALF = 0, WIN = 0, DSPLIT = 1, OUTLD = 2048;
#define D 128   // in this file D is the head dim (common.metal's D, the model width, is OUTLD here)

// Measurement only (EMBLA_ATT8): QK8 multiplies Q·Kᵀ in int8 (q quantized per row by q_prep_b,
// scales after its rows), PV8 multiplies P·V in int8 (P quantized per query row per key block);
// both int8 x int8 -> int32 on the accelerators, then scaled. Need KV8.
#ifndef QK8
#define QK8 0
#endif
#ifndef PV8
#define PV8 0
#endif
struct MaxOp { template <typename U> METAL_FUNC static constexpr U apply(U x, U y) { return metal::max(x, y); } };
constant constexpr int SLD8 = 128 + 16;
/// Rows k0 .. k0 + 31 of an int8 cache copied into s [32][SLD8] (rows from Tk on as zeros).
METAL_FUNC void stage_raw8(threadgroup int8_t* s, device const char* base, int k0, int Tk, uint tid) {
    const int r = int(tid) / 4, c0 = (int(tid) % 4) * 32;
    threadgroup uint4* out = (threadgroup uint4*)(s + r * SLD8 + c0);
    device const uint4* src = (device const uint4*)(base + int64_t(k0 + r) * 128 + c0);
    out[0] = k0 + r < Tk ? src[0] : uint4(0);
    out[1] = k0 + r < Tk ? src[1] : uint4(0);
}
#if KV8
typedef char KV;
/// Rows k0 .. k0 + 31 of an int8 cache as bf16 integers in s [32][SLD] (rows from Tk on as zeros),
/// by the threadgroup's 128 threads, 32 values each: once for its 4 simdgroups.
constant constexpr int SLD = 128 + 8;
METAL_FUNC void stage_rows8(threadgroup T* s, device const char* base, int k0, int Tk, uint tid) {
    const int r = int(tid) / 4, c0 = (int(tid) % 4) * 32;
    threadgroup T* out = s + r * SLD + c0;
    uint4 w[2] = {uint4(0), uint4(0)};
    if (k0 + r < Tk) {
        device const uint4* src = (device const uint4*)(base + int64_t(k0 + r) * 128 + c0);
        w[0] = src[0];
        w[1] = src[1];
    }
    STEEL_PRAGMA_UNROLL
    for (int h = 0; h < 2; h++) {
        STEEL_PRAGMA_UNROLL
        for (int e = 0; e < 4; e++) {
            const char4 c = as_type<char4>(w[h][e]);
            STEEL_PRAGMA_UNROLL
            for (int b = 0; b < 4; b++) out[16 * h + 4 * e + b] = T(float(c[b]));
        }
    }
}
#else
typedef T KV;
#endif

kernel void relu2_prefill_nax(
#if QK8
    device const char*    q           [[buffer(0)]],   // int8 rows [H][Tq][D], then their scales
#else
    device const T*       q           [[buffer(0)]],
#endif
    device const KV*      k           [[buffer(1)]],
    device const KV*      v           [[buffer(2)]],
    constant int64_t*     k_strides   [[buffer(3)]],
    constant int64_t*     v_strides   [[buffer(4)]],
    constant int&         Tq_         [[buffer(5)]],
    constant int&         Tk_         [[buffer(6)]],
    constant int&         n_heads     [[buffer(7)]],
    constant int&         gqa_factor  [[buffer(8)]],
    constant float&       scale       [[buffer(9)]],
    constant float&       sink_val    [[buffer(10)]],
    device const float*   threshold   [[buffer(11)]],
    device T*             out         [[buffer(12)]],
    uint3  threadgroup_position_in_grid  [[threadgroup_position_in_grid]],
    uint   simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint   tid                            [[thread_index_in_threadgroup]])
{
    // (added to relu2_prefill_nax: WIN, a sliding window of WIN keys ending at the
    //  query's own position (WIN = 0: full causal). Key blocks entirely before a threadgroup's window are not visited.
    //  And DSPLIT: DSPLIT threadgroups per query block, each owning D / DSPLIT output dims (S = Q·Kᵀ is recomputed by
    //  each over the full D): at D = 256 the whole 16×256 fp32 O tile per simdgroup spills, DSPLIT = 2 keeps 16×128.)
    // ReLU² prefill on the neural accelerators (Metal 4 tensor ops through MLX's steel NAX tiles, header_nax): the loop of
    // MLX's steel_attention_nax with the online softmax replaced by ReLU² — p = max(s·scale − τ, 0)², a running row sum,
    // out = Σ p·v / (sink + Σ p). Q/K pre-normed (RMS applied at cache write), fp16 q/K/V.
    // One threadgroup per (batch, q head, BQ queries), WM simdgroups of 16 query rows; key blocks of BK. Each simdgroup
    // skips P·V for a key block where its whole 16×BK tile of p is zero (exact: those products are 0).
    // The two HSP_ marker lines below are where SparsePrefill's variant inserts its 128-key block mask (a simdgroup's
    // 16 rows lie in one QT-row selection tile, so it skips a dropped block's K loads, QKᵀ and P·V outright).
    //   q [B,H,Tq,D] contiguous, k/v [B,Hkv,≥Tk,D] (strided cache views) -> out [B,H,Tq,D] T, Lout [B,H,Tq] T (= sink + Σ p)
    constexpr short UQ = 16;                  // query rows per simdgroup
    constexpr short UD = 32;                  // output columns per O subtile
    constexpr short UKs = 32;                 // keys per S subtile
    constexpr short UDs = 16;                 // reduction step of S = Q·Kᵀ
    constexpr short UK = 16;                  // keys per P subtile (the reduction step of O = P·V)
    static_assert(BQ == WM * UQ, "one 16-row query tile per simdgroup");
    static_assert(BK % UKs == 0 && D % (UD * DSPLIT) == 0 && D % UDs == 0, "tile shapes");
    constexpr short DO = D / DSPLIT;          // output dims of this threadgroup
    constexpr short TD = DO / UD, TDs = D / UDs, TKs = BK / UKs, TK = BK / UK;

    using OSubTile = NAXSubTile<float, UQ, UD>;
    using SSubTile = NAXSubTile<float, UQ, UKs>;
    using QSubTile = NAXSubTile<T, UQ, UDs>;
    using KSubTile = NAXSubTile<T, UKs, UDs>;
    using PSubTile = NAXSubTile<float, UQ, UK>;
    using VSubTile = NAXSubTile<T, UK, UD>;

    const uint sgid = simdgroup_index_in_threadgroup;
    const uint3 tg  = threadgroup_position_in_grid;   // x = query block * DSPLIT + output-dim part, y = q head, z = batch
    const int qb = int(tg.x) / DSPLIT, dpart = int(tg.x) % DSPLIT;
    const int Tq = Tq_, Tk = Tk_, NH = n_heads, G = gqa_factor;
    const int offset = Tk - Tq;                        // absolute position of query row 0
    const int h = int(tg.y), b = int(tg.z), kvh = h / G;
    const float sc = scale, thr = threshold[h], sink = sink_val;

    const short2 coord = OSubTile::NAXFrag_t::get_coord();
    const short sm = coord.y, sn = coord.x;
    const int q0 = qb * BQ + UQ * int(sgid);   // first query row of this simdgroup
    const int64_t ks = k_strides[2], vs = v_strides[2];
#if QK8
    device const char* Qp = q + ((int64_t(b) * NH + h) * Tq + q0 + sm) * D + sn;
    device const float* qscale = (device const float*)(q + int64_t(NH) * Tq * D) + int64_t(h) * Tq;
    float qsc[2];
    for (short i = 0; i < 2; i++) qsc[i] = q0 + sm + 8 * i < Tq ? qscale[q0 + sm + 8 * i] : 0.0f;
#else
    const device T* Qp = q + ((int64_t(b) * NH + h) * Tq + q0 + sm) * D + sn;
#endif
#if KV8
    // the cache's [2][maxS][128] rows, then their scales
    const int maxS = int(k_strides[1] / D);
    device const char* K8 = k + kvh * k_strides[1];
    device const char* V8 = v + kvh * v_strides[1];
    device const float* kscale = kv_scales(k, maxS) + kvh * maxS;
    device const float* vscale = kv_scales(v, maxS) + kvh * maxS;
#else
    const device T* Kp = k + b * k_strides[0] + kvh * k_strides[1] + sm * ks + sn;
    const device T* Vp = v + b * v_strides[0] + kvh * v_strides[1] + sm * vs + sn + dpart * DO;
#endif

    NAXTile<float, 1, TD, OSubTile> Otile;
    Otile.clear();
    constexpr short kRowsPT = decltype(Otile)::kRowsPerThread;
    metal::vec<float, kRowsPT> sum_p(0.0f);

    const short lim_rows_q = short(min(Tq - (q0 + sm), 32767));
    const bool q_partial = q0 + UQ > Tq;
    // causal: the block's last query sees keys up to its position; nothing past Tk
    const int last_key = min(Tk - 1, qb * BQ + BQ - 1 + offset);
    const int n_kb = last_key < 0 ? 0 : last_key / BK + 1;
    // the first key block that can reach past this simdgroup's first query (needs the causal mask)
    const int kb_diag = (q0 + offset) / BK;
    // sliding window: the earliest key any query of this threadgroup sees, and the first key this simdgroup's LAST row
    // still sees (blocks from there on need the window mask too)
    const int kb_lo = WIN > 0 ? max(0, qb * BQ + offset - WIN + 1) / BK : 0;
    const int kb_win = WIN > 0 ? max(0, q0 + UQ - 1 + offset - WIN + 1) / BK : 0;

#if KV8
    // each key block's K and V rows, converted once for the 4 simdgroups (V only when one needs it)
    static_assert(BK == 32 && WM == 4 && DSPLIT == 1, "staging: 128 threads, 32 rows");
    threadgroup T kst[BK * SLD], vst[BK * SLD];
#if QK8 || PV8
    threadgroup int8_t kst8[QK8 ? BK * SLD8 : 1], vst8[PV8 ? BK * SLD8 : 1];
#endif
    threadgroup bool tlive[WM];
#endif
    for (int kb = kb_lo; kb < n_kb; kb++) {
        const int k0 = kb * BK;
#if KV8
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if QK8
        stage_raw8(kst8, K8, k0, Tk, tid);
#else
        stage_rows8(kst, K8, k0, Tk, tid);
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
        // HSP_BLOCK_SKIP
        const bool k_partial = k0 + BK > Tk;
        const short lim_rows_k = short(Tk - k0 - sm);

        // S = Q·Kᵀ
        NAXTile<float, 1, TKs, SSubTile> Stile;
        Stile.clear();
#if QK8
        {
            NAXTile<int, 1, 1, NAXSubTile<int, UQ, UKs>> Si;
            Si.clear();
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < D / 32; id++) {
                NAXTile<int8_t, 1, 1, NAXSubTile<int8_t, UQ, 32>> Q8;
                NAXTile<int8_t, 1, 1, NAXSubTile<int8_t, UKs, 32>> K8t;
                if (q_partial) Q8.load_safe(Qp + id * 32, D, short2(D, lim_rows_q));
                else           Q8.load(Qp + id * 32, D);
                K8t.subtile_at(0, 0).load(kst8 + sm * SLD8 + id * 32 + sn, SLD8, Int<1>{});
                subtile_matmad_nax(Si.subtile_at(0, 0), Q8.subtile_at(0, 0), metal::false_type{},
                                   K8t.subtile_at(0, 0), metal::true_type{});
            }
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < decltype(Stile)::kElemsPerTile; i++) Stile.elems()[i] = float(Si.elems()[i]) * qsc[(i % 8) / 4];
        }
        for (short ik = 0; ik < 0; ik++) {
#else
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TKs; ik++) {
#endif
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TDs; id++) {
                NAXTile<T, 1, 1, QSubTile> Qtile;
                NAXTile<T, 1, 1, KSubTile> Ktile;
                if (q_partial) Qtile.load_safe(Qp + id * UDs, D, short2(D, lim_rows_q));
                else           Qtile.load(Qp + id * UDs, D);
#if KV8
                Ktile.subtile_at(0, 0).load(kst + (ik * UKs + sm) * SLD + id * UDs + sn, SLD, Int<1>{});
#else
                const device T* kp = Kp + int64_t(k0 + ik * UKs) * ks + id * UDs;
                if (k_partial) Ktile.load_safe(kp, int(ks), short2(D, short(lim_rows_k - ik * UKs)));
                else           Ktile.load(kp, int(ks));
#endif
                subtile_matmad_nax(Stile.subtile_at(0, ik), Qtile.subtile_at(0, 0), metal::false_type{},
                                   Ktile.subtile_at(0, 0), metal::true_type{});
            }
        }

        // P = ReLU²(S·scale − τ), retiled 16×16 (same element order), masked
        NAXTile<float, 1, TK, PSubTile> Ptile;
#if KV8
        // S and P share the element order: element i is key k0 + (i / 8) · 16 + sn + i % 4
        float kscl[TK][4], vscl[TK][4];
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < 4; jj++) {
                const int c = k0 + ik * UK + sn + jj;
                kscl[ik][jj] = c < Tk ? kscale[c] : 0.0f;
                vscl[ik][jj] = c < Tk ? vscale[c] : 0.0f;
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < decltype(Stile)::kElemsPerTile; i++) {
            const float s = Stile.elems()[i] * kscl[i / 8][i % 4] * sc - thr;
            Ptile.elems()[i] = s > 0.0f ? s * s : 0.0f;
        }
#else
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < decltype(Stile)::kElemsPerTile; i++) {
            const float s = Stile.elems()[i] * sc - thr;
            Ptile.elems()[i] = s > 0.0f ? s * s : 0.0f;
        }
#endif
        if (k_partial || kb >= kb_diag || q_partial || (WIN > 0 && kb <= kb_win)) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik++) {
                thread auto& fg = Ptile.subtile_at(0, ik).frag_at(0, 0);
                STEEL_PRAGMA_UNROLL
                for (short ii = 0; ii < PSubTile::kFragThrRows; ii++) {
                    STEEL_PRAGMA_UNROLL
                    for (short jj = 0; jj < PSubTile::kFragThrCols; jj++) {
                        const int r = q0 + sm + ii * PSubTile::kFragRowsJump;       // query row
                        const int c = k0 + ik * UK + sn + jj;                        // key
                        const short loc = ii * PSubTile::kFragThrCols + jj;
                        if (c >= Tk || c > r + offset || r >= Tq || (WIN > 0 && c <= r + offset - WIN)) fg[loc] = 0.0f;
                    }
                }
            }
        }
        Ptile.template row_reduce<SumOp>(sum_p);

        // O += P·V, unless this simdgroup's whole P tile is zero
        bool live = false;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < decltype(Ptile)::kElemsPerTile; i++) live = live || Ptile.elems()[i] > 0.0f;
#if KV8
        const bool sg_live = simd_any(live);
        if (simd_is_first()) tlive[sgid] = sg_live;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tlive[0] || tlive[1] || tlive[2] || tlive[3]) {
#if PV8
            stage_raw8(vst8, V8, k0, Tk, tid);
#else
            stage_rows8(vst, V8, k0, Tk, tid);
#endif
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (sg_live) {
#else
        if (simd_any(live)) {
#endif
#if KV8
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < decltype(Ptile)::kElemsPerTile; i++) Ptile.elems()[i] *= vscl[i / 8][i % 4];
#endif
#if PV8
            // P (with V's scales) as int8 per query row of this block
            metal::vec<float, kRowsPT> pmax(0.0f);
            Ptile.template row_reduce<MaxOp>(pmax);
            NAXTile<int8_t, 1, 1, NAXSubTile<int8_t, UQ, BK>> P8;
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < decltype(Ptile)::kElemsPerTile; i++) {
                const float m = pmax[(i % 8) / 4];
                P8.elems()[i] = int8_t(m > 0.0f ? rint(Ptile.elems()[i] * (127.0f / m)) : 0.0f);
            }
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id++) {
                NAXTile<int, 1, 1, NAXSubTile<int, UQ, UD>> Oi;
                NAXTile<int8_t, 1, 1, NAXSubTile<int8_t, BK, UD>> V8t;
                Oi.clear();
                V8t.subtile_at(0, 0).load(vst8 + sm * SLD8 + id * UD + sn, SLD8, Int<1>{});
                subtile_matmad_nax(Oi.subtile_at(0, 0), P8.subtile_at(0, 0), metal::bool_constant<false>{},
                                   V8t.subtile_at(0, 0), metal::bool_constant<false>{});
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < decltype(Oi)::kElemsPerTile; i++)
                    Otile.subtile_at(0, id).elems()[i] += float(Oi.elems()[i]) * (pmax[(i % 8) / 4] / 127.0f);
            }
            if (false) {
#endif
            // PHALF: P in fp16 for the P·V multiply (both operands half), as the simdgroup-matrix kernel does
            NAXTile<T, 1, TK, NAXSubTile<T, UQ, UK>> Ph;
            if (PHALF) {
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < decltype(Ptile)::kElemsPerTile; i++) Ph.elems()[i] = T(Ptile.elems()[i]);
            }
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id++) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    NAXTile<T, 1, 1, VSubTile> Vtile;
#if KV8
                    Vtile.subtile_at(0, 0).load(vst + (ik * UK + sm) * SLD + id * UD + sn, SLD, Int<1>{});
#else
                    const device T* vp = Vp + int64_t(k0 + ik * UK) * vs + id * UD;
                    if (k_partial) Vtile.load_safe(vp, int(vs), short2(D, short(lim_rows_k - ik * UK)));
                    else           Vtile.load(vp, int(vs));
#endif
                    if (PHALF) subtile_matmad_nax(Otile.subtile_at(0, id), Ph.subtile_at(0, ik), metal::bool_constant<false>{},
                                                  Vtile.subtile_at(0, 0), metal::bool_constant<false>{});
                    else       subtile_matmad_nax(Otile.subtile_at(0, id), Ptile.subtile_at(0, ik), metal::bool_constant<false>{},
                                                  Vtile.subtile_at(0, 0), metal::bool_constant<false>{});
                }
            }
#if PV8
            }
#endif
        }
    }

    // out = O / (sink + Σ p)
    metal::vec<float, kRowsPT> L;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; i++) L[i] = sink + sum_p[i];
    metal::vec<float, kRowsPT> rcp;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; i++) rcp[i] = 1.0f / L[i];
    Otile.template row_bin_op<MulOp>(rcp);
    device T* Op = out + int64_t(q0 + sm) * OUTLD + h * D + sn + dpart * DO;   // embla-metal: token-major
    if (q_partial) {
        if (lim_rows_q > 0) Otile.store_safe(Op, OUTLD, short2(D, lim_rows_q));
    } else {
        Otile.store(Op, OUTLD);
    }
}
