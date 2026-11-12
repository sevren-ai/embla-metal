// Y = X Wᵀ for a batch of tokens (prefill): X [T, K] bf16, W a 4-bit matrix of the model
// (MLX's layout, as in gemv.metal: [O, K/8] words, [O, K/64] scales and biases), Y [T, O]. A
// threadgroup computes BM tokens × BN rows, stepping through the inputs one quantization group (64)
// at a time: it stages the X tile and the dequantized W tile (bf16, as MLX's own GEMM does) in
// threadgroup memory, and its 8 simdgroups multiply 8x8 fragments into f32 sums.
//
// EXPERTS: the MoE's routed experts. Threadgroup row y is a tile of up to BM entries of one expert
// (tiles[y] = expert, first entry, entries; past ntiles: none), W is that expert's matrix
// (wo + expert · ew), and with GATHER entry r's X row is xrow[r] (its token), else r.
//
// EPI: 0 y = bf16, 1 y = relu(bf16)² as bf16, 2 y (f32) += bf16, 3 y = bf16 as f32, 4 y = f32.

#include <metal_simdgroup_matrix>

#ifndef EPI
#define EPI 0
#endif
#ifndef EXPERTS
#define EXPERTS 0
#endif
#ifndef GATHER
#define GATHER 0
#endif

struct GemmParams {
    int T, O, ldx, ldy, K, pad0, pad1, pad2;
    ulong3 ew;   // EXPERTS: bytes between experts' words, scales, biases
};

#ifndef BM
#define BM 32          // tokens a threadgroup
#endif
#ifndef TILE
#define TILE float     // the staged tiles' and fragments' type (bfloat: as fast here, as the reference rounds); f32 sums
#endif
constant constexpr int BN = 64, BK = 64, PAD = 8;
// 8 simdgroups: SGM along the tokens (16 each), SGN along the rows (NF fragments of 8 each)
constant constexpr int SGM = BM / 16, SGN = 8 / SGM, NF = BN / SGN / 8;
constant constexpr int XS = BM * (BK + PAD) * sizeof(TILE), WS = BN * (BK + PAD) * sizeof(TILE);
constant constexpr int OS = BM * (BN + PAD) * 4;
constant constexpr int SMEM = (XS + WS) > OS ? (XS + WS) : OS;

kernel void gemm(
    device const uchar*  W  [[buffer(0)]],
    constant ulong3&     wo [[buffer(1)]],   // byte offsets in W: words, scales, biases
    device const bfloat* X  [[buffer(2)]],   // [T, ldx]
    device void*         Y  [[buffer(3)]],   // [T, ldy]
    constant GemmParams& gp [[buffer(4)]],
    device const int4*   tiles [[buffer(5)]],   // EXPERTS: expert, first entry, entries
    device const int*    ntiles [[buffer(6)]],
    device const int*    xrow  [[buffer(7)]],   // GATHER
    uint2 tg   [[threadgroup_position_in_grid]],
    uint  t    [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int O = gp.O, ldx = gp.ldx, ldy = gp.ldy, K = gp.K;
    const int n0 = int(tg.x) * BN;
    int m0 = int(tg.y) * BM, T = gp.T;
    ulong3 off = wo;
#if EXPERTS
    if (int(tg.y) >= ntiles[0]) return;
    const int4 tile = tiles[tg.y];
    off += ulong(tile.x) * gp.ew;
    m0 = tile.y;
    T = tile.y + tile.z;
#endif
    device const uint*   w  = (device const uint*)(W + off.x);
    device const bfloat* sc = (device const bfloat*)(W + off.y);
    device const bfloat* bs = (device const bfloat*)(W + off.z);

    // the staged tiles, and afterwards the output tile in the same memory
    threadgroup uchar smem[SMEM];
    threadgroup TILE (*xs)[BK + PAD] = (threadgroup TILE (*)[BK + PAD])smem;
    threadgroup TILE (*ws)[BK + PAD] = (threadgroup TILE (*)[BK + PAD])(smem + XS);
    // simdgroup (i, j): tokens 16 i .., rows (BN / SGN) j ..
    const int si = int(sg) / SGN, sj = int(sg) % SGN;
    simdgroup_matrix<float, 8, 8> acc[2][NF];
    for (int a = 0; a < 2; a++)
        for (int b = 0; b < NF; b++) acc[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // X tile: BM x 64 bf16, 8 values a thread a pass
        for (int i = int(t); i < BM * 8; i += 256) {
            const int r = i / 8, c = (i % 8) * 8;
            const int m = min(m0 + r, T - 1);
            const int xr = GATHER ? xrow[m] : m;
            device const bfloat4* xp = (device const bfloat4*)(X + size_t(xr) * ldx + k0 + c);
            const float4 a = float4(xp[0]), b = float4(xp[1]);
            xs[r][c + 0] = TILE(a.x); xs[r][c + 1] = TILE(a.y); xs[r][c + 2] = TILE(a.z); xs[r][c + 3] = TILE(a.w);
            xs[r][c + 4] = TILE(b.x); xs[r][c + 5] = TILE(b.y); xs[r][c + 6] = TILE(b.z); xs[r][c + 7] = TILE(b.w);
        }
        // W tile: 64 rows x 64 inputs (one group: 8 words, one scale and bias a row), 2 words a thread
        for (int i = int(t); i < BN * 8; i += 256) {
            const int r = i / 8, c = (i % 8) * 8;
            const int n = min(n0 + r, O - 1);
            const uint word = w[size_t(n) * (K / 8) + (k0 + c) / 8];
            const float s = float(sc[size_t(n) * (K / 64) + k0 / 64]), b = float(bs[size_t(n) * (K / 64) + k0 / 64]);
            for (int q = 0; q < 8; q++) ws[r][c + q] = TILE(s * float((word >> (4 * q)) & 0xFu) + b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<TILE, 8, 8> a[2], b[NF];
            for (int i = 0; i < 2; i++) simdgroup_load(a[i], &xs[16 * si + 8 * i][kk], BK + PAD);
            for (int j = 0; j < NF; j++) simdgroup_load(b[j], &ws[8 * NF * sj + 8 * j][kk], BK + PAD, ulong2(0, 0), true);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < NF; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // through threadgroup memory to the epilogue
    threadgroup float* out = (threadgroup float*)smem;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < NF; j++) simdgroup_store(acc[i][j], out + (16 * si + 8 * i) * (BN + PAD) + 8 * NF * sj + 8 * j, BN + PAD);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < BM * BN; i += 256) {
        const int r = i / BN, c = i % BN, m = m0 + r, n = n0 + c;
        if (m >= T || n >= O) continue;
        const float v = out[r * (BN + PAD) + c];
        const size_t at = size_t(m) * ldy + n;
#if EPI == 0
        ((device bfloat*)Y)[at] = bfloat(v);
#elif EPI == 1
        const float u = max(bf(v), 0.0f);
        ((device bfloat*)Y)[at] = bfloat(u * u);
#elif EPI == 2
        ((device float*)Y)[at] += bf(v);
#elif EPI == 3
        ((device float*)Y)[at] = bf(v);
#else
        ((device float*)Y)[at] = v;
#endif
    }
}
