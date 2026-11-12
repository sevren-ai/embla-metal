// W4A8 prefill GEMMs (opt-in, EMBLA_W4A8=1): the
// activations quantized to int8 per (row, 64-input group), the 4-bit weights multiplied as their
// integer codes on the matrix units' int8 path (int32 sums), one matmul a 64-group rescaled into f32:
//   Σ_k x_k·w_k = s_x·s_w·Σ_k x̂_k·q_k + b_w·s_x·Σ_k x̂_k   (w = s_w·q + b_w, x ≈ s_x·x̂)
// No weight is dequantized: a word's 8 codes become int8 bytes in two masks, w & 0x0F0F0F0F (codes
// 0, 2, 4, 6) and (w >> 4) & 0x0F0F0F0F (1, 3, 5, 7), and the activations are stored in that order
// (q8_act), so each dot product is the same sum. Exact but for the activations' rounding (~0.4 % of
// a group's largest value), which the bf16 paths do not have: hence opt-in.

#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

#ifndef EPI
#define EPI 0
#endif
#ifndef EXPERTS
#define EXPERTS 0
#endif
#ifndef GATHER
#define GATHER 0
#endif

// x [T, K] bf16 (row stride ldx) -> xq [T, K] int8 in the code order above, sx [T, K/64] (scale
// max|x| / 127), xz [T, K/64] (sx · Σ x̂). Grid (32 · K/64, T): a simdgroup per (row, group), two
// consecutive inputs a lane.
kernel void q8_act(
    device const bfloat* x  [[buffer(0)]],
    device char*         xq [[buffer(1)]],
    device float*        sx [[buffer(2)]],
    device float*        xz [[buffer(3)]],
    constant int2&       kl [[buffer(4)]],   // K, ldx
    uint2 gid  [[thread_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]])
{
    const int K = kl.x, G = K / 64, g = int(gid.x) / 32, t = int(gid.y);
    const int k = g * 64 + 2 * int(lane);
    const float2 v = float2(x[size_t(t) * kl.y + k], x[size_t(t) * kl.y + k + 1]);
    const float a = simd_max(max(fabs(v.x), fabs(v.y)));
    const float inv = a > 0.0f ? 127.0f / a : 0.0f;
    const int2 q = clamp(int2(rint(v * inv)), -127, 127);
    // inputs 8b + j (j even) and 8b + j + 1 go to 8b + j/2 and 8b + 4 + j/2
    const int b8 = k & ~7, j = k & 7;
    xq[size_t(t) * K + b8 + j / 2] = char(q.x);
    xq[size_t(t) * K + b8 + 4 + j / 2] = char(q.y);
    const int s = simd_sum(q.x + q.y);
    if (lane == 0) {
        sx[size_t(t) * G + g] = a / 127.0f;
        xz[size_t(t) * G + g] = a / 127.0f * float(s);
    }
}

struct GemmParams {
    int T, O, ldx, ldy, K, pad0, pad1, pad2;
    ulong3 ew;
};

constant constexpr int BM = 64, BN = 64, NSG = 4, NTH = 32 * NSG;

// As gemm.metal (EPI, EXPERTS, GATHER), on xq / sx / xz. 64 tokens × 64 rows a threadgroup; the
// staging of group g + 1 is in flight while the matrix units multiply group g.
kernel void gemm_w4a8(
    device const uchar*  W      [[buffer(0)]],
    constant ulong3&     wo     [[buffer(1)]],
    device const char*   xq     [[buffer(2)]],
    device void*         Y      [[buffer(3)]],
    constant GemmParams& gp     [[buffer(4)]],
    device const int4*   tiles  [[buffer(5)]],
    device const int*    ntiles [[buffer(6)]],
    device const int*    xrow   [[buffer(7)]],
    device const float*  sx     [[buffer(8)]],
    device const float*  xz     [[buffer(9)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  t   [[thread_index_in_threadgroup]])
{
    const int O = gp.O, ldy = gp.ldy, K = gp.K, G = K / 64;
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
    device const bfloat* ws = (device const bfloat*)(W + off.y);
    device const bfloat* wb = (device const bfloat*)(W + off.z);

    threadgroup int8_t As[2][BM * 64], Bs[2][BN * 64];
    threadgroup int rrow[BM];                       // row m's X row (its token, gathered), -1: none
    threadgroup float2 rsc[2][BM], csc[2][BN];       // per row (s_x, s_x·Σx̂), per column (s_w, b_w)
    for (int m = int(t); m < BM; m += NTH) rrow[m] = m0 + m < T ? (GATHER ? xrow[m0 + m] : m0 + m) : -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr int NA = BM * 4 / NTH, NB = BN * 8 / NTH;   // A: 16 bytes an item; B: one word (8 codes) an item
    uint4 areg[NA];
    uint breg[NB];
    float2 rs = 0.0f, cs = 0.0f;
    const int myrow = int(t) < BM ? rrow[t] : -1;
    const int mycol = min(n0 + int(t), O - 1);
    auto load = [&](int g) {
        for (int i = 0; i < NA; i++) {
            const int j = int(t) + i * NTH, r = rrow[j / 4];
            areg[i] = r >= 0 ? *(const device uint4*)(xq + size_t(r) * K + g * 64 + (j % 4) * 16) : uint4(0u);
        }
        for (int i = 0; i < NB; i++) {
            const int j = int(t) + i * NTH, n = min(n0 + j / 8, O - 1);
            breg[i] = w[size_t(n) * (K / 8) + g * 8 + j % 8];
        }
        if (myrow >= 0) rs = float2(sx[size_t(myrow) * G + g], xz[size_t(myrow) * G + g]);
        if (int(t) < BN) cs = float2(float(ws[size_t(mycol) * G + g]), float(wb[size_t(mycol) * G + g]));
    };

    constexpr auto desc = matmul2d_descriptor(BM, BN, 64, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto tA0 = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(As[0], dextents<int32_t, 2>(64, BM));
    auto tB0 = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(Bs[0], dextents<int32_t, 2>(64, BN));
    auto tA1 = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(As[1], dextents<int32_t, 2>(64, BM));
    auto tB1 = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>(Bs[1], dextents<int32_t, 2>(64, BN));
    auto cI = op.template get_destination_cooperative_tensor<decltype(tA0), decltype(tB0), int>();
    constexpr int CAP = BM * BN / NTH;
    float acc[CAP];
    for (int i = 0; i < CAP; i++) acc[i] = 0.0f;
    load(0);
    for (int g = 0; g < G; g++) {
        // buffer g % 2 was last read in group g - 2, finished before group g - 1's barrier
        const int b = g & 1;
        for (int i = 0; i < NA; i++) {
            const int j = int(t) + i * NTH;
            *(threadgroup uint4*)(As[b] + (j / 4) * 64 + (j % 4) * 16) = areg[i];
        }
        for (int i = 0; i < NB; i++) {
            const int j = int(t) + i * NTH;
            const uint q = breg[i];
            *(threadgroup uint2*)(Bs[b] + (j / 8) * 64 + (j % 8) * 8) = uint2(q & 0x0F0F0F0Fu, (q >> 4) & 0x0F0F0F0Fu);
        }
        if (int(t) < BM) rsc[b][t] = rs;
        if (int(t) < BN) csc[b][t] = cs;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g + 1 < G) load(g + 1);
        if (b) op.run(tA1, tB1, cI); else op.run(tA0, tB0, cI);
        for (uint16_t i = 0; i < CAP; i++) if (i < cI.get_capacity() && cI.is_valid_element(i)) {
            const auto idx = cI.get_multidimensional_index(i);
            const float2 r = rsc[b][idx[1]], c = csc[b][idx[0]];
            acc[i] = fma(float(cI[i]), r.x * c.x, fma(r.y, c.y, acc[i]));
        }
    }
    for (uint16_t i = 0; i < CAP; i++) if (i < cI.get_capacity() && cI.is_valid_element(i)) {
        const auto idx = cI.get_multidimensional_index(i);
        const int n = n0 + idx[0], m = m0 + idx[1];
        if (m >= T || n >= O) continue;
        const float v = acc[i];
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
