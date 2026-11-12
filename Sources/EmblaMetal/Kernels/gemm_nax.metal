// gemm.metal on the M5's neural accelerators: the same Y = X Wᵀ (4-bit W dequantized in
// threadgroup memory, the same EPI, EXPERTS and GATHER), but the products are Metal 4 tensor ops
// (MLX's NAX tiles, nax.metal). A threadgroup of 4 simdgroups computes 64 tokens × 64 rows, each
// simdgroup a 32 × 32 subtile, 16 inputs a step. NBM = 32 makes it 32 tokens × 128 rows. (128-row tiles need the output staged in two parts
// to fit threadgroup memory; writing it from the fragments instead was slower.)

#ifndef EPI
#define EPI 0
#endif
#ifndef EXPERTS
#define EXPERTS 0
#endif
#ifndef GATHER
#define GATHER 0
#endif
#ifndef NT
#define NT half   // the staged tiles' type (bfloat: as the reference rounds); f32 sums
#endif

#ifndef NBM
#define NBM 64    // tokens a threadgroup (64 x 64, or 32 x 128 for the routed experts' small tiles)
#endif
#ifndef NPAD
#define NPAD 8
#endif
#ifndef MT
#define MT 1      // token subtiles of NBM a threadgroup sharing each dequantized W tile (2 measured slower)
#endif
constant constexpr int NBN = 4096 / NBM, NBK = 64, NTH = 128;   // 4 simdgroups of 32 x 32
constant constexpr int SGC = NBN / 32;                                   // simdgroups along the rows
constant constexpr int NXS = MT * NBM * (NBK + NPAD) * sizeof(NT), NWS = NBN * (NBK + NPAD) * sizeof(NT);
constant constexpr int NOS = NBM * (NBN + 4) * 4;
constant constexpr int NSMEM = (NXS + NWS) > NOS ? (NXS + NWS) : NOS;

struct GemmParams {
    int T, O, ldx, ldy, K, pad0, pad1, pad2;
    ulong3 ew;
};

kernel void gemm_nax(
    device const uchar*  W      [[buffer(0)]],
    constant ulong3&     wo     [[buffer(1)]],
    device const bfloat* X      [[buffer(2)]],
    device void*         Y      [[buffer(3)]],
    constant GemmParams& gp     [[buffer(4)]],
    device const int4*   tiles  [[buffer(5)]],
    device const int*    ntiles [[buffer(6)]],
    device const int*    xrow   [[buffer(7)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint  t   [[thread_index_in_threadgroup]],
    uint  sg  [[simdgroup_index_in_threadgroup]])
{
    const int O = gp.O, ldx = gp.ldx, ldy = gp.ldy, K = gp.K;
    const int n0 = int(tg.x) * NBN;
    int m0 = int(tg.y) * NBM, T = gp.T;
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

    threadgroup uchar smem[NSMEM];
    threadgroup NT (*xs)[NBK + NPAD] = (threadgroup NT (*)[NBK + NPAD])smem;
    threadgroup NT (*ws)[NBK + NPAD] = (threadgroup NT (*)[NBK + NPAD])(smem + NXS);

    using CSub = NAXSubTile<float, 32, 32>;
    using ASub = NAXSubTile<NT, 32, 16>;
    const short2 coord = CSub::NAXFrag_t::get_coord();
    const short sm = coord.y, sn = coord.x;
    const int si = int(sg) / SGC, sj = int(sg) % SGC;
    CSub c[MT];
    STEEL_PRAGMA_UNROLL
    for (int u = 0; u < MT; u++) c[u].clear();

    for (int k0 = 0; k0 < K; k0 += NBK) {
        // X tile: MT x NBM tokens x 64, 8 values a thread a pass
        for (int i = int(t); i < MT * NBM * 8; i += NTH) {
            const int r = i / 8, cc = (i % 8) * 8;
            const int m = min(m0 + r, T - 1);
            const int xr = GATHER ? xrow[m] : m;
            device const bfloat4* xp = (device const bfloat4*)(X + size_t(xr) * ldx + k0 + cc);
            *(threadgroup vec<NT, 4>*)&xs[r][cc] = vec<NT, 4>(float4(xp[0]));
            *(threadgroup vec<NT, 4>*)&xs[r][cc + 4] = vec<NT, 4>(float4(xp[1]));
        }
        // W tile: 64 rows x 64 inputs (one quantization group), a word a thread a pass
        for (int i = int(t); i < NBN * 8; i += NTH) {
            const int r = i / 8, cc = (i % 8) * 8;
            const int n = min(n0 + r, O - 1);
            const uint word = w[size_t(n) * (K / 8) + (k0 + cc) / 8];
            const float s = float(sc[size_t(n) * (K / 64) + k0 / 64]), b = float(bs[size_t(n) * (K / 64) + k0 / 64]);
            const float4 lo = float4(word & 0xFu, (word >> 4) & 0xFu, (word >> 8) & 0xFu, (word >> 12) & 0xFu);
            const float4 hi = float4((word >> 16) & 0xFu, (word >> 20) & 0xFu, (word >> 24) & 0xFu, word >> 28);
            *(threadgroup vec<NT, 4>*)&ws[r][cc] = vec<NT, 4>(s * lo + b);
            *(threadgroup vec<NT, 4>*)&ws[r][cc + 4] = vec<NT, 4>(s * hi + b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (int kk = 0; kk < NBK; kk += 16) {
            ASub b;
            b.load(&ws[32 * sj + sm][kk + sn], NBK + NPAD, Int<1>{});
            STEEL_PRAGMA_UNROLL
            for (int u = 0; u < MT; u++) {
                ASub a;
                a.load(&xs[u * NBM + 32 * si + sm][kk + sn], NBK + NPAD, Int<1>{});
                subtile_matmad_nax(c[u], a, metal::false_type{}, b, metal::true_type{});
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // through threadgroup memory to the epilogue (coalesced; straight from the fragments is slower),
    // a token subtile at a time
    threadgroup float* out = (threadgroup float*)smem;
    STEEL_PRAGMA_UNROLL
    for (int u = 0; u < MT; u++) {
    if (u > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
    c[u].store(out + (32 * si + sm) * (NBN + 4) + 32 * sj + sn, NBN + 4, Int<1>{});
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < NBM * NBN; i += NTH) {
        const int r = i / NBN, cc = i % NBN, m = m0 + u * NBM + r, n = n0 + cc;
        if (m >= T || n >= O) continue;
        const float v = out[r * (NBN + 4) + cc];
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
}
