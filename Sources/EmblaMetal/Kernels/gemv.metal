// y = W x for one token over stacked segments: every segment is a 4-bit matrix of the model with
// its own input and output, so q, k and v (or several layers' k and v) are one dispatch.
//
// A simdgroup owns ROWS rows (GemvRows: a row is one coalesced 1 KB read by the 32 lanes, its loads
// issued before x is read).
//
// MIX: the inputs are the K Block AttnRes mixes from column `col`, which every threadgroup computes
// from x and prep's M0 (mix_input) while its weight loads are in flight; segment x offsets then
// index the mixes, and threadgroup 0 writes them to `mixout` (as X).

//
// EPI: 0 y = bf16, 1 y = relu(bf16)² as bf16, 2 x += bf16 (x f32), 3 x = bf16, 4 y = f32.

#ifndef EPI
#define EPI 0
#endif
#ifndef NSG
#define NSG 8
#endif
#ifndef ROWS
#define ROWS 4
#endif
#ifndef MIX
#define MIX 0
#define K 1
#define HAS_X 0
#endif

struct Seg {
    ulong w, s, b;  // byte offsets in W of the words, scales and biases
    uint end;       // end row (exclusive) of the segment in the stacked rows; a multiple of ROWS
    uint x;         // element offset of the segment's input in X (or in the mixes)
    uint y;         // element offset of its first output in Y
    uint pad;
};

kernel void gemv(
    device const uchar*  W    [[buffer(0)]],
    device const bfloat* X    [[buffer(1)]],
    device void*         Y    [[buffer(2)]],
    constant Seg*        segs [[buffer(3)]],
    constant int&        nseg [[buffer(4)]],
    device const float*  queries [[buffer(7)]],
    device const bfloat* m0      [[buffer(8)]],
    device const float4* stats   [[buffer(9)]],
    device const float*  xres    [[buffer(10)]],
    constant int&        col     [[buffer(11)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint t    [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int row0 = (int(tg) * NSG + int(sg)) * ROWS;
    int si = 0, start = 0;
    while (si < nseg && row0 >= int(segs[si].end)) {
        start = int(segs[si].end);
        si++;
    }
    const bool active = si < nseg;
    const Seg seg = segs[min(si, nseg - 1)];
    const int r0 = row0 - start;
    GemvRows<ROWS> rows;
    if (active) {
        rows.load((device const uint4*)(W + seg.w), (device const bfloat*)(W + seg.s), (device const bfloat*)(W + seg.b), r0, lane);
    }
    float acc[ROWS];
#if MIX
    threadgroup bfloat xm[K * D];
    threadgroup float red[NSG * (2 * K + 1) + 2 * K];
    mix_input<K, HAS_X, NSG * 32>(queries, m0, stats, col, xres, xm, (device bfloat*)X, tg == 0, red, t, lane, sg);
    if (active) rows.dot((threadgroup const bfloat4*)(xm + seg.x), lane, acc);
#else
    if (active) rows.dot((device const bfloat4*)(X + seg.x), lane, acc);
#endif
    if (active && lane == 0) {
        for (int u = 0; u < ROWS; u++) {
            const uint r = seg.y + uint(r0 + u);
#if EPI == 0
            ((device bfloat*)Y)[r] = bfloat(acc[u]);
#elif EPI == 1
            const float v = max(bf(acc[u]), 0.0f);
            ((device bfloat*)Y)[r] = bfloat(v * v);
#elif EPI == 2
            ((device float*)Y)[r] += bf(acc[u]);
#elif EPI == 3
            ((device float*)Y)[r] = bf(acc[u]);
#else
            ((device float*)Y)[r] = acc[u];
#endif
        }
    }
}
