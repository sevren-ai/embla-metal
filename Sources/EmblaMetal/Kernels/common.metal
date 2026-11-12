// Shared by every kernel file: the model's sizes and small helpers. Each file is compiled with this
// prepended, and with its own preprocessor macros per specialization.
#include <metal_stdlib>
using namespace metal;

constant constexpr int D = 2048;                   // model width
constant constexpr int HD = 128;                   // head dim
constant constexpr float EPS = 0.0078125f;         // rms_norm_eps, 2^-7
constant constexpr float EPS32 = 1.1920929e-07f;   // fp32 eps: head norms, engram norms
constant constexpr float NEG = -1e30f;             // -inf for running maxima (fast math has no inf)

/// The value v takes once stored as bfloat.
inline float bf(float v) { return float(bfloat(v)); }

/// Sum over a threadgroup of TG threads; `buf` holds TG / 32 floats.
template <int TG>
inline float block_sum(float v, threadgroup float* buf, uint lane, uint sg) {
    v = simd_sum(v);
    if (TG > 32) {
        if (lane == 0) buf[sg] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v = simd_sum(lane < TG / 32 ? buf[lane] : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return v;
}

/// RMS norm of a head held 4 dims a lane (dims lane + 32 i), times the gains 1 + w of the first
/// `ng` of g1, g2, in fp32 like the reference.
inline void head_norm(thread float* v, device const bfloat* g1, device const bfloat* g2, int ng, uint lane) {
    float ss = 0.0f;
    for (int i = 0; i < 4; i++) ss += v[i] * v[i];
    const float r = precise::rsqrt(simd_sum(ss) / HD + EPS32);
    for (int i = 0; i < 4; i++) {
        const int d = int(lane) + 32 * i;
        float g = r;
        if (ng >= 1) g *= 1.0f + float(g1[d]);
        if (ng >= 2) g *= 1.0f + float(g2[d]);
        v[i] *= g;
    }
}

/// RoPE (non-traditional) on the first 64 dims at position pos, by -θ as the model was trained:
/// dims lane and lane + 32 rotate together.
inline void rope(thread float* v, int pos, uint lane) {
    const float theta = -float(pos) * precise::exp2(-float(lane) / 32.0f * 13.287712379549449f);
    const float c = precise::cos(theta), s = precise::sin(theta);
    const float x1 = v[0], x2 = v[1];
    v[0] = x1 * c - x2 * s;
    v[1] = x1 * s + x2 * c;
}

/// The full layers' caches: bf16 rows, or with KV8 int8 rows each with a float scale (absmax / 127),
/// the scales after the rows of both kv heads.
#ifndef KV8
#define KV8 0
#endif
#if KV8
typedef char kv_t;
#else
typedef bfloat kv_t;
#endif
inline device const float* kv_scales(device const char* c, int maxS) { return (device const float*)(c + 2 * size_t(maxS) * HD); }

/// The simdgroup matrix kernels (attn_full8, attn_sparse8) load a lane's int8 bytes as contiguous
/// runs, so their 8x8 slices take the head dims in another order: Q's slice k holds dims k, 16 + k,
/// .. (qperm: where dim d goes), and O's slice n columns c the dims (c / 2) · 32 + 2n + c % 2
/// (ocol: the column of dim d).
inline int qperm(int d) { return 8 * (d % 16) + d / 16; }
inline int ocol(int d) { return 8 * ((d % 32) / 2) + (d / 32) * 2 + d % 2; }

/// Writes a head row (4 values a lane, dims lane + 32 i, bf16-rounded first) as int8 at `row` of
/// the cache at `base` (2 · maxS rows, then their scales).
inline void write_row8(device char* base, size_t row, int maxS, thread const float* x, uint lane) {
    float a = 0.0f, v[4];
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[i]);
        a = max(a, fabs(v[i]));
    }
    a = simd_max(a);
    const float inv = a > 0.0f ? 127.0f / a : 0.0f;
    for (int i = 0; i < 4; i++) base[row * HD + lane + 32 * i] = char(int(rint(v[i] * inv)));
    if (lane == 0) ((device float*)(base + 2 * size_t(maxS) * HD))[row] = a / 127.0f;
}

/// Last threadgroup of a dispatch: true in every thread of the threadgroup that arrives last at
/// `counter` (which it resets), after this threadgroup's device writes are made visible.
inline bool arrive_last(device atomic_uint* counter, uint total, uint t, threadgroup bool* flag) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (t == 0) {
        const uint n = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        *flag = n == total - 1;
        if (*flag) atomic_store_explicit(counter, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    return *flag;
}

/// The 4-bit dot products of NR rows with x, a lane holding 64 inputs: the word loads are issued
/// by `gemv_load` first (they do not depend on x), `gemv_dot` does the math once x is known.
/// MLX's layout: words [O, 2048/8] u32 low nibble first, scales and biases [O, 32] bf16.
template <int NR>
struct GemvRows {
    uint4 q[NR][2];
    float a[NR][2], b[NR][2];

    void load(device const uint4* w, device const bfloat* sc, device const bfloat* bs, int r0, uint lane) {
        for (int u = 0; u < NR; u++) {
            const size_t r = size_t(r0 + u);
            for (int j = 0; j < 2; j++) {
                const int c = int(lane) + 32 * j;
                q[u][j] = w[r * 64 + c];
                a[u][j] = float(sc[r * 32 + c / 2]);
                b[u][j] = float(bs[r * 32 + c / 2]);
            }
        }
    }

    /// acc[u] = row u · x (summed over the simdgroup), x4 = x as bfloat4 (device or threadgroup).
    template <typename XP4>
    void dot(XP4 x4, uint lane, thread float* acc) {
        // x pre-scaled by 16^-n for the nibble n of its word, so a nibble is masked in place.
        float xr[64], xs[2];
        for (int j = 0; j < 2; j++) {
            float s = 0.0f;
            for (int e = 0; e < 8; e++) {
                const float4 v = float4(x4[(int(lane) + 32 * j) * 8 + e]);
                const float f = (e % 2) ? 1.0f / 65536.0f : 1.0f;
                xr[32 * j + 4 * e + 0] = v.x * f;
                xr[32 * j + 4 * e + 1] = v.y * f * (1.0f / 16.0f);
                xr[32 * j + 4 * e + 2] = v.z * f * (1.0f / 256.0f);
                xr[32 * j + 4 * e + 3] = v.w * f * (1.0f / 4096.0f);
                s += (v.x + v.y) + (v.z + v.w);
            }
            xs[j] = s;
        }
        for (int u = 0; u < NR; u++) {
            float r = 0.0f;
            for (int j = 0; j < 2; j++) {
                const uint wv[4] = {q[u][j].x, q[u][j].y, q[u][j].z, q[u][j].w};
                float dq = 0.0f;
                for (int k = 0; k < 4; k++) {
                    for (int n = 0; n < 8; n++) dq += float(wv[k] & (0xFu << (4 * n))) * xr[32 * j + 8 * k + n];
                }
                r += a[u][j] * dq + b[u][j] * xs[j];
            }
            acc[u] = simd_sum(r);
        }
    }
};

/// The K RMS-normed mixes of one sub-layer input (columns col ..), from the precomputed M0 and stats
/// of `prep` and, when MERGE, the partial sum x: m = M0 + w (x - M0), w = σ(q·rms(x) - lse), with
/// |m|² expanded from |M0|², M0·x and |x|². Computed by a whole threadgroup of TGS threads into
/// `out [K][D]` (bf16, threadgroup); threadgroup 0 also writes them to `mixout` when given.
/// `red` holds (TGS / 32) * (2K + 1) + 2K floats.
template <int NQ, bool MERGE, int TGS>
inline void mix_input(
    device const float* queries, device const bfloat* m0, device const float4* stats, int col,
    device const float* x, threadgroup bfloat* out, device bfloat* mixout, bool write,
    threadgroup float* red, uint t, uint lane, uint sg)
{
    constexpr int E = D / TGS, SG = TGS / 32, NP = 2 * NQ + 1;
    float scale[NQ], w[NQ];
    float xv[E];
    if (MERGE) {
        float acc[NP];
        for (int p = 0; p < NP; p++) acc[p] = 0.0f;
        for (int e = 0; e < E; e++) {
            const int d = int(t) + e * TGS;
            xv[e] = x[d];
            acc[2 * NQ] += xv[e] * xv[e];
            for (int j = 0; j < NQ; j++) {
                acc[2 * j] += queries[(col + j) * D + d] * xv[e];
                acc[2 * j + 1] += float(m0[(col + j) * D + d]) * xv[e];
            }
        }
        for (int p = 0; p < NP; p++) {
            const float s = simd_sum(acc[p]);
            if (lane == 0) red[sg * NP + p] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot[NP];
        for (int p = 0; p < NP; p++) {
            float s = 0.0f;
            for (int i = 0; i < SG; i++) s += red[i * NP + p];
            tot[p] = s;
        }
        const float xx = tot[2 * NQ], rx = precise::rsqrt(xx / D + EPS);
        for (int j = 0; j < NQ; j++) {
            const float4 st = stats[col + j];
            w[j] = 1.0f / (1.0f + precise::exp(st.x - tot[2 * j] * rx));
            const float u = 1.0f - w[j];
            const float ss = u * u * st.y + 2.0f * w[j] * u * tot[2 * j + 1] + w[j] * w[j] * xx;
            scale[j] = precise::rsqrt(ss / D + EPS);
        }
    } else {
        for (int j = 0; j < NQ; j++) {
            w[j] = 0.0f;
            scale[j] = precise::rsqrt(stats[col + j].y / D + EPS);
        }
    }
    for (int j = 0; j < NQ; j++) {
        for (int e = 0; e < E; e++) {
            const int d = int(t) + e * TGS;
            float m = float(m0[(col + j) * D + d]);
            if (MERGE) m += w[j] * (xv[e] - m);
            const bfloat r = bfloat(m * scale[j]);
            out[j * D + d] = r;
            if (write) mixout[j * D + d] = r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
