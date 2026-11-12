import Metal

/// y = W x for one token, W in MLX's 4-bit affine layout (group 64):
/// `w [O, I/8]` uint32 (8 inputs a word, low nibble first), `scales`, `biases [O, I/64]` bf16.
///
/// A simdgroup owns ROWS output rows. A lane holds 32 consecutive inputs of x per 1024 (in registers,
/// with their sum for the biases) and reads the matching 16 bytes of each of its rows, so a row is
/// one coalesced read by the 32 lanes. All of a simdgroup's row loads are issued before the math.
public enum GEMV4 {
    public static let source = """
    #include <metal_stdlib>
    using namespace metal;

    constant int I    [[function_constant(0)]];   // input dims: 1024 or 2048
    constant int ROWS [[function_constant(1)]];   // rows per simdgroup
    constant int NSG  [[function_constant(2)]];   // simdgroups per threadgroup

    kernel void gemv4(
        device const uint4*  w      [[buffer(0)]],
        device const bfloat* scales [[buffer(1)]],
        device const bfloat* biases [[buffer(2)]],
        device const bfloat* x      [[buffer(3)]],
        device bfloat*       y      [[buffer(4)]],
        constant int&        O      [[buffer(5)]],
        uint tg   [[threadgroup_position_in_grid]],
        uint sg   [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]])
    {
        constexpr int C = 2;                       // chunks of 32 inputs (one uint4) a lane holds, at most
        const int nc = I / 1024;                   // chunks per lane
        const int W4 = I / 32;                     // uint4 per row
        const int G  = I / 64;                     // groups per row

        float xs[C];
        float xr[C * 32];
        for (int j = 0; j < nc; j++) {
            device const bfloat* xp = x + (lane + 32 * j) * 32;
            float s = 0.0f;
            for (int e = 0; e < 32; e++) {
                const float v = float(xp[e]);
                xr[32 * j + e] = v;
                s += v;
            }
            xs[j] = s;
        }

        const int row0 = (int(tg) * NSG + int(sg)) * ROWS;
        for (int r0 = 0; r0 < ROWS; r0 += 4) {
            uint4 q[4][C];
            float sc[4][C], bs[4][C];
            for (int u = 0; u < 4; u++) {
                const int row = min(row0 + r0 + u, O - 1);
                for (int j = 0; j < nc; j++) {
                    const int c = int(lane) + 32 * j;
                    q[u][j]  = w[size_t(row) * W4 + c];
                    sc[u][j] = float(scales[size_t(row) * G + c / 2]);
                    bs[u][j] = float(biases[size_t(row) * G + c / 2]);
                }
            }
            for (int u = 0; u < 4; u++) {
                float acc = 0.0f;
                for (int j = 0; j < nc; j++) {
                    const uint wv[4] = {q[u][j].x, q[u][j].y, q[u][j].z, q[u][j].w};
                    float d = 0.0f;
                    for (int k = 0; k < 4; k++) {
                        for (int n = 0; n < 8; n++) {
                            d += float((wv[k] >> (4 * n)) & 0xFu) * xr[32 * j + 8 * k + n];
                        }
                    }
                    acc += sc[u][j] * d + bs[u][j] * xs[j];
                }
                acc = simd_sum(acc);
                const int row = row0 + r0 + u;
                if (lane == 0 && row < O && r0 + u < ROWS) {
                    y[row] = bfloat(acc);
                }
            }
        }
    }
    """

    public struct Config {
        public var rows: Int
        public var simdgroups: Int
        public init(rows: Int = 4, simdgroups: Int = 8) {
            self.rows = rows
            self.simdgroups = simdgroups
        }
    }

    /// Encodes y = W x. `w`, `scales` and `biases` are (buffer, byte offset) pairs.
    public static func encode(
        _ ctx: MetalContext, _ enc: MTLComputeCommandEncoder,
        w: (MTLBuffer, Int), scales: (MTLBuffer, Int), biases: (MTLBuffer, Int),
        x: (MTLBuffer, Int), y: (MTLBuffer, Int), rows O: Int, inputs I: Int, config c: Config = Config()
    ) throws {
        precondition(I == 1024 || I == 2048, "GEMV4 takes 1024 or 2048 inputs")
        let p = try ctx.pipeline("gemv4", source: source, constants: ["I": I, "ROWS": c.rows, "NSG": c.simdgroups])
        enc.setComputePipelineState(p)
        enc.setBuffer(w.0, offset: w.1, index: 0)
        enc.setBuffer(scales.0, offset: scales.1, index: 1)
        enc.setBuffer(biases.0, offset: biases.1, index: 2)
        enc.setBuffer(x.0, offset: x.1, index: 3)
        enc.setBuffer(y.0, offset: y.1, index: 4)
        var o = Int32(O)
        enc.setBytes(&o, length: 4, index: 5)
        let perGroup = c.rows * c.simdgroups
        enc.dispatchThreadgroups(
            MTLSize(width: (O + perGroup - 1) / perGroup, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * c.simdgroups, height: 1, depth: 1))
    }
}
