/// bfloat16 as raw bits, for building inputs and reading outputs on the CPU.
public enum BF16 {
    /// Round to nearest even.
    public static func bits(_ f: Float) -> UInt16 {
        let u = f.bitPattern
        let rounding = ((u >> 16) & 1) + 0x7FFF
        return UInt16(truncatingIfNeeded: (u &+ rounding) >> 16)
    }

    public static func float(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }

    /// The value `f` takes once stored as bfloat16.
    public static func round(_ f: Float) -> Float { float(bits(f)) }
}
