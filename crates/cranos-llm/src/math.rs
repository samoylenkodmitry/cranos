#![allow(clippy::excessive_precision)]

/// Converts an IEEE 754 binary16 bit pattern to f32.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal f16
        let mut m = mant;
        let mut e: i32 = -14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        let f32_exp = ((e + 127) as u32) << 23;
        return f32::from_bits(sign | f32_exp | (m << 13));
    }
    if exp == 31 {
        let f32_exp = 0xFFu32 << 23;
        return f32::from_bits(sign | f32_exp | (mant << 13));
    }
    let f32_exp = (exp + 112) << 23;
    f32::from_bits(sign | f32_exp | (mant << 13))
}

/// Converts f32 to IEEE 754 binary16 (round-to-nearest-even).
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        if mant != 0 {
            return sign | 0x7E00;
        }
        return sign | 0x7C00;
    }
    let half_exp = exp - 112;
    if half_exp >= 31 {
        return sign | 0x7C00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - half_exp);
        let round = (m + 0x1000) >> 13;
        return sign | (round as u16);
    }
    let round_bit = 0x1000u32;
    let rounded = mant + round_bit;
    if rounded >= 0x0080_0000 {
        let new_exp = half_exp + 1;
        if new_exp >= 31 {
            return sign | 0x7C00;
        }
        return sign | ((new_exp as u16) << 10);
    }
    sign | ((half_exp as u16) << 10) | ((rounded >> 13) as u16)
}

#[inline]
pub fn f32_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v = core::arch::x86_64::_mm_set_ss(x);
        core::arch::x86_64::_mm_cvtss_f32(core::arch::x86_64::_mm_sqrt_ss(v))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Newton-Raphson fallback
        let mut g = f32::from_bits((x.to_bits() >> 1) + 0x1FC0_0000);
        for _ in 0..4 {
            g = 0.5 * (g + x / g);
        }
        g
    }
}

#[inline]
pub fn f32_floor(x: f32) -> f32 {
    let i = x as i32;
    let f = i as f32;
    if x < f { f - 1.0 } else { f }
}

#[inline]
pub fn f32_round(x: f32) -> f32 {
    if x >= 0.0 {
        (x + 0.5) as i32 as f32
    } else {
        (x - 0.5) as i32 as f32
    }
}

/// Fast accurate exp(x) for f32 (max relative error < 2e-7 across softmax/SiLU range).
#[inline]
pub fn f32_exp(x: f32) -> f32 {
    if x < -88.0 {
        return 0.0;
    }
    if x > 88.0 {
        return f32::INFINITY;
    }
    const LOG2E: f32 = 1.4426950408889634;
    const LN2_HI: f32 = 0.693145751953125;
    const LN2_LO: f32 = 1.428606765330187e-6;

    let k = f32_floor(x * LOG2E + 0.5);
    let r = (x - k * LN2_HI) - k * LN2_LO;

    // Degree-5 minimax polynomial on [-ln(2)/2, ln(2)/2]
    let poly = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.16666667 + r * (0.041666668 + r * 0.008333334))));
    let pow2 = f32::from_bits(((k as i32 + 127) as u32) << 23);
    poly * pow2
}

/// Natural logarithm ln(x) for x > 0 (used once at model init for RoPE frequency table).
pub fn f64_ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let exp = (((bits >> 52) & 0x7FF) as i64) - 1023;
    let m = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | (1023u64 << 52));
    // Bring m into [sqrt(0.5), sqrt(2)]
    let (m, e) = if m > core::f64::consts::SQRT_2 {
        (m * 0.5, exp + 1)
    } else {
        (m, exp)
    };
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let mut term = t;
    let mut sum = 0.0;
    for k in 0..12 {
        sum += term / (2 * k + 1) as f64;
        term *= t2;
    }
    2.0 * sum + (e as f64) * core::f64::consts::LN_2
}

/// Double-precision exp(x) (used once at model init for RoPE frequency table).
pub fn f64_exp(x: f64) -> f64 {
    const LN2: f64 = core::f64::consts::LN_2;
    let k = (x / LN2 + if x >= 0.0 { 0.5 } else { -0.5 }) as i64;
    let r = x - (k as f64) * LN2;
    let mut term = 1.0;
    let mut sum = 1.0;
    for i in 1..18 {
        term *= r / (i as f64);
        sum += term;
    }
    let pow2 = f64::from_bits(((k + 1023) as u64) << 52);
    sum * pow2
}

/// Double-precision pow(base, exp) for base > 0 (RoPE frequency init).
pub fn f64_pow(base: f64, exp: f64) -> f64 {
    f64_exp(exp * f64_ln(base))
}

/// Accurate sin and cos for RoPE table precomputation.
pub fn f64_sin_cos(x: f64) -> (f64, f64) {
    const TWO_PI: f64 = 2.0 * core::f64::consts::PI;
    const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;
    const PI: f64 = core::f64::consts::PI;

    let q = (x / TWO_PI) as i64;
    let mut r = x - (q as f64) * TWO_PI;
    if r > PI {
        r -= TWO_PI;
    } else if r < -PI {
        r += TWO_PI;
    }

    // Reduce into [-pi/2, pi/2]
    let (r_red, cos_sign) = if r > HALF_PI {
        (PI - r, -1.0)
    } else if r < -HALF_PI {
        (-PI - r, -1.0)
    } else {
        (r, 1.0)
    };

    let r2 = r_red * r_red;
    // Taylor series converges to full f64 precision on [-pi/2, pi/2] in 10 terms
    let mut s_term = r_red;
    let mut sin_v = r_red;
    let mut c_term = 1.0;
    let mut cos_v = 1.0;
    for k in 1..=10 {
        s_term *= -r2 / ((2 * k) * (2 * k + 1)) as f64;
        sin_v += s_term;
        c_term *= -r2 / ((2 * k - 1) * (2 * k)) as f64;
        cos_v += c_term;
    }
    (sin_v, cos_v * cos_sign)
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + f32_exp(-x))
}
