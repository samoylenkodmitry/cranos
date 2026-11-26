use alloc::vec::Vec;

use crate::gguf::GgmlType;
use crate::math::{f16_to_f32, f32_round};

#[derive(Clone, Copy, Debug)]
pub struct SimdCaps {
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avx_vnni: bool,
}

impl SimdCaps {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::{__cpuid, __cpuid_count};
            let max_leaf = __cpuid(0).eax;
            let leaf1 = __cpuid(1);
            let leaf7_0 = if max_leaf >= 7 {
                __cpuid_count(7, 0)
            } else {
                core::arch::x86_64::CpuidResult {
                    eax: 0,
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                }
            };
            let leaf7_1 = if max_leaf >= 7 && leaf7_0.eax >= 1 {
                __cpuid_count(7, 1)
            } else {
                core::arch::x86_64::CpuidResult {
                    eax: 0,
                    ebx: 0,
                    ecx: 0,
                    edx: 0,
                }
            };
            let has_osxsave = (leaf1.ecx & (1 << 27)) != 0;
            let xcr0_avx = if has_osxsave {
                let eax: u32;
                unsafe {
                    core::arch::asm!(
                        "xgetbv",
                        in("ecx") 0u32,
                        out("eax") eax,
                        out("edx") _,
                        options(nomem, nostack)
                    );
                }
                (eax & 0x6) == 0x6
            } else {
                false
            };
            Self {
                avx2: xcr0_avx && ((leaf7_0.ebx & (1 << 5)) != 0),
                fma: xcr0_avx && ((leaf1.ecx & (1 << 12)) != 0),
                f16c: xcr0_avx && ((leaf1.ecx & (1 << 29)) != 0),
                avx_vnni: xcr0_avx && ((leaf7_1.eax & (1 << 4)) != 0),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx2: false,
                fma: false,
                f16c: false,
                avx_vnni: false,
            }
        }
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct BlockQ8_0 {
    pub qs: [i8; 32],
    pub d: f32,
    pub sum_q: i32,
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub struct BlockQ8K {
    pub qs: [i8; 256],
    pub bsums: [i16; 16],
    pub d: f32,
}

/// Pre-quantized input activation vector ready for fast quantized matvec.
pub struct QuantInput {
    pub f32_len: usize,
    pub q8_0: Vec<BlockQ8_0>,
    pub q8_k: Vec<BlockQ8K>,
}

impl QuantInput {
    pub fn with_max_dim(max_dim: usize) -> Self {
        Self {
            f32_len: 0,
            q8_0: Vec::with_capacity((max_dim + 31) / 32),
            q8_k: Vec::with_capacity((max_dim + 255) / 256),
        }
    }

    pub fn prepare_q8_0(&mut self, x: &[f32]) {
        self.f32_len = x.len();
        let nb = x.len() / 32;
        self.q8_0.clear();
        for b in 0..nb {
            let src = &x[b * 32..b * 32 + 32];
            let mut amax = 0.0f32;
            for &v in src {
                let a = if v < 0.0 { -v } else { v };
                if a > amax {
                    amax = a;
                }
            }
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let mut qs = [0i8; 32];
            let mut sum_q = 0i32;
            for i in 0..32 {
                let q = f32_round(src[i] * id).clamp(-127.0, 127.0) as i8;
                qs[i] = q;
                sum_q += q as i32;
            }
            self.q8_0.push(BlockQ8_0 { d, qs, sum_q });
        }
    }

    pub fn prepare_q8_k(&mut self, x: &[f32]) {
        self.f32_len = x.len();
        let nb = x.len() / 256;
        self.q8_k.clear();
        for b in 0..nb {
            let src = &x[b * 256..b * 256 + 256];
            let mut max_v = 0.0f32;
            let mut amax = 0.0f32;
            for &v in src {
                let a = if v < 0.0 { -v } else { v };
                if a > amax {
                    amax = a;
                    max_v = v;
                }
            }
            if amax == 0.0 {
                self.q8_k.push(BlockQ8K {
                    d: 0.0,
                    qs: [0; 256],
                    bsums: [0; 16],
                });
                continue;
            }
            let iscale = -127.0 / max_v;
            let d = 1.0 / iscale;
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..16 {
                let mut s = 0i32;
                let base = j * 16;
                for l in 0..16 {
                    let q = f32_round(src[base + l] * iscale).clamp(-127.0, 127.0) as i8;
                    qs[base + l] = q;
                    s += q as i32;
                }
                bsums[j] = s as i16;
            }
            self.q8_k.push(BlockQ8K { d, qs, bsums });
        }
    }
}

#[inline]
fn read_f16_le(bytes: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantizes a single row of `cols` elements from `row_bytes` into `out`.
pub fn dequantize_row(ty: GgmlType, row_bytes: &[u8], out: &mut [f32]) {
    let cols = out.len();
    match ty {
        GgmlType::F32 => {
            for i in 0..cols {
                let o = i * 4;
                out[i] = f32::from_le_bytes([
                    row_bytes[o],
                    row_bytes[o + 1],
                    row_bytes[o + 2],
                    row_bytes[o + 3],
                ]);
            }
        }
        GgmlType::F16 => {
            for i in 0..cols {
                out[i] = read_f16_le(row_bytes, i * 2);
            }
        }
        GgmlType::Bf16 => {
            for i in 0..cols {
                let hi = u16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]) as u32;
                out[i] = f32::from_bits(hi << 16);
            }
        }
        GgmlType::Q8_0 => {
            let nb = cols / 32;
            for b in 0..nb {
                let blk = &row_bytes[b * 34..b * 34 + 34];
                let d = read_f16_le(blk, 0);
                let dst = &mut out[b * 32..b * 32 + 32];
                for i in 0..32 {
                    dst[i] = d * (blk[2 + i] as i8 as f32);
                }
            }
        }
        GgmlType::Q4_0 => {
            let nb = cols / 32;
            for b in 0..nb {
                let blk = &row_bytes[b * 18..b * 18 + 18];
                let d = read_f16_le(blk, 0);
                let dst = &mut out[b * 32..b * 32 + 32];
                for i in 0..16 {
                    let byte = blk[2 + i];
                    dst[i] = d * ((byte & 0xF) as i32 - 8) as f32;
                    dst[i + 16] = d * ((byte >> 4) as i32 - 8) as f32;
                }
            }
        }
        GgmlType::Q4K => {
            let nb = cols / 256;
            for b in 0..nb {
                let blk = &row_bytes[b * 144..b * 144 + 144];
                let d = read_f16_le(blk, 0);
                let dmin = read_f16_le(blk, 2);
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let dst = &mut out[b * 256..b * 256 + 256];
                for j in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(2 * j, scales);
                    let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
                    let d1 = d * (sc1 as f32);
                    let min1 = dmin * (m1 as f32);
                    let d2 = d * (sc2 as f32);
                    let min2 = dmin * (m2 as f32);
                    let q_slice = &qs[j * 32..j * 32 + 32];
                    let base = j * 64;
                    for l in 0..32 {
                        dst[base + l] = d1 * (q_slice[l] & 0xF) as f32 - min1;
                        dst[base + 32 + l] = d2 * (q_slice[l] >> 4) as f32 - min2;
                    }
                }
            }
        }
        GgmlType::Q5K => {
            let nb = cols / 256;
            for b in 0..nb {
                let blk = &row_bytes[b * 176..b * 176 + 176];
                let d = read_f16_le(blk, 0);
                let dmin = read_f16_le(blk, 2);
                let scales = &blk[4..16];
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                let dst = &mut out[b * 256..b * 256 + 256];
                let mut u1 = 1u8;
                let mut u2 = 2u8;
                for j in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(2 * j, scales);
                    let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
                    let d1 = d * (sc1 as f32);
                    let min1 = dmin * (m1 as f32);
                    let d2 = d * (sc2 as f32);
                    let min2 = dmin * (m2 as f32);
                    let q_slice = &qs[j * 32..j * 32 + 32];
                    let base = j * 64;
                    for l in 0..32 {
                        let h1 = if (qh[l] & u1) != 0 { 16 } else { 0 };
                        let h2 = if (qh[l] & u2) != 0 { 16 } else { 0 };
                        dst[base + l] = d1 * ((q_slice[l] & 0xF) + h1) as f32 - min1;
                        dst[base + 32 + l] = d2 * ((q_slice[l] >> 4) + h2) as f32 - min2;
                    }
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GgmlType::Q6K => {
            let nb = cols / 256;
            for b in 0..nb {
                let blk = &row_bytes[b * 210..b * 210 + 210];
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let sc = &blk[192..208];
                let d = read_f16_le(blk, 208);
                let dst = &mut out[b * 256..b * 256 + 256];

                for half in 0..2 {
                    let ql_h = &ql[half * 64..half * 64 + 64];
                    let qh_h = &qh[half * 32..half * 32 + 32];
                    let sc_h = &sc[half * 8..half * 8 + 8];
                    let base = half * 128;
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql_h[l] & 0xF) | ((qh_h[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) as i32 - 32;
                        dst[base + l] = d * (sc_h[is] as i8 as f32) * (q1 as f32);
                        dst[base + l + 32] = d * (sc_h[is + 2] as i8 as f32) * (q2 as f32);
                        dst[base + l + 64] = d * (sc_h[is + 4] as i8 as f32) * (q3 as f32);
                        dst[base + l + 96] = d * (sc_h[is + 6] as i8 as f32) * (q4 as f32);
                    }
                }
            }
        }
        _ => {
            out.fill(0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar dot-product kernels (reference & fallback)
// ---------------------------------------------------------------------------

fn dot_q8_0_q8_0_scalar(w_row: &[u8], x_q: &[BlockQ8_0]) -> f32 {
    let mut sum = 0.0f32;
    for (b, xb) in x_q.iter().enumerate() {
        let blk = &w_row[b * 34..b * 34 + 34];
        let dw = read_f16_le(blk, 0);
        let mut isum = 0i32;
        for i in 0..32 {
            isum += (blk[2 + i] as i8 as i32) * (xb.qs[i] as i32);
        }
        sum += (dw * xb.d) * (isum as f32);
    }
    sum
}

fn dot_q4_0_q8_0_scalar(w_row: &[u8], x_q: &[BlockQ8_0]) -> f32 {
    let mut sum = 0.0f32;
    for (b, xb) in x_q.iter().enumerate() {
        let blk = &w_row[b * 18..b * 18 + 18];
        let dw = read_f16_le(blk, 0);
        let mut isum = 0i32;
        for i in 0..16 {
            let byte = blk[2 + i];
            let lo = (byte & 0xF) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            isum += lo * (xb.qs[i] as i32) + hi * (xb.qs[i + 16] as i32);
        }
        sum += (dw * xb.d) * (isum as f32);
    }
    sum
}

fn dot_q4_k_q8_k_scalar(w_row: &[u8], x_q: &[BlockQ8K]) -> f32 {
    let mut sum = 0.0f32;
    for (b, xb) in x_q.iter().enumerate() {
        let blk = &w_row[b * 144..b * 144 + 144];
        let d = read_f16_le(blk, 0) * xb.d;
        let dmin = read_f16_le(blk, 2) * xb.d;
        let scales = &blk[4..16];
        let qs = &blk[16..144];

        let mut isum = 0i32;
        let mut min_sum = 0i32;
        for j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(2 * j, scales);
            let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
            min_sum += (m1 as i32) * (xb.bsums[4 * j] as i32 + xb.bsums[4 * j + 1] as i32);
            min_sum += (m2 as i32) * (xb.bsums[4 * j + 2] as i32 + xb.bsums[4 * j + 3] as i32);

            let q_slice = &qs[j * 32..j * 32 + 32];
            let x1 = &xb.qs[j * 64..j * 64 + 32];
            let x2 = &xb.qs[j * 64 + 32..j * 64 + 64];
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            for l in 0..32 {
                s1 += (q_slice[l] & 0xF) as i32 * (x1[l] as i32);
                s2 += (q_slice[l] >> 4) as i32 * (x2[l] as i32);
            }
            isum += (sc1 as i32) * s1 + (sc2 as i32) * s2;
        }
        sum += d * (isum as f32) - dmin * (min_sum as f32);
    }
    sum
}

fn dot_q5_k_q8_k_scalar(w_row: &[u8], x_q: &[BlockQ8K]) -> f32 {
    let mut sum = 0.0f32;
    for (b, xb) in x_q.iter().enumerate() {
        let blk = &w_row[b * 176..b * 176 + 176];
        let d = read_f16_le(blk, 0) * xb.d;
        let dmin = read_f16_le(blk, 2) * xb.d;
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];

        let mut isum = 0i32;
        let mut min_sum = 0i32;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(2 * j, scales);
            let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
            min_sum += (m1 as i32) * (xb.bsums[4 * j] as i32 + xb.bsums[4 * j + 1] as i32);
            min_sum += (m2 as i32) * (xb.bsums[4 * j + 2] as i32 + xb.bsums[4 * j + 3] as i32);

            let q_slice = &qs[j * 32..j * 32 + 32];
            let x1 = &xb.qs[j * 64..j * 64 + 32];
            let x2 = &xb.qs[j * 64 + 32..j * 64 + 64];
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            for l in 0..32 {
                let h1 = if (qh[l] & u1) != 0 { 16 } else { 0 };
                let h2 = if (qh[l] & u2) != 0 { 16 } else { 0 };
                s1 += ((q_slice[l] & 0xF) + h1) as i32 * (x1[l] as i32);
                s2 += ((q_slice[l] >> 4) + h2) as i32 * (x2[l] as i32);
            }
            isum += (sc1 as i32) * s1 + (sc2 as i32) * s2;
            u1 <<= 2;
            u2 <<= 2;
        }
        sum += d * (isum as f32) - dmin * (min_sum as f32);
    }
    sum
}

fn dot_q6_k_q8_k_scalar(w_row: &[u8], x_q: &[BlockQ8K]) -> f32 {
    let mut sum = 0.0f32;
    for (b, xb) in x_q.iter().enumerate() {
        let blk = &w_row[b * 210..b * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d = read_f16_le(blk, 208) * xb.d;

        let mut isum = 0i32;
        for half in 0..2 {
            let ql_h = &ql[half * 64..half * 64 + 64];
            let qh_h = &qh[half * 32..half * 32 + 32];
            let sc_h = &sc[half * 8..half * 8 + 8];
            let base = half * 128;
            for is in 0..2 {
                let mut s0 = 0i32;
                let mut s2 = 0i32;
                let mut s4 = 0i32;
                let mut s6 = 0i32;
                for ll in 0..16 {
                    let l = is * 16 + ll;
                    let q1 = ((ql_h[l] & 0xF) | ((qh_h[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql_h[l + 32] & 0xF) | (((qh_h[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) as i32 - 32;
                    s0 += q1 * (xb.qs[base + l] as i32);
                    s2 += q2 * (xb.qs[base + l + 32] as i32);
                    s4 += q3 * (xb.qs[base + l + 64] as i32);
                    s6 += q4 * (xb.qs[base + l + 96] as i32);
                }
                isum += (sc_h[is] as i8 as i32) * s0
                    + (sc_h[is + 2] as i8 as i32) * s2
                    + (sc_h[is + 4] as i8 as i32) * s4
                    + (sc_h[is + 6] as i8 as i32) * s6;
            }
        }
        sum += d * (isum as f32);
    }
    sum
}

fn dot_f32_scalar(w_row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (i, &xv) in x.iter().enumerate() {
        let o = i * 4;
        let w = f32::from_le_bytes([w_row[o], w_row[o + 1], w_row[o + 2], w_row[o + 3]]);
        sum += w * xv;
    }
    sum
}

fn dot_f16_scalar(w_row: &[u8], x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (i, &xv) in x.iter().enumerate() {
        sum += read_f16_le(w_row, i * 2) * xv;
    }
    sum
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 + FMA + F16C + AVX-VNNI SIMD Kernels
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
mod simd_x86 {
    use core::arch::x86_64::*;

    use super::{BlockQ8K, BlockQ8_0, get_scale_min_k4};

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn f16_bits_to_f32(bits: u16) -> f32 {
        let v = _mm_cvtsi32_si128(bits as i32);
        let f = _mm_cvtph_ps(v);
        _mm_cvtss_f32(f)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_epi32(v: __m256i) -> i32 {
        let hi = _mm256_extracti128_si256::<1>(v);
        let lo = _mm256_castsi256_si128(v);
        let sum128 = _mm_add_epi32(lo, hi);
        let hi64 = _mm_unpackhi_epi64(sum128, sum128);
        let sum64 = _mm_add_epi32(sum128, hi64);
        let hi32 = _mm_shuffle_epi32::<0x55>(sum64);
        let sum32 = _mm_add_epi32(sum64, hi32);
        _mm_cvtsi128_si32(sum32)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let sh = _mm_movehl_ps(s, s);
        let s2 = _mm_add_ps(s, sh);
        let sh2 = _mm_shuffle_ps::<1>(s2, s2);
        _mm_cvtss_f32(_mm_add_ss(s2, sh2))
    }

    /// 32x u8 * i8 -> 8x i32 using AVX2 (vpmaddubsw + vpmaddwd).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_u8_i8_avx2(u: __m256i, s: __m256i, ones: __m256i) -> __m256i {
        let p16 = _mm256_maddubs_epi16(u, s);
        _mm256_madd_epi16(p16, ones)
    }

    /// 32x u8 * i8 -> accumulate into 8x i32 using AVX-VNNI VEX `vpdpbusd`
    /// (single instruction on Alder Lake i5-12400, zero i16 saturation!).
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn dpbusd_vnni(mut acc: __m256i, u: __m256i, s: __m256i) -> __m256i {
        unsafe {
            core::arch::asm!(
                "{{vex}} vpdpbusd {acc}, {u}, {s}",
                acc = inout(ymm_reg) acc,
                u = in(ymm_reg) u,
                s = in(ymm_reg) s,
                options(pure, nomem, nostack)
            );
        }
        acc
    }

    /// 4-row unrolled Q8_0 x Q8_0 matvec with AVX-VNNI (`vpdpbusd`) or AVX2.
    /// Computes 4 dot products simultaneously while reusing every activation register 4x.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn matvec4_q8_0(
        w0: *const u8,
        w1: *const u8,
        w2: *const u8,
        w3: *const u8,
        x_q: &[BlockQ8_0],
        use_vnni: bool,
    ) -> [f32; 4] {
        let ones = _mm256_set1_epi16(1);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let nb = x_q.len();
        for b in 0..nb {
            let off = b * 34;
            if b + 4 < nb {
                let pf = (b + 4) * 34;
                _mm_prefetch::<_MM_HINT_T0>(w0.add(pf) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(w1.add(pf) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(w2.add(pf) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(w3.add(pf) as *const i8);
            }

            let xb = x_q.get_unchecked(b);
            let vx = _mm256_load_si256(xb.qs.as_ptr() as *const __m256i);
            let dx = xb.d;

            let p0 = w0.add(off);
            let p1 = w1.add(off);
            let p2 = w2.add(off);
            let p3 = w3.add(off);

            let d0 = f16_bits_to_f32((p0 as *const u16).read_unaligned()) * dx;
            let d1 = f16_bits_to_f32((p1 as *const u16).read_unaligned()) * dx;
            let d2 = f16_bits_to_f32((p2 as *const u16).read_unaligned()) * dx;
            let d3 = f16_bits_to_f32((p3 as *const u16).read_unaligned()) * dx;

            let q0 = _mm256_loadu_si256(p0.add(2) as *const __m256i);
            let q1 = _mm256_loadu_si256(p1.add(2) as *const __m256i);
            let q2 = _mm256_loadu_si256(p2.add(2) as *const __m256i);
            let q3 = _mm256_loadu_si256(p3.add(2) as *const __m256i);

            // Convert signed i8*i8 to unsigned |q| * sign(vx, q). Because vx is
            // quantized to [-127, 127], sign(vx, q) never overflows.
            let a0 = _mm256_sign_epi8(q0, q0);
            let s0 = _mm256_sign_epi8(vx, q0);
            let a1 = _mm256_sign_epi8(q1, q1);
            let s1 = _mm256_sign_epi8(vx, q1);
            let a2 = _mm256_sign_epi8(q2, q2);
            let s2 = _mm256_sign_epi8(vx, q2);
            let a3 = _mm256_sign_epi8(q3, q3);
            let s3 = _mm256_sign_epi8(vx, q3);

            let (i0, i1, i2, i3) = if use_vnni {
                let z = _mm256_setzero_si256();
                (
                    dpbusd_vnni(z, a0, s0),
                    dpbusd_vnni(z, a1, s1),
                    dpbusd_vnni(z, a2, s2),
                    dpbusd_vnni(z, a3, s3),
                )
            } else {
                (
                    dot_u8_i8_avx2(a0, s0, ones),
                    dot_u8_i8_avx2(a1, s1, ones),
                    dot_u8_i8_avx2(a2, s2, ones),
                    dot_u8_i8_avx2(a3, s3, ones),
                )
            };

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(i0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(i1), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d2), _mm256_cvtepi32_ps(i2), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d3), _mm256_cvtepi32_ps(i3), acc3);
        }

        [hsum_ps(acc0), hsum_ps(acc1), hsum_ps(acc2), hsum_ps(acc3)]
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q8_0(w: *const u8, x_q: &[BlockQ8_0], use_vnni: bool) -> f32 {
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        for (b, xb) in x_q.iter().enumerate() {
            let p = w.add(b * 34);
            let d = f16_bits_to_f32((p as *const u16).read_unaligned()) * xb.d;
            let q = _mm256_loadu_si256(p.add(2) as *const __m256i);
            let vx = _mm256_load_si256(xb.qs.as_ptr() as *const __m256i);
            let a = _mm256_sign_epi8(q, q);
            let s = _mm256_sign_epi8(vx, q);
            let prod = if use_vnni {
                dpbusd_vnni(_mm256_setzero_si256(), a, s)
            } else {
                dot_u8_i8_avx2(a, s, ones)
            };
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(prod), acc);
        }
        hsum_ps(acc)
    }

    /// Ultra-fast Q4_K x Q8_K single-row kernel using AVX-VNNI (`vpdpbusd`) or AVX2.
    /// Each 4-bit nibble is unsigned [0, 15], directly feeding `vpdpbusd` with zero sign fixup!
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q4_k(w: *const u8, x_q: &[BlockQ8K], use_vnni: bool) -> f32 {
        let mask_low = _mm256_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let mut total = 0.0f32;
        let nb = x_q.len();

        for b in 0..nb {
            let blk = w.add(b * 144);
            if b + 2 < nb {
                _mm_prefetch::<_MM_HINT_T0>(w.add((b + 2) * 144) as *const i8);
            }
            let xb = x_q.get_unchecked(b);
            let d = f16_bits_to_f32((blk as *const u16).read_unaligned()) * xb.d;
            let dmin = f16_bits_to_f32((blk.add(2) as *const u16).read_unaligned()) * xb.d;
            let scales = core::slice::from_raw_parts(blk.add(4), 12);
            let qs_ptr = blk.add(16);
            let x_ptr = xb.qs.as_ptr() as *const __m256i;

            let mut acc_i32 = _mm256_setzero_si256();
            let mut min_sum = 0i32;

            for j in 0..4usize {
                let (sc1, m1) = get_scale_min_k4(2 * j, scales);
                let (sc2, m2) = get_scale_min_k4(2 * j + 1, scales);
                min_sum += (m1 as i32) * (xb.bsums[4 * j] as i32 + xb.bsums[4 * j + 1] as i32);
                min_sum += (m2 as i32) * (xb.bsums[4 * j + 2] as i32 + xb.bsums[4 * j + 3] as i32);

                let qbytes = _mm256_loadu_si256(qs_ptr.add(j * 32) as *const __m256i);
                let q_lo = _mm256_and_si256(qbytes, mask_low);
                let q_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qbytes), mask_low);

                let x_lo = _mm256_load_si256(x_ptr.add(2 * j));
                let x_hi = _mm256_load_si256(x_ptr.add(2 * j + 1));

                let (p_lo, p_hi) = if use_vnni {
                    let z = _mm256_setzero_si256();
                    (dpbusd_vnni(z, q_lo, x_lo), dpbusd_vnni(z, q_hi, x_hi))
                } else {
                    (
                        dot_u8_i8_avx2(q_lo, x_lo, ones),
                        dot_u8_i8_avx2(q_hi, x_hi, ones),
                    )
                };

                let s_lo = _mm256_mullo_epi32(p_lo, _mm256_set1_epi32(sc1 as i32));
                let s_hi = _mm256_mullo_epi32(p_hi, _mm256_set1_epi32(sc2 as i32));
                acc_i32 = _mm256_add_epi32(acc_i32, _mm256_add_epi32(s_lo, s_hi));
            }

            let isum = hsum_epi32(acc_i32);
            total += d * (isum as f32) - dmin * (min_sum as f32);
        }
        total
    }

    /// 4-row unrolled Q4_K x Q8_K matvec with zero YMM register spilling:
    /// processes one 64-weight sub-chunk (x_lo, x_hi = 2 YMM regs) across all 4 rows
    /// while accumulating into 4 float YMM vectors (only 1 horizontal sum per row at end).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn matvec4_q4_k(
        w0: *const u8,
        w1: *const u8,
        w2: *const u8,
        w3: *const u8,
        x_q: &[BlockQ8K],
        use_vnni: bool,
    ) -> [f32; 4] {
        let mask_low = _mm256_set1_epi8(0x0F);
        let ones = _mm256_set1_epi16(1);
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut min_tot = [0.0f32; 4];
        let ptrs = [w0, w1, w2, w3];
        let nb = x_q.len();

        for b in 0..nb {
            let boff = b * 144;
            if b + 2 < nb {
                let pf = (b + 2) * 144;
                _mm_prefetch::<_MM_HINT_T0>(w0.add(pf) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(w1.add(pf) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(w2.add(pf) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(w3.add(pf) as *const i8);
            }
            let xb = x_q.get_unchecked(b);
            let x_ptr = xb.qs.as_ptr() as *const __m256i;

            let bpair = [
                xb.bsums[0] as i32 + xb.bsums[1] as i32,
                xb.bsums[2] as i32 + xb.bsums[3] as i32,
                xb.bsums[4] as i32 + xb.bsums[5] as i32,
                xb.bsums[6] as i32 + xb.bsums[7] as i32,
                xb.bsums[8] as i32 + xb.bsums[9] as i32,
                xb.bsums[10] as i32 + xb.bsums[11] as i32,
                xb.bsums[12] as i32 + xb.bsums[13] as i32,
                xb.bsums[14] as i32 + xb.bsums[15] as i32,
            ];

            let mut sc = [[0i32; 8]; 4];
            let mut d_vec = [0.0f32; 4];
            for r in 0..4usize {
                let blk = ptrs[r].add(boff);
                d_vec[r] = f16_bits_to_f32((blk as *const u16).read_unaligned()) * xb.d;
                let dmin = f16_bits_to_f32((blk.add(2) as *const u16).read_unaligned()) * xb.d;
                let scales = core::slice::from_raw_parts(blk.add(4), 12);
                let mut msum = 0i32;
                for j in 0..8usize {
                    let (s, m) = get_scale_min_k4(j, scales);
                    sc[r][j] = s as i32;
                    msum += (m as i32) * bpair[j];
                }
                min_tot[r] -= dmin * (msum as f32);
            }

            let mut iacc0 = _mm256_setzero_si256();
            let mut iacc1 = _mm256_setzero_si256();
            let mut iacc2 = _mm256_setzero_si256();
            let mut iacc3 = _mm256_setzero_si256();

            for j in 0..4usize {
                let x_lo = _mm256_load_si256(x_ptr.add(2 * j));
                let x_hi = _mm256_load_si256(x_ptr.add(2 * j + 1));
                let qoff = boff + 16 + j * 32;

                macro_rules! step_row {
                    ($r:literal, $iacc:ident) => {
                        let qb = _mm256_loadu_si256(ptrs[$r].add(qoff) as *const __m256i);
                        let q_lo = _mm256_and_si256(qb, mask_low);
                        let q_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(qb), mask_low);
                        let (p_lo, p_hi) = if use_vnni {
                            let z = _mm256_setzero_si256();
                            (dpbusd_vnni(z, q_lo, x_lo), dpbusd_vnni(z, q_hi, x_hi))
                        } else {
                            (
                                dot_u8_i8_avx2(q_lo, x_lo, ones),
                                dot_u8_i8_avx2(q_hi, x_hi, ones),
                            )
                        };
                        let s_lo = _mm256_mullo_epi32(p_lo, _mm256_set1_epi32(sc[$r][2 * j]));
                        let s_hi = _mm256_mullo_epi32(p_hi, _mm256_set1_epi32(sc[$r][2 * j + 1]));
                        $iacc = _mm256_add_epi32($iacc, _mm256_add_epi32(s_lo, s_hi));
                    };
                }
                step_row!(0, iacc0);
                step_row!(1, iacc1);
                step_row!(2, iacc2);
                step_row!(3, iacc3);
            }

            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(d_vec[0]), _mm256_cvtepi32_ps(iacc0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(d_vec[1]), _mm256_cvtepi32_ps(iacc1), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(d_vec[2]), _mm256_cvtepi32_ps(iacc2), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(d_vec[3]), _mm256_cvtepi32_ps(iacc3), acc3);
        }

        [
            hsum_ps(acc0) + min_tot[0],
            hsum_ps(acc1) + min_tot[1],
            hsum_ps(acc2) + min_tot[2],
            hsum_ps(acc3) + min_tot[3],
        ]
    }

    /// AVX2 + AVX-VNNI Q6_K x Q8_K kernel.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q6_k(w: *const u8, x_q: &[BlockQ8K], use_vnni: bool) -> f32 {
        let mask_lo4 = _mm256_set1_epi8(0x0F);
        let mask_hi2 = _mm256_set1_epi8(0x03);
        let ones = _mm256_set1_epi16(1);
        let mut total = 0.0f32;
        let nb = x_q.len();

        for b in 0..nb {
            let blk = w.add(b * 210);
            if b + 2 < nb {
                _mm_prefetch::<_MM_HINT_T0>(w.add((b + 2) * 210) as *const i8);
            }
            let xb = x_q.get_unchecked(b);
            let ql_base = blk;
            let qh_base = blk.add(128);
            let sc_base = blk.add(192) as *const i8;
            let d = f16_bits_to_f32((blk.add(208) as *const u16).read_unaligned()) * xb.d;
            let x_ptr = xb.qs.as_ptr() as *const __m256i;

            let mut block_sum = 0i32;
            for half in 0..2usize {
                let ql0 = _mm256_loadu_si256(ql_base.add(half * 64) as *const __m256i);
                let ql1 = _mm256_loadu_si256(ql_base.add(half * 64 + 32) as *const __m256i);
                let qh = _mm256_loadu_si256(qh_base.add(half * 32) as *const __m256i);

                // Extract 4 unsigned 6-bit [0..63] vectors of 32 bytes each.
                let q1 = _mm256_or_si256(
                    _mm256_and_si256(ql0, mask_lo4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(qh, mask_hi2)),
                );
                let q2 = _mm256_or_si256(
                    _mm256_and_si256(ql1, mask_lo4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<2>(qh), mask_hi2)),
                );
                let q3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(ql0), mask_lo4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<4>(qh), mask_hi2)),
                );
                let q4 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(ql1), mask_lo4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<6>(qh), mask_hi2)),
                );

                let qs_arr = [q1, q2, q3, q4];
                let sc_off = half * 8;
                let x_off = half * 4;

                for g in 0..4usize {
                    let xv = _mm256_load_si256(x_ptr.add(x_off + g));
                    let p32 = if use_vnni {
                        dpbusd_vnni(_mm256_setzero_si256(), qs_arr[g], xv)
                    } else {
                        dot_u8_i8_avx2(qs_arr[g], xv, ones)
                    };
                    // Low 128-bit lane = first 16 elements (scale sc[sc_off + g*2 + 0])
                    // High 128-bit lane = next 16 elements (scale sc[sc_off + g*2 + 1])
                    let lo128 = _mm256_castsi256_si128(p32);
                    let hi128 = _mm256_extracti128_si256::<1>(p32);
                    let hsum128 = |v: __m128i| -> i32 {
                        let a = _mm_add_epi32(v, _mm_unpackhi_epi64(v, v));
                        let b = _mm_add_epi32(a, _mm_shuffle_epi32::<0x55>(a));
                        _mm_cvtsi128_si32(b)
                    };
                    let s_lo = *sc_base.add(sc_off + g * 2) as i32;
                    let s_hi = *sc_base.add(sc_off + g * 2 + 1) as i32;
                    let b_idx = half * 8 + g * 2;
                    let sum_lo = hsum128(lo128) - 32 * (xb.bsums[b_idx] as i32);
                    let sum_hi = hsum128(hi128) - 32 * (xb.bsums[b_idx + 1] as i32);
                    block_sum += s_lo * sum_lo + s_hi * sum_hi;
                }
            }
            total += d * (block_sum as f32);
        }
        total
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_f16_f32(w: *const u16, x: *const f32, len: usize) -> f32 {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 16 <= len {
            let h0 = _mm_loadu_si128(w.add(i) as *const __m128i);
            let h1 = _mm_loadu_si128(w.add(i + 8) as *const __m128i);
            let f0 = _mm256_cvtph_ps(h0);
            let f1 = _mm256_cvtph_ps(h1);
            let x0 = _mm256_loadu_ps(x.add(i));
            let x1 = _mm256_loadu_ps(x.add(i + 8));
            acc0 = _mm256_fmadd_ps(f0, x0, acc0);
            acc1 = _mm256_fmadd_ps(f1, x1, acc1);
            i += 16;
        }
        let mut sum = hsum_ps(_mm256_add_ps(acc0, acc1));
        while i < len {
            sum += f16_bits_to_f32(*w.add(i)) * *x.add(i);
            i += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn axpy_f16_f32(dst: *mut f32, src_f16: *const u16, scale: f32, len: usize) {
        let vs = _mm256_set1_ps(scale);
        let mut i = 0usize;
        while i + 16 <= len {
            let h0 = _mm_loadu_si128(src_f16.add(i) as *const __m128i);
            let h1 = _mm_loadu_si128(src_f16.add(i + 8) as *const __m128i);
            let f0 = _mm256_cvtph_ps(h0);
            let f1 = _mm256_cvtph_ps(h1);
            let d0 = _mm256_loadu_ps(dst.add(i));
            let d1 = _mm256_loadu_ps(dst.add(i + 8));
            _mm256_storeu_ps(dst.add(i), _mm256_fmadd_ps(f0, vs, d0));
            _mm256_storeu_ps(dst.add(i + 8), _mm256_fmadd_ps(f1, vs, d1));
            i += 16;
        }
        while i < len {
            *dst.add(i) += f16_bits_to_f32(*src_f16.add(i)) * scale;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn convert_f32_to_f16_slice(src: *const f32, dst: *mut u16, len: usize) {
        let mut i = 0usize;
        while i + 8 <= len {
            let v = _mm256_loadu_ps(src.add(i));
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            _mm_storeu_si128(dst.add(i) as *mut __m128i, h);
            i += 8;
        }
        while i < len {
            *dst.add(i) = super::super::math::f32_to_f16(*src.add(i));
            i += 1;
        }
    }
}

/// Computes `out[r - row_start] = dot(W[r], x)` for rows `row_start..row_end`.
pub fn matvec_row_range(
    ty: GgmlType,
    weights: &[u8],
    cols: usize,
    row_start: usize,
    row_end: usize,
    x_f32: &[f32],
    x_q: &QuantInput,
    out: &mut [f32],
    caps: SimdCaps,
) {
    let row_stride = ty.row_bytes(cols);
    let mut r = row_start;

    #[cfg(target_arch = "x86_64")]
    if caps.avx2 && caps.fma && caps.f16c {
        let use_vnni = caps.avx_vnni;
        match ty {
            GgmlType::Q8_0 => {
                while r + 4 <= row_end {
                    let base = weights.as_ptr();
                    let res = unsafe {
                        simd_x86::matvec4_q8_0(
                            base.add(r * row_stride),
                            base.add((r + 1) * row_stride),
                            base.add((r + 2) * row_stride),
                            base.add((r + 3) * row_stride),
                            &x_q.q8_0,
                            use_vnni,
                        )
                    };
                    let oi = r - row_start;
                    out[oi] = res[0];
                    out[oi + 1] = res[1];
                    out[oi + 2] = res[2];
                    out[oi + 3] = res[3];
                    r += 4;
                }
                while r < row_end {
                    let w = unsafe { weights.as_ptr().add(r * row_stride) };
                    out[r - row_start] = unsafe { simd_x86::dot_q8_0(w, &x_q.q8_0, use_vnni) };
                    r += 1;
                }
                return;
            }
            GgmlType::Q4K => {
                while r + 4 <= row_end {
                    let base = weights.as_ptr();
                    let res = unsafe {
                        simd_x86::matvec4_q4_k(
                            base.add(r * row_stride),
                            base.add((r + 1) * row_stride),
                            base.add((r + 2) * row_stride),
                            base.add((r + 3) * row_stride),
                            &x_q.q8_k,
                            use_vnni,
                        )
                    };
                    let oi = r - row_start;
                    out[oi] = res[0];
                    out[oi + 1] = res[1];
                    out[oi + 2] = res[2];
                    out[oi + 3] = res[3];
                    r += 4;
                }
                while r < row_end {
                    let w = unsafe { weights.as_ptr().add(r * row_stride) };
                    out[r - row_start] = unsafe { simd_x86::dot_q4_k(w, &x_q.q8_k, use_vnni) };
                    r += 1;
                }
                return;
            }
            GgmlType::Q6K => {
                while r < row_end {
                    let w = unsafe { weights.as_ptr().add(r * row_stride) };
                    out[r - row_start] = unsafe { simd_x86::dot_q6_k(w, &x_q.q8_k, use_vnni) };
                    r += 1;
                }
                return;
            }
            GgmlType::F16 => {
                while r < row_end {
                    let w = unsafe { weights.as_ptr().add(r * row_stride) as *const u16 };
                    out[r - row_start] = unsafe { simd_x86::dot_f16_f32(w, x_f32.as_ptr(), cols) };
                    r += 1;
                }
                return;
            }
            _ => {}
        }
    }

    // Scalar fallback for all types
    while r < row_end {
        let row = &weights[r * row_stride..(r + 1) * row_stride];
        out[r - row_start] = match ty {
            GgmlType::Q8_0 => dot_q8_0_q8_0_scalar(row, &x_q.q8_0),
            GgmlType::Q4_0 => dot_q4_0_q8_0_scalar(row, &x_q.q8_0),
            GgmlType::Q4K => dot_q4_k_q8_k_scalar(row, &x_q.q8_k),
            GgmlType::Q5K => dot_q5_k_q8_k_scalar(row, &x_q.q8_k),
            GgmlType::Q6K => dot_q6_k_q8_k_scalar(row, &x_q.q8_k),
            GgmlType::F16 => dot_f16_scalar(row, x_f32),
            GgmlType::F32 => dot_f32_scalar(row, x_f32),
            _ => 0.0,
        };
        r += 1;
    }
}

/// Fast F16 <-> F32 attention helpers (AVX2+F16C accelerated when available).
pub fn dot_f16_vec(a_f16: &[u16], b_f32: &[f32], caps: SimdCaps) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if caps.avx2 && caps.fma && caps.f16c {
        return unsafe { simd_x86::dot_f16_f32(a_f16.as_ptr(), b_f32.as_ptr(), b_f32.len()) };
    }
    let mut s = 0.0f32;
    for i in 0..b_f32.len() {
        s += f16_to_f32(a_f16[i]) * b_f32[i];
    }
    s
}

pub fn axpy_f16_vec(dst: &mut [f32], src_f16: &[u16], scale: f32, caps: SimdCaps) {
    #[cfg(target_arch = "x86_64")]
    if caps.avx2 && caps.fma && caps.f16c {
        unsafe { simd_x86::axpy_f16_f32(dst.as_mut_ptr(), src_f16.as_ptr(), scale, dst.len()) };
        return;
    }
    for i in 0..dst.len() {
        dst[i] += f16_to_f32(src_f16[i]) * scale;
    }
}

pub fn f32_to_f16_slice(src: &[f32], dst: &mut [u16], caps: SimdCaps) {
    #[cfg(target_arch = "x86_64")]
    if caps.avx2 && caps.fma && caps.f16c {
        unsafe { simd_x86::convert_f32_to_f16_slice(src.as_ptr(), dst.as_mut_ptr(), src.len()) };
        return;
    }
    for i in 0..src.len() {
        dst[i] = crate::math::f32_to_f16(src[i]);
    }
}

pub fn needs_q8_k(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K)
}

pub fn needs_q8_0(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Q8_0 | GgmlType::Q4_0)
}
