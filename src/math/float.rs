//! [`Float`]: the operations the generic math kernels are written with, for
//! every lane type (`f32`, `f64` and the SIMD registers).
//!
//! A kernel such as `acos` is written once, generic over `V: Float`, and
//! compiles to each backend's instructions; the scalar backend runs the same
//! code one element at a time. Branches become [`select`](Float::select)s:
//! every case is computed and each lane keeps its own.

use core::ops::Neg;

use super::{Abs, Sqrt};
use crate::ops::{Arith, Compare};

/// A float, or a SIMD register of floats.
///
/// Methods taking `self` are safe: a register value proves the CPU supports
/// its ISA (see `SimdVector`).
pub(crate) trait Float: Arith + Neg<Output = Self> + Compare + Abs + Sqrt {
    /// The element type: `f32` or `f64`.
    type Elem: Element;

    /// `c` in every lane; `self` is only the proof of the ISA.
    fn constant(self, c: Self::Elem) -> Self;

    /// `self · b + c`: fused (one rounding) on SIMD backends, two roundings
    /// on the scalar one, as `core` has no fused multiply-add.
    fn mul_add(self, b: Self, c: Self) -> Self;

    /// Lane-wise `if mask { a } else { b }`.
    fn select(mask: Self::Mask, a: Self, b: Self) -> Self;

    /// The bits of every lane, ANDed with `mask`.
    fn and_bits(self, mask: Bits<Self>) -> Self;

    /// `c − self · b`, **exactly** whenever the result is a float, as for
    /// the remainder of a division (`c = num`, `b = den`, `self ≈ num/den`):
    /// a fused multiply-add on SIMD backends, Dekker's exact product on the
    /// scalar one.
    fn residual(self, b: Self, c: Self) -> Self;

    /// Lanes whose sign bit is set: negative numbers, `-0.0` and `-NaN`.
    fn is_sign_negative(self) -> Self::Mask;
}

/// `f32` or `f64`.
pub(crate) trait Element: Copy {
    /// The bits: `u32` or `u64`.
    type Bits: Copy + core::ops::Not<Output = Self::Bits>;

    /// The sign bit.
    #[cfg(test)]
    const SIGN: Self::Bits;

    /// `x`, exactly: kernels write their small constants as `f32`.
    fn from_f32(x: f32) -> Self;

    /// `2^s + 1`, `s` half the significand width rounded up: multiplying by
    /// it splits a float into two halves whose products are exact (Dekker).
    const SPLIT: Self;
}

impl Element for f32 {
    type Bits = u32;
    #[cfg(test)]
    const SIGN: u32 = 1 << 31;
    const SPLIT: f32 = 4097.0; // 2^12 + 1

    #[inline(always)]
    fn from_f32(x: f32) -> f32 {
        x
    }
}

impl Element for f64 {
    type Bits = u64;
    #[cfg(test)]
    const SIGN: u64 = 1 << 63;
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1

    #[inline(always)]
    fn from_f32(x: f32) -> f64 {
        x.into()
    }
}

/// The bits of `V`'s elements.
pub(crate) type Bits<V> = <<V as Float>::Elem as Element>::Bits;

macro_rules! float_scalar {
    ($($T:ident: $Bits:ident),*) => {
        $(
            impl Float for $T {
                type Elem = $T;

                #[inline(always)]
                fn constant(self, c: $T) -> $T {
                    c
                }

                #[inline(always)]
                fn mul_add(self, b: $T, c: $T) -> $T {
                    self * b + c
                }

                #[inline(always)]
                fn select(mask: bool, a: $T, b: $T) -> $T {
                    if mask { a } else { b }
                }

                #[inline(always)]
                fn and_bits(self, mask: $Bits) -> $T {
                    $T::from_bits(self.to_bits() & mask)
                }

                /// Dekker: `self · b = p + e` exactly, from halves of each
                /// factor whose products are exact. `c − p` is exact when
                /// the residual is small (Sterbenz).
                #[inline(always)]
                fn residual(self, b: $T, c: $T) -> $T {
                    let p = self * b;
                    let (ah, al) = split(self);
                    let (bh, bl) = split(b);
                    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
                    (c - p) - e
                }

                #[inline(always)]
                fn is_sign_negative(self) -> bool {
                    $T::is_sign_negative(self)
                }
            }
        )*
    };
}

/// `v = hi + lo`, each with half of `v`'s significand bits.
#[inline(always)]
fn split<T: Element + Arith>(v: T) -> (T, T) {
    let t = v * T::SPLIT;
    let hi = t - (t - v);
    (hi, v - hi)
}

float_scalar!(f32: u32, f64: u64);

cfg_x86! {
    mod x86 {
        use core::arch::x86_64::{
            __m256i, __mmask8, __mmask16, _mm256_and_si256, _mm256_blendv_pd, _mm256_blendv_ps,
            _mm256_castpd_si256, _mm256_castps_si256, _mm256_castsi256_pd, _mm256_castsi256_ps,
            _mm256_cmpgt_epi32, _mm256_cmpgt_epi64, _mm256_fmadd_pd, _mm256_fmadd_ps,
            _mm256_fnmadd_pd, _mm256_fnmadd_ps, _mm256_set1_epi32, _mm256_set1_epi64x,
            _mm256_setzero_si256, _mm512_and_si512, _mm512_castpd_si512, _mm512_castps_si512,
            _mm512_castsi512_pd, _mm512_castsi512_ps, _mm512_cmplt_epi32_mask,
            _mm512_cmplt_epi64_mask, _mm512_fmadd_pd, _mm512_fmadd_ps, _mm512_fnmadd_pd,
            _mm512_fnmadd_ps, _mm512_mask_blend_pd, _mm512_mask_blend_ps, _mm512_set1_epi32,
            _mm512_set1_epi64, _mm512_setzero_si512,
        };

        use super::Float;
        use crate::simd::SimdVector;
        use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};

        /// `impl Float` for an x86 register. `$and` takes and returns the raw
        /// register, so it can cast through the integer one.
        macro_rules! float_x86 {
            (
                $V:ident: $T:ident, $Bits:ident, $Mask:ty;
                fmadd: $fmadd:ident,
                fnmadd: $fnmadd:ident,
                select: |$m:ident, $a:ident, $b:ident| $select:expr,
                and: |$v:ident, $bits:ident| $and:expr,
                sign: |$s:ident| $sign:expr $(,)?
            ) => {
                impl Float for $V {
                    type Elem = $T;

                    #[inline(always)]
                    fn constant(self, c: $T) -> Self {
                        // SAFETY: a register of this type exists, so its ISA
                        // is available.
                        unsafe { <$V as SimdVector>::splat(c) }
                    }

                    #[inline(always)]
                    fn mul_add(self, b: Self, c: Self) -> Self {
                        // SAFETY: a register of this type exists, so its ISA
                        // is available; AVX2 registers also need FMA, which
                        // their constructors require.
                        unsafe { $V::from_raw($fmadd(self.raw(), b.raw(), c.raw())) }
                    }

                    #[inline(always)]
                    fn select($m: $Mask, a: Self, b: Self) -> Self {
                        let ($a, $b) = (a.raw(), b.raw());
                        // SAFETY: a register of this type exists, so its ISA
                        // is available.
                        unsafe { $V::from_raw($select) }
                    }

                    #[inline(always)]
                    fn and_bits(self, $bits: $Bits) -> Self {
                        let $v = self.raw();
                        // SAFETY: a register of this type exists, so its ISA
                        // is available.
                        unsafe { $V::from_raw($and) }
                    }

                    #[inline(always)]
                    fn residual(self, b: Self, c: Self) -> Self {
                        // SAFETY: as for `mul_add`.
                        unsafe { $V::from_raw($fnmadd(self.raw(), b.raw(), c.raw())) }
                    }

                    #[inline(always)]
                    fn is_sign_negative(self) -> $Mask {
                        let $s = self.raw();
                        // SAFETY: a register of this type exists, so its ISA
                        // is available.
                        unsafe { $sign }
                    }
                }
            };
        }

        float_x86!(F32x8: f32, u32, __m256i;
            fmadd: _mm256_fmadd_ps,
            fnmadd: _mm256_fnmadd_ps,
            select: |m, a, b| _mm256_blendv_ps(b, a, _mm256_castsi256_ps(m)),
            and: |v, bits| _mm256_castsi256_ps(_mm256_and_si256(
                _mm256_castps_si256(v),
                _mm256_set1_epi32(bits as i32),
            )),
            sign: |v| _mm256_cmpgt_epi32(_mm256_setzero_si256(), _mm256_castps_si256(v)),
        );

        float_x86!(F64x4: f64, u64, __m256i;
            fmadd: _mm256_fmadd_pd,
            fnmadd: _mm256_fnmadd_pd,
            select: |m, a, b| _mm256_blendv_pd(b, a, _mm256_castsi256_pd(m)),
            and: |v, bits| _mm256_castsi256_pd(_mm256_and_si256(
                _mm256_castpd_si256(v),
                _mm256_set1_epi64x(bits as i64),
            )),
            sign: |v| _mm256_cmpgt_epi64(_mm256_setzero_si256(), _mm256_castpd_si256(v)),
        );

        float_x86!(F32x16: f32, u32, __mmask16;
            fmadd: _mm512_fmadd_ps,
            fnmadd: _mm512_fnmadd_ps,
            select: |m, a, b| _mm512_mask_blend_ps(m, b, a),
            and: |v, bits| _mm512_castsi512_ps(_mm512_and_si512(
                _mm512_castps_si512(v),
                _mm512_set1_epi32(bits as i32),
            )),
            sign: |v| _mm512_cmplt_epi32_mask(_mm512_castps_si512(v), _mm512_setzero_si512()),
        );

        float_x86!(F64x8: f64, u64, __mmask8;
            fmadd: _mm512_fmadd_pd,
            fnmadd: _mm512_fnmadd_pd,
            select: |m, a, b| _mm512_mask_blend_pd(m, b, a),
            and: |v, bits| _mm512_castsi512_pd(_mm512_and_si512(
                _mm512_castpd_si512(v),
                _mm512_set1_epi64(bits as i64),
            )),
            sign: |v| _mm512_cmplt_epi64_mask(_mm512_castpd_si512(v), _mm512_setzero_si512()),
        );
    }
}

cfg_neon! {
    mod neon {
        use core::arch::aarch64::{
            uint32x4_t, uint64x2_t, vandq_u32, vandq_u64, vbslq_f32, vbslq_f64, vcltzq_s32,
            vcltzq_s64, vdupq_n_u32, vdupq_n_u64, vfmaq_f32, vfmaq_f64, vfmsq_f32, vfmsq_f64,
            vreinterpretq_f32_u32, vreinterpretq_f64_u64, vreinterpretq_s32_f32,
            vreinterpretq_s64_f64, vreinterpretq_u32_f32, vreinterpretq_u64_f64,
        };

        use super::Float;
        use crate::simd::SimdVector;
        use crate::simd::neon::{f32x4::F32x4, f64x2::F64x2};

        macro_rules! float_neon {
            ($($V:ident: $T:ident, $Bits:ident, $Mask:ty, $fma:ident, $fms:ident, $bsl:ident,
                $and:ident, $dup:ident, $to_bits:ident, $from_bits:ident,
                $cltz:ident, $to_signed:ident);* $(;)?) => {
                $(
                    impl Float for $V {
                        type Elem = $T;

                        #[inline(always)]
                        fn constant(self, c: $T) -> Self {
                            // SAFETY: NEON is enabled at compile time.
                            unsafe { <$V as SimdVector>::splat(c) }
                        }

                        #[inline(always)]
                        fn mul_add(self, b: Self, c: Self) -> Self {
                            // SAFETY: NEON is enabled at compile time.
                            unsafe { $V::from_raw($fma(c.raw(), self.raw(), b.raw())) }
                        }

                        #[inline(always)]
                        fn select(mask: $Mask, a: Self, b: Self) -> Self {
                            // SAFETY: NEON is enabled at compile time.
                            unsafe { $V::from_raw($bsl(mask, a.raw(), b.raw())) }
                        }

                        #[inline(always)]
                        fn and_bits(self, mask: $Bits) -> Self {
                            // SAFETY: NEON is enabled at compile time.
                            unsafe {
                                $V::from_raw($from_bits($and($to_bits(self.raw()), $dup(mask))))
                            }
                        }

                        #[inline(always)]
                        fn residual(self, b: Self, c: Self) -> Self {
                            // SAFETY: NEON is enabled at compile time.
                            unsafe { $V::from_raw($fms(c.raw(), self.raw(), b.raw())) }
                        }

                        #[inline(always)]
                        fn is_sign_negative(self) -> $Mask {
                            // SAFETY: NEON is enabled at compile time.
                            unsafe { $cltz($to_signed(self.raw())) }
                        }
                    }
                )*
            };
        }

        float_neon!(
            F32x4: f32, u32, uint32x4_t, vfmaq_f32, vfmsq_f32, vbslq_f32,
                vandq_u32, vdupq_n_u32, vreinterpretq_u32_f32, vreinterpretq_f32_u32,
                vcltzq_s32, vreinterpretq_s32_f32;
            F64x2: f64, u64, uint64x2_t, vfmaq_f64, vfmsq_f64, vbslq_f64,
                vandq_u64, vdupq_n_u64, vreinterpretq_u64_f64, vreinterpretq_f64_u64,
                vcltzq_s64, vreinterpretq_s64_f64;
        );
    }
}

#[cfg(test)]
mod tests {
    use super::Element;
    use crate::math::{Math, MathElement, UnaryOp};
    use crate::testing::{TestFloat, assert_same};
    use std::vec::Vec;

    /// `and_bits` clearing the sign: `|x|`, bit for bit.
    struct ClearSign;

    impl UnaryOp for ClearSign {
        fn apply<V: Math>(v: V) -> V {
            v.and_bits(!<V::Elem as Element>::SIGN)
        }
    }

    fn check<T: TestFloat + MathElement>() {
        let x: Vec<T> = (0..40)
            .map(|i| T::SPECIAL[i % 8])
            .chain((0..40).map(|i| T::from_f64(i as f64 - 20.5)))
            .collect();
        let want: Vec<T> = x
            .iter()
            .map(|v| T::from_bits64(v.bits() & !T::SIGN))
            .collect();
        for (backend, got) in T::every_backend::<ClearSign>(&x) {
            assert_same(&got, &want, &std::format!("and_bits {} {backend}", T::NAME));
        }
    }

    #[test]
    fn and_bits_masks_every_lane() {
        check::<f32>();
        check::<f64>();
    }
}
