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
}

impl Element for f32 {
    type Bits = u32;
    #[cfg(test)]
    const SIGN: u32 = 1 << 31;

    #[inline(always)]
    fn from_f32(x: f32) -> f32 {
        x
    }
}

impl Element for f64 {
    type Bits = u64;
    #[cfg(test)]
    const SIGN: u64 = 1 << 63;

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
            }
        )*
    };
}

float_scalar!(f32: u32, f64: u64);

cfg_x86! {
    mod x86 {
        use core::arch::x86_64::{
            __m256i, __mmask8, __mmask16, _mm256_and_si256, _mm256_blendv_pd, _mm256_blendv_ps,
            _mm256_castpd_si256, _mm256_castps_si256, _mm256_castsi256_pd, _mm256_castsi256_ps,
            _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_set1_epi32, _mm256_set1_epi64x,
            _mm512_and_si512, _mm512_castpd_si512, _mm512_castps_si512, _mm512_castsi512_pd,
            _mm512_castsi512_ps, _mm512_fmadd_pd, _mm512_fmadd_ps, _mm512_mask_blend_pd,
            _mm512_mask_blend_ps, _mm512_set1_epi32, _mm512_set1_epi64,
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
                select: |$m:ident, $a:ident, $b:ident| $select:expr,
                and: |$v:ident, $bits:ident| $and:expr $(,)?
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
                }
            };
        }

        float_x86!(F32x8: f32, u32, __m256i;
            fmadd: _mm256_fmadd_ps,
            select: |m, a, b| _mm256_blendv_ps(b, a, _mm256_castsi256_ps(m)),
            and: |v, bits| _mm256_castsi256_ps(_mm256_and_si256(
                _mm256_castps_si256(v),
                _mm256_set1_epi32(bits as i32),
            )),
        );

        float_x86!(F64x4: f64, u64, __m256i;
            fmadd: _mm256_fmadd_pd,
            select: |m, a, b| _mm256_blendv_pd(b, a, _mm256_castsi256_pd(m)),
            and: |v, bits| _mm256_castsi256_pd(_mm256_and_si256(
                _mm256_castpd_si256(v),
                _mm256_set1_epi64x(bits as i64),
            )),
        );

        float_x86!(F32x16: f32, u32, __mmask16;
            fmadd: _mm512_fmadd_ps,
            select: |m, a, b| _mm512_mask_blend_ps(m, b, a),
            and: |v, bits| _mm512_castsi512_ps(_mm512_and_si512(
                _mm512_castps_si512(v),
                _mm512_set1_epi32(bits as i32),
            )),
        );

        float_x86!(F64x8: f64, u64, __mmask8;
            fmadd: _mm512_fmadd_pd,
            select: |m, a, b| _mm512_mask_blend_pd(m, b, a),
            and: |v, bits| _mm512_castsi512_pd(_mm512_and_si512(
                _mm512_castpd_si512(v),
                _mm512_set1_epi64(bits as i64),
            )),
        );
    }
}

cfg_neon! {
    mod neon {
        use core::arch::aarch64::{
            uint32x4_t, uint64x2_t, vandq_u32, vandq_u64, vbslq_f32, vbslq_f64, vdupq_n_u32,
            vdupq_n_u64, vfmaq_f32, vfmaq_f64, vreinterpretq_f32_u32, vreinterpretq_f64_u64,
            vreinterpretq_u32_f32, vreinterpretq_u64_f64,
        };

        use super::Float;
        use crate::simd::SimdVector;
        use crate::simd::neon::{f32x4::F32x4, f64x2::F64x2};

        macro_rules! float_neon {
            ($($V:ident: $T:ident, $Bits:ident, $Mask:ty, $fma:ident, $bsl:ident,
                $and:ident, $dup:ident, $to_bits:ident, $from_bits:ident);* $(;)?) => {
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
                    }
                )*
            };
        }

        float_neon!(
            F32x4: f32, u32, uint32x4_t, vfmaq_f32, vbslq_f32,
                vandq_u32, vdupq_n_u32, vreinterpretq_u32_f32, vreinterpretq_f32_u32;
            F64x2: f64, u64, uint64x2_t, vfmaq_f64, vbslq_f64,
                vandq_u64, vdupq_n_u64, vreinterpretq_u64_f64, vreinterpretq_f64_u64;
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
