//! `√x`, correctly rounded (0 ULP) on every backend: `√-0.0` is `-0.0`,
//! `√+∞` is `+∞`, and negative inputs and NaN give NaN.
//!
//! Every ISA has an IEEE-754 square root instruction (`vsqrtps` / `vsqrtpd`,
//! NEON `fsqrt`), and `std`'s `f32::sqrt` compiles to the scalar one. Without
//! `std`, `core` has no `sqrt`, so the scalar fallback computes it exactly with
//! integers ([`soft`]).

/// A lane type with a square root.
pub(crate) trait Sqrt: Copy {
    fn sqrt(self) -> Self;
}

macro_rules! sqrt_float {
    ($($T:ident: $soft:ident),*) => {
        $(
            impl Sqrt for $T {
                #[inline(always)]
                fn sqrt(self) -> Self {
                    #[cfg(feature = "std")]
                    {
                        $T::sqrt(self)
                    }
                    #[cfg(not(feature = "std"))]
                    {
                        soft::$soft(self)
                    }
                }
            }
        )*
    };
}

sqrt_float!(f32: sqrt_f32, f64: sqrt_f64);

/// Correctly rounded square roots without an FPU square root.
#[cfg_attr(
    feature = "std",
    allow(dead_code, reason = "used without `std`; tested always")
)]
mod soft {
    /// `soft!(name: float, bits, wide, fraction bits, exponent bias)`.
    ///
    /// For finite `x > 0`, write `x = m · 2^e` with an integer `m` of `P`
    /// significant bits (`P = FRAC + 1`), shifted left once if needed so that
    /// `e - (P + 1)` is even. Then `√x = √(m · 2^(P+1)) · 2^((e - P - 1) / 2)`,
    /// and `s = ⌊√(m · 2^(P+1))⌋` has `P + 1` bits: the result's `P` bits plus one
    /// rounding bit. Rounding up when that bit is set is round-to-nearest,
    /// because a square root is never exactly halfway between two floats.
    /// The result is always normal: `√` of the smallest subnormal is far
    /// above the smallest normal.
    macro_rules! soft {
        ($name:ident: $T:ident, $Bits:ty, $Wide:ty, $FRAC:expr, $BIAS:expr) => {
            pub(super) fn $name(x: $T) -> $T {
                const FRAC: u32 = $FRAC;
                const P: u32 = FRAC + 1;
                if x.is_nan() || x == 0.0 || x == $T::INFINITY {
                    // Quiets a signalling NaN; keeps ±0.0 and +∞.
                    return x + x;
                }
                if x < 0.0 {
                    return $T::NAN;
                }

                let bits = x.to_bits();
                let biased = (bits >> FRAC) as i32;
                let fraction = bits & ((1 << FRAC) - 1);
                // `x = m · 2^e`, `m` in [2^FRAC, 2^P).
                let (mut m, mut e) = if biased == 0 {
                    // Subnormal: normalise.
                    let shift = fraction.leading_zeros() - (<$Bits>::BITS - P);
                    (fraction << shift, 1 - $BIAS - FRAC as i32 - shift as i32)
                } else {
                    (fraction | 1 << FRAC, biased - $BIAS - FRAC as i32)
                };
                // Make `e - (P + 1)` even.
                if (e - P as i32 - 1) & 1 != 0 {
                    m <<= 1;
                    e -= 1;
                }

                // `s = ⌊√(m · 2^(P+1))⌋`, one bit per step, from 4^P: the
                // operand is below 4^(P+1).
                let (mut rem, mut s): ($Wide, $Wide) = ((m as $Wide) << (P + 1), 0);
                let mut bit: $Wide = 1 << (2 * P);
                while bit != 0 {
                    if rem >= s + bit {
                        rem -= s + bit;
                        s = (s >> 1) + bit;
                    } else {
                        s >>= 1;
                    }
                    bit >>= 2;
                }
                // `P` bits, rounded; may carry to 2^P.
                let q = ((s >> 1) + (s & 1)) as $Bits;
                // `√x = q · 2^k`; `q`'s implicit bit adds one to the exponent
                // field, which also absorbs the carry.
                let k = (e - P as i32 - 1) / 2 + 1;
                let field = (k + FRAC as i32 + $BIAS - 1) as $Bits;
                $T::from_bits((field << FRAC) + q)
            }
        };
    }

    soft!(sqrt_f32: f32, u32, u64, 23, 127);
    soft!(sqrt_f64: f64, u64, u128, 52, 1023);

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::testing::{TestFloat, assert_same};
        use std::vec::Vec;

        /// Special values, every power of two, their neighbours, perfect
        /// squares, subnormals, and pseudo-random bit patterns.
        fn inputs<T: TestFloat>(from_bits: fn(u64) -> T, width: u32) -> Vec<T> {
            let mut xs: Vec<T> = T::SPECIAL.to_vec();
            let mask = if width == 64 { !0 } else { (1 << width) - 1 };
            let frac = if width == 64 { 52 } else { 23 };
            for e in 0..(1 << (width - frac - 1)) {
                let p = (e as u64) << frac;
                xs.extend([p, p + 1, p.wrapping_sub(1) & mask].map(from_bits));
            }
            for i in 0..2000_u64 {
                xs.push(T::from_f64((i * i) as f64));
                xs.push(from_bits(i));
            }
            // xorshift64
            let mut r = 0x9E37_79B9_7F4A_7C15_u64;
            for _ in 0..200_000 {
                r ^= r << 13;
                r ^= r >> 7;
                r ^= r << 17;
                xs.push(from_bits(r & mask));
            }
            xs
        }

        #[test]
        fn soft_sqrt_f32_is_correctly_rounded() {
            let xs = inputs(|b| f32::from_bits(b as u32), 32);
            let got: Vec<f32> = xs.iter().map(|&x| sqrt_f32(x)).collect();
            let want: Vec<f32> = xs.iter().map(|&x| x.sqrt()).collect();
            assert_same(&got, &want, "soft sqrt f32");
        }

        /// Every `f32`: `cargo test --release -- --ignored exhaustive`.
        #[test]
        #[ignore = "slow: 2^32 inputs"]
        fn soft_sqrt_f32_exhaustive() {
            for b in 0..=u32::MAX {
                let x = f32::from_bits(b);
                let (got, want) = (sqrt_f32(x), x.sqrt());
                let same = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
                assert!(same, "sqrt({x:e}) = {got:e}, want {want:e}");
            }
        }

        #[test]
        fn soft_sqrt_f64_is_correctly_rounded() {
            let xs = inputs(f64::from_bits, 64);
            let got: Vec<f64> = xs.iter().map(|&x| sqrt_f64(x)).collect();
            let want: Vec<f64> = xs.iter().map(|&x| x.sqrt()).collect();
            assert_same(&got, &want, "soft sqrt f64");
        }
    }
}

cfg_x86! {
    mod x86 {
        use core::arch::x86_64::{_mm256_sqrt_pd, _mm256_sqrt_ps, _mm512_sqrt_pd, _mm512_sqrt_ps};

        use super::Sqrt;
        use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};

        impl Sqrt for F32x8 {
            #[inline(always)]
            fn sqrt(self) -> Self {
                // SAFETY: an F32x8 exists, so AVX2 + FMA are available.
                unsafe { F32x8::from_raw(_mm256_sqrt_ps(self.raw())) }
            }
        }

        impl Sqrt for F64x4 {
            #[inline(always)]
            fn sqrt(self) -> Self {
                // SAFETY: an F64x4 exists, so AVX2 + FMA are available.
                unsafe { F64x4::from_raw(_mm256_sqrt_pd(self.raw())) }
            }
        }

        impl Sqrt for F32x16 {
            #[inline(always)]
            fn sqrt(self) -> Self {
                // SAFETY: an F32x16 exists, so AVX-512F is available.
                unsafe { F32x16::from_raw(_mm512_sqrt_ps(self.raw())) }
            }
        }

        impl Sqrt for F64x8 {
            #[inline(always)]
            fn sqrt(self) -> Self {
                // SAFETY: an F64x8 exists, so AVX-512F is available.
                unsafe { F64x8::from_raw(_mm512_sqrt_pd(self.raw())) }
            }
        }
    }
}

cfg_neon! {
    mod neon {
        use core::arch::aarch64::{vsqrtq_f32, vsqrtq_f64};

        use super::Sqrt;
        use crate::simd::neon::{f32x4::F32x4, f64x2::F64x2};

        impl Sqrt for F32x4 {
            #[inline(always)]
            fn sqrt(self) -> Self {
                // SAFETY: NEON is enabled at compile time.
                unsafe { F32x4::from_raw(vsqrtq_f32(self.raw())) }
            }
        }

        impl Sqrt for F64x2 {
            #[inline(always)]
            fn sqrt(self) -> Self {
                // SAFETY: NEON is enabled at compile time.
                unsafe { F64x2::from_raw(vsqrtq_f64(self.raw())) }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{MathElement, SqrtOp};
    use crate::testing::{TestFloat, assert_same};
    use std::vec::Vec;

    /// Every special value among positive and negative finite values, and
    /// perfect squares, whose roots are exact.
    fn inputs<T: TestFloat>(len: usize) -> Vec<T> {
        (0..len)
            .map(|i| match i % 5 {
                1 => T::SPECIAL[i % 8],
                3 => T::from_f64((i * i) as f64),
                _ => T::from_f64(i as f64 * 0.75 - 9.0),
            })
            .collect()
    }

    /// The correctly rounded result: `std`'s. For `f32` through `f64`, which
    /// rounds correctly too: `f64` has more than `2 · 24 + 2` bits.
    fn reference<T: TestFloat>(x: &[T]) -> Vec<T> {
        x.iter().map(|&v| T::from_f64(v.to_f64().sqrt())).collect()
    }

    fn check<T: TestFloat + MathElement>() {
        for len in 0..=40 {
            let x = inputs::<T>(len);
            let want = reference(&x);
            let what = std::format!("sqrt {}, len {len}", T::NAME);

            let mut out = Vec::with_capacity(len);
            T::unary_uninit::<SqrtOp>(&x, &mut out.spare_capacity_mut()[..len]);
            // SAFETY: `unary_uninit` initialised the first `len` elements.
            unsafe { out.set_len(len) };
            assert_same(&out, &want, &what);

            let mut y = x.clone();
            T::unary_in_place::<SqrtOp>(&mut y);
            assert_same(&y, &want, &std::format!("{what}, in place"));

            for (backend, y) in T::every_backend::<SqrtOp>(&x) {
                assert_same(&y, &want, &std::format!("{what}, {backend}"));
            }
        }
    }

    #[test]
    fn sqrt_is_correctly_rounded() {
        check::<f32>();
        check::<f64>();
    }
}
