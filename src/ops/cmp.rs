//! Element-wise comparisons on `f32` slices, producing `bool`s.
//!
//! Each comparison is a zero-sized type implementing [`CmpOp`], written once
//! for every lane type (`f32` and the SIMD registers). Comparisons with NaN
//! are false, except `!=`, which is true, as with Rust's operators.
//!
//! AVX2 compares 32 elements per step (4 registers) and narrows the 32-bit
//! lane masks to bytes with two rounds of saturating packs, so all 32 `bool`s
//! are written with one store.

use core::mem::MaybeUninit;

/// A lane type that can be compared: `f32`, or a SIMD register of `f32`.
pub(crate) trait Compare: Copy {
    /// `bool` for `f32`; for a register, all-ones or all-zero lanes.
    type Mask: Copy;

    fn lt(self, rhs: Self) -> Self::Mask;
    fn le(self, rhs: Self) -> Self::Mask;
    fn gt(self, rhs: Self) -> Self::Mask;
    fn ge(self, rhs: Self) -> Self::Mask;
    fn eq(self, rhs: Self) -> Self::Mask;
    fn ne(self, rhs: Self) -> Self::Mask;
}

/// An element-wise comparison.
pub(crate) trait CmpOp {
    /// Applies the comparison lane-wise.
    fn apply<V: Compare>(a: V, b: V) -> V::Mask;
}

macro_rules! cmp_op {
    ($($name:ident: $method:ident $op:tt),* $(,)?) => {
        impl Compare for f32 {
            type Mask = bool;
            $(
                #[inline(always)]
                fn $method(self, rhs: Self) -> bool {
                    self $op rhs
                }
            )*
        }

        $(
            #[doc = concat!("`a ", stringify!($op), " b`.")]
            pub(crate) struct $name;

            impl CmpOp for $name {
                #[inline(always)]
                fn apply<V: Compare>(a: V, b: V) -> V::Mask {
                    a.$method(b)
                }
            }
        )*
    };
}

cmp_op!(LtOp: lt <, LeOp: le <=, GtOp: gt >, GeOp: ge >=, EqOp: eq ==, NeOp: ne !=);

/// `out[i] = Op(a[i], b[i])`; on return all of `out` is initialised.
///
/// # Panics
///
/// If `a`, `b` and `out` do not all have the same length.
#[inline]
pub(crate) fn cmp_uninit<Op: CmpOp>(a: &[f32], b: &[f32], out: &mut [MaybeUninit<bool>]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    assert_eq!(a.len(), out.len(), "length mismatch");
    dispatch! {
        avx2 => x86::avx2_cmp::<Op>(a, b, out),
        scalar => scalar_cmp::<Op>(a, b, out),
    }
}

/// `out[i] = Op(a[i], b)`; on return all of `out` is initialised.
///
/// # Panics
///
/// If `a` and `out` have different lengths.
#[inline]
pub(crate) fn cmp_scalar_uninit<Op: CmpOp>(a: &[f32], b: f32, out: &mut [MaybeUninit<bool>]) {
    assert_eq!(a.len(), out.len(), "length mismatch");
    dispatch! {
        avx2 => x86::avx2_cmp_scalar::<Op>(a, b, out),
        scalar => scalar_cmp_scalar::<Op>(a, b, out),
    }
}

fn scalar_cmp<Op: CmpOp>(a: &[f32], b: &[f32], out: &mut [MaybeUninit<bool>]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        o.write(Op::apply(x, y));
    }
}

fn scalar_cmp_scalar<Op: CmpOp>(a: &[f32], b: f32, out: &mut [MaybeUninit<bool>]) {
    for (o, &x) in out.iter_mut().zip(a) {
        o.write(Op::apply(x, b));
    }
}

cfg_x86! {
    mod x86 {
        use core::arch::x86_64::{
            __m256i, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _CMP_NEQ_UQ,
            _mm256_and_si256, _mm256_castps_si256, _mm256_cmp_ps, _mm256_packs_epi16,
            _mm256_packs_epi32, _mm256_permutevar8x32_epi32, _mm256_set1_epi8,
            _mm256_setr_epi32, _mm256_storeu_si256,
        };
        use core::mem::MaybeUninit;

        use super::{CmpOp, Compare};
        use crate::simd::SimdVector;
        use crate::simd::avx2::f32x8::F32x8;

        macro_rules! compare {
            ($($method:ident: $pred:ident),* $(,)?) => {
                impl Compare for F32x8 {
                    type Mask = __m256i;
                    $(
                        #[inline(always)]
                        fn $method(self, rhs: Self) -> __m256i {
                            // SAFETY: an F32x8 exists.
                            unsafe {
                                _mm256_castps_si256(_mm256_cmp_ps::<{ $pred }>(self.raw(), rhs.raw()))
                            }
                        }
                    )*
                }
            };
        }

        // Ordered (`_OQ`): false if either side is NaN; `!=` is unordered
        // (`_UQ`): true if either side is NaN. Quiet: NaN raises no flag.
        compare!(
            lt: _CMP_LT_OQ,
            le: _CMP_LE_OQ,
            gt: _CMP_GT_OQ,
            ge: _CMP_GE_OQ,
            eq: _CMP_EQ_OQ,
            ne: _CMP_NEQ_UQ,
        );

        /// The right-hand side: a slice, or one value broadcast to every lane.
        trait Rhs: Copy {
            /// # Safety
            ///
            /// The CPU must support AVX2, and `i + 8` elements must be readable.
            unsafe fn get(self, i: usize) -> F32x8;

            /// # Safety
            ///
            /// The CPU must support AVX2, `n < 8`, and `i + n` elements must be
            /// readable.
            unsafe fn get_partial(self, i: usize, n: usize) -> F32x8;
        }

        impl Rhs for &[f32] {
            #[inline(always)]
            unsafe fn get(self, i: usize) -> F32x8 {
                // SAFETY: guaranteed by the caller.
                unsafe { F32x8::load(self.as_ptr().add(i)) }
            }

            #[inline(always)]
            unsafe fn get_partial(self, i: usize, n: usize) -> F32x8 {
                // SAFETY: guaranteed by the caller.
                unsafe { F32x8::load_partial(self.as_ptr().add(i), n) }
            }
        }

        impl Rhs for F32x8 {
            #[inline(always)]
            unsafe fn get(self, _: usize) -> F32x8 {
                self
            }

            #[inline(always)]
            unsafe fn get_partial(self, _: usize, _: usize) -> F32x8 {
                self
            }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_cmp<Op: CmpOp>(a: &[f32], b: &[f32], out: &mut [MaybeUninit<bool>]) {
            // SAFETY: AVX2 + FMA are enabled in this function, and the caller
            // checked that all three slices have the same length.
            unsafe { cmp::<Op, _>(a, b, out) }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_cmp_scalar<Op: CmpOp>(a: &[f32], b: f32, out: &mut [MaybeUninit<bool>]) {
            // SAFETY: AVX2 + FMA are enabled in this function, and the caller
            // checked that `a` and `out` have the same length.
            unsafe { cmp::<Op, _>(a, F32x8::splat(b), out) }
        }

        /// `out[i] = Op(a[i], b[i])`, 32 elements per step.
        ///
        /// # Safety
        ///
        /// The CPU must support AVX2 + FMA, `out` must have `a.len()` elements
        /// and `b` must be readable for `a.len()` elements.
        #[inline(always)]
        unsafe fn cmp<Op: CmpOp, R: Rhs>(a: &[f32], b: R, out: &mut [MaybeUninit<bool>]) {
            const STEP: usize = 4 * F32x8::LANES;
            let (len, x, dst) = (a.len(), a.as_ptr(), out.as_mut_ptr().cast::<u8>());
            let full = len - len % STEP;

            let mut i = 0;
            while i < full {
                // SAFETY: the caller guarantees AVX2 + FMA and the lengths, and
                // `i + 32 <= len`.
                unsafe {
                    let masks = core::array::from_fn(|k| {
                        let j = i + k * F32x8::LANES;
                        Op::apply(F32x8::load(x.add(j)), b.get(j))
                    });
                    _mm256_storeu_si256(dst.add(i).cast(), pack(masks));
                }
                i += STEP;
            }

            let tail = len - full;
            if tail > 0 {
                let mut buf = [0_u8; STEP];
                // SAFETY: the caller guarantees AVX2 + FMA and the lengths. Each
                // register reads only the lanes left in `a` and `b`; registers
                // past the end are zero, and their results are not copied.
                // `pack` produces only 0 and 1, which are valid `bool`s.
                unsafe {
                    let masks = core::array::from_fn(|k| {
                        let j = full + k * F32x8::LANES;
                        let (va, vb) = match tail.saturating_sub(k * F32x8::LANES) {
                            0 => (F32x8::zero(), F32x8::zero()),
                            n if n >= F32x8::LANES => (F32x8::load(x.add(j)), b.get(j)),
                            n => (F32x8::load_partial(x.add(j), n), b.get_partial(j, n)),
                        };
                        Op::apply(va, vb)
                    });
                    _mm256_storeu_si256(buf.as_mut_ptr().cast(), pack(masks));
                    core::ptr::copy_nonoverlapping(buf.as_ptr(), dst.add(full), tail);
                }
            }
        }

        /// Narrows four 8-lane masks (each 32-bit lane all-ones or zero) to 32
        /// `bool` bytes, in order.
        ///
        /// # Safety
        ///
        /// The CPU must support AVX2.
        #[inline(always)]
        unsafe fn pack(m: [__m256i; 4]) -> __m256i {
            // SAFETY: the caller guarantees AVX2.
            unsafe {
                // Signed saturation keeps -1 as -1: i32 -> i16 -> i8.
                let p = _mm256_packs_epi16(
                    _mm256_packs_epi32(m[0], m[1]),
                    _mm256_packs_epi32(m[2], m[3]),
                );
                // The packs work per 128-bit half, so 4-byte groups come out as
                // m0[0..4] m1[0..4] m2[0..4] m3[0..4] | m0[4..8] m1[4..8] ...
                let p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
                // 0xFF -> 0x01.
                _mm256_and_si256(p, _mm256_set1_epi8(1))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    /// Covers empty, tail-only, one full 32-element step, and steps + tail.
    const MAX_LEN: usize = 100;

    const SPECIAL: [f32; 8] = [
        0.0,
        -0.0,
        f32::INFINITY,
        -f32::INFINITY,
        f32::NAN,
        1e-40,
        -1e-40,
        1.0,
    ];

    /// Mixed values where `a[i] == b[i]` regularly, plus signed zeros,
    /// infinities, NaN and subnormals on either side.
    fn inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
        let a = (0..len).map(|i| match i % 9 {
            2 => SPECIAL[i % 8],
            _ => (i % 13) as f32 - 6.0,
        });
        let b = (0..len).map(|i| match i % 7 {
            3 => SPECIAL[(i / 7) % 8],
            _ => (i % 11) as f32 - 5.0,
        });
        (a.collect(), b.collect())
    }

    /// Runs `f` on an output of `len` elements inside a buffer of sentinels,
    /// checks every byte written is a valid `bool` and none past `len` was
    /// touched, then returns the result.
    fn run(len: usize, f: impl FnOnce(&mut [MaybeUninit<bool>])) -> Vec<bool> {
        const SENTINEL: u8 = 0xAA;
        let mut buf = std::vec![SENTINEL; len + 8];
        // SAFETY: `MaybeUninit<bool>` has the layout of `u8`, and `buf` holds
        // at least `len` bytes.
        let out = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), len) };
        f(out);
        assert!(buf[..len].iter().all(|&b| b <= 1), "invalid bool byte");
        assert!(
            buf[len..].iter().all(|&b| b == SENTINEL),
            "wrote past {len}"
        );
        buf[..len].iter().map(|&b| b == 1).collect()
    }

    fn check<Op: CmpOp>(name: &str) {
        for len in 0..=MAX_LEN {
            let (a, b) = inputs(len);
            let want: Vec<bool> = a.iter().zip(&b).map(|(&x, &y)| Op::apply(x, y)).collect();

            let got = run(len, |out| cmp_uninit::<Op>(&a, &b, out));
            assert_eq!(got, want, "{name}, len {len}");
            let got = run(len, |out| scalar_cmp::<Op>(&a, &b, out));
            assert_eq!(got, want, "{name} scalar, len {len}");
            cfg_x86!(if {
                if crate::detect::has_avx2_fma() {
                    // SAFETY: AVX2 + FMA checked just above.
                    let got = run(len, |out| unsafe { x86::avx2_cmp::<Op>(&a, &b, out) });
                    assert_eq!(got, want, "{name} avx2, len {len}");
                }
            } else {});

            for &s in &SPECIAL {
                let want: Vec<bool> = a.iter().map(|&x| Op::apply(x, s)).collect();
                let got = run(len, |out| cmp_scalar_uninit::<Op>(&a, s, out));
                assert_eq!(got, want, "{name} vs {s}, len {len}");
                let got = run(len, |out| scalar_cmp_scalar::<Op>(&a, s, out));
                assert_eq!(got, want, "{name} scalar vs {s}, len {len}");
                cfg_x86!(if {
                    if crate::detect::has_avx2_fma() {
                        // SAFETY: AVX2 + FMA checked just above.
                        let got =
                            run(len, |out| unsafe { x86::avx2_cmp_scalar::<Op>(&a, s, out) });
                        assert_eq!(got, want, "{name} avx2 vs {s}, len {len}");
                    }
                } else {});
            }
        }
    }

    #[test]
    fn lt_matches_scalar() {
        check::<LtOp>("lt");
    }

    #[test]
    fn le_matches_scalar() {
        check::<LeOp>("le");
    }

    #[test]
    fn gt_matches_scalar() {
        check::<GtOp>("gt");
    }

    #[test]
    fn ge_matches_scalar() {
        check::<GeOp>("ge");
    }

    #[test]
    fn eq_matches_scalar() {
        check::<EqOp>("eq");
    }

    #[test]
    fn ne_matches_scalar() {
        check::<NeOp>("ne");
    }

    /// NaN and signed-zero semantics, spelled out.
    #[test]
    fn nan_and_signed_zero() {
        let nan = f32::NAN;
        assert!(!LtOp::apply(nan, 1.0) && !GeOp::apply(nan, 1.0) && !EqOp::apply(nan, nan));
        assert!(NeOp::apply(nan, nan));
        assert!(EqOp::apply(0.0_f32, -0.0) && !LtOp::apply(-0.0_f32, 0.0));
    }
}
