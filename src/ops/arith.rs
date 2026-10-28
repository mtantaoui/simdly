//! Element-wise `+ - * /` on `f32` slices, dispatched to the fastest backend.
//!
//! Each operation is a zero-sized type implementing [`BinaryOp`], written
//! once for every lane type (`f32` and the SIMD registers), so every
//! `(operation, backend)` pair compiles to its own inlined loop.

use core::ops::{Add, Div, Mul, Sub};

/// A lane type supporting the four arithmetic operators: `f32` or a SIMD
/// register of `f32`.
pub(crate) trait Arith:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
}

impl<T> Arith for T where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
{
}

/// An element-wise binary operation.
pub(crate) trait BinaryOp {
    /// Applies the operation lane-wise.
    fn apply<V: Arith>(a: V, b: V) -> V;
}

macro_rules! binary_op {
    ($($name:ident: $op:tt),* $(,)?) => {
        $(
            #[doc = concat!("`a ", stringify!($op), " b`.")]
            pub(crate) struct $name;

            impl BinaryOp for $name {
                #[inline(always)]
                fn apply<V: Arith>(a: V, b: V) -> V {
                    a $op b
                }
            }
        )*
    };
}

binary_op!(AddOp: +, SubOp: -, MulOp: *, DivOp: /);

#[cfg(any(feature = "alloc", test))]
/// `out[i] = Op(a[i], b[i])`; on return all of `out` is initialised.
///
/// # Panics
///
/// If `a`, `b` and `out` do not all have the same length.
#[inline]
pub(crate) fn binary_uninit<Op: BinaryOp>(
    a: &[f32],
    b: &[f32],
    out: &mut [core::mem::MaybeUninit<f32>],
) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    assert_eq!(a.len(), out.len(), "length mismatch");
    dispatch! {
        avx2 => x86::avx2_binary_uninit::<Op>(a, b, out),
        scalar => scalar_binary_uninit::<Op>(a, b, out),
    }
}

/// `a[i] = Op(a[i], b[i])`.
///
/// # Panics
///
/// If `a` and `b` have different lengths.
#[inline]
pub(crate) fn binary_in_place<Op: BinaryOp>(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    dispatch! {
        avx2 => x86::avx2_binary_in_place::<Op>(a, b),
        scalar => scalar_binary_in_place::<Op>(a, b),
    }
}

#[cfg(any(feature = "alloc", test))]
fn scalar_binary_uninit<Op: BinaryOp>(
    a: &[f32],
    b: &[f32],
    out: &mut [core::mem::MaybeUninit<f32>],
) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        o.write(Op::apply(x, y));
    }
}

fn scalar_binary_in_place<Op: BinaryOp>(a: &mut [f32], b: &[f32]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = Op::apply(*x, y);
    }
}

cfg_x86! {
    mod x86 {
        use super::BinaryOp;
        use crate::math::driver;
        use crate::simd::avx2::f32x8::F32x8;

        #[cfg(any(feature = "alloc", test))]
        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_binary_uninit<Op: BinaryOp>(
            a: &[f32],
            b: &[f32],
            out: &mut [core::mem::MaybeUninit<f32>],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function.
            unsafe { driver::binary_uninit::<F32x8>(a, b, out, Op::apply) }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_binary_in_place<Op: BinaryOp>(a: &mut [f32], b: &[f32]) {
            // SAFETY: AVX2 + FMA are enabled in this function.
            unsafe { driver::binary_in_place::<F32x8>(a, b, Op::apply) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    /// Lengths covering empty, tail-only, exact multiples and full + tail.
    const MAX_LEN: usize = 40;

    /// Finite values, signed zeros, infinities and NaN, so division hits
    /// `x / 0` and `inf / inf`.
    fn inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
        const SPECIAL: [f32; 6] = [0.0, -0.0, f32::INFINITY, -f32::INFINITY, f32::NAN, 1e-40];
        let a = (0..len).map(|i| {
            if i % 7 == 3 {
                SPECIAL[i % 6]
            } else {
                i as f32 * 0.37 - 5.0
            }
        });
        let b = (0..len).map(|i| {
            if i % 5 == 1 {
                SPECIAL[i % 6]
            } else {
                3.0 - i as f32 * 0.21
            }
        });
        (a.collect(), b.collect())
    }

    /// Bit-exact equality, except that any two NaNs match.
    fn assert_same(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let same = g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan());
            assert!(same, "{what}, index {i}: got {g:?}, want {w:?}");
        }
    }

    /// Plain loop the backends must match.
    fn reference<Op: BinaryOp>(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(&x, &y)| Op::apply(x, y)).collect()
    }

    fn check<Op: BinaryOp>(name: &str) {
        for len in 0..=MAX_LEN {
            let (a, b) = inputs(len);
            let want = reference::<Op>(&a, &b);

            let mut out = Vec::with_capacity(len);
            binary_uninit::<Op>(&a, &b, &mut out.spare_capacity_mut()[..len]);
            // SAFETY: `binary_uninit` initialised the first `len` elements.
            unsafe { out.set_len(len) };
            assert_same(&out, &want, &std::format!("{name}, len {len}"));

            let mut x = a.clone();
            binary_in_place::<Op>(&mut x, &b);
            assert_same(&x, &want, &std::format!("{name} in place, len {len}"));

            let mut x = a.clone();
            scalar_binary_in_place::<Op>(&mut x, &b);
            assert_same(&x, &want, &std::format!("{name} scalar, len {len}"));

            cfg_x86!(if {
                if crate::detect::has_avx2_fma() {
                    let mut x = a.clone();
                    // SAFETY: AVX2 + FMA checked just above.
                    unsafe { x86::avx2_binary_in_place::<Op>(&mut x, &b) };
                    assert_same(&x, &want, &std::format!("{name} avx2, len {len}"));
                }
            } else {});
        }
    }

    #[test]
    fn add_matches_scalar() {
        check::<AddOp>("add");
    }

    #[test]
    fn sub_matches_scalar() {
        check::<SubOp>("sub");
    }

    #[test]
    fn mul_matches_scalar() {
        check::<MulOp>("mul");
    }

    #[test]
    fn div_matches_scalar() {
        check::<DivOp>("div");
    }
}
