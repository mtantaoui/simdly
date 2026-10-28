//! Element-wise `+ - * /` on `f32` and `f64` slices, dispatched to the fastest
//! backend.
//!
//! Each operation is a zero-sized type implementing [`BinaryOp`], written
//! once for every lane type (`f32`, `f64` and the SIMD registers), so every
//! `(operation, element, backend)` triple compiles to its own inlined loop.

#[cfg(any(feature = "alloc", test))]
use core::mem::MaybeUninit;
use core::ops::{Add, Div, Mul, Sub};

/// A lane type supporting the four arithmetic operators: a float or a SIMD
/// register of floats.
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

/// An element type with arithmetic kernels: `f32` or `f64`.
pub(crate) trait ArithElement: Arith + Sized {
    /// `out[i] = Op(a[i], b[i])`; on return all of `out` is initialised.
    ///
    /// # Panics
    ///
    /// If `a`, `b` and `out` do not all have the same length.
    #[cfg(any(feature = "alloc", test))]
    fn binary_uninit<Op: BinaryOp>(a: &[Self], b: &[Self], out: &mut [MaybeUninit<Self>]);

    /// `a[i] = Op(a[i], b[i])`.
    ///
    /// # Panics
    ///
    /// If `a` and `b` have different lengths.
    fn binary_in_place<Op: BinaryOp>(a: &mut [Self], b: &[Self]);

    /// `Op(a, b)` on every backend this CPU supports, scalar included,
    /// whatever dispatch would pick: `(backend, result)` pairs.
    #[cfg(test)]
    fn every_backend<Op: BinaryOp>(
        a: &[Self],
        b: &[Self],
    ) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)>;
}

macro_rules! arith_element {
    ($($T:ty => $avx512:ident, $avx2:ident);* $(;)?) => {
        $(
            impl ArithElement for $T {
                #[cfg(any(feature = "alloc", test))]
                #[inline]
                fn binary_uninit<Op: BinaryOp>(
                    a: &[Self],
                    b: &[Self],
                    out: &mut [MaybeUninit<Self>],
                ) {
                    assert_eq!(a.len(), b.len(), "length mismatch");
                    assert_eq!(a.len(), out.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_binary_uninit::<x86::$avx512, Op>(a, b, out),
                        avx2 => x86::avx2_binary_uninit::<x86::$avx2, Op>(a, b, out),
                        scalar => scalar_binary_uninit::<Op, Self>(a, b, out),
                    }
                }

                #[inline]
                fn binary_in_place<Op: BinaryOp>(a: &mut [Self], b: &[Self]) {
                    assert_eq!(a.len(), b.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_binary_in_place::<x86::$avx512, Op>(a, b),
                        avx2 => x86::avx2_binary_in_place::<x86::$avx2, Op>(a, b),
                        scalar => scalar_binary_in_place::<Op, Self>(a, b),
                    }
                }

                #[cfg(test)]
                fn every_backend<Op: BinaryOp>(
                    a: &[Self],
                    b: &[Self],
                ) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)> {
                    let mut x = a.to_vec();
                    scalar_binary_in_place::<Op, Self>(&mut x, b);
                    #[allow(unused_mut, reason = "only scalar on some targets")]
                    let mut results = std::vec![("scalar", x)];
                    cfg_x86!(if {
                        if crate::detect::has_avx2_fma() {
                            let mut x = a.to_vec();
                            // SAFETY: AVX2 + FMA checked just above.
                            unsafe { x86::avx2_binary_in_place::<x86::$avx2, Op>(&mut x, b) };
                            results.push(("avx2", x));
                        }
                        if crate::detect::has_avx512f() {
                            let mut x = a.to_vec();
                            // SAFETY: AVX-512F checked just above.
                            unsafe { x86::avx512_binary_in_place::<x86::$avx512, Op>(&mut x, b) };
                            results.push(("avx512", x));
                        }
                    } else {});
                    results
                }
            }
        )*
    };
}

arith_element!(f32 => F32x16, F32x8; f64 => F64x8, F64x4);

#[cfg(any(feature = "alloc", test))]
fn scalar_binary_uninit<Op: BinaryOp, T: Arith>(a: &[T], b: &[T], out: &mut [MaybeUninit<T>]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        o.write(Op::apply(x, y));
    }
}

fn scalar_binary_in_place<Op: BinaryOp, T: Arith>(a: &mut [T], b: &[T]) {
    for (x, &y) in a.iter_mut().zip(b) {
        *x = Op::apply(*x, y);
    }
}

cfg_x86! {
    mod x86 {
        use super::{Arith, BinaryOp};
        use crate::math::driver;
        use crate::simd::SimdVector;
        pub(super) use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        pub(super) use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};

        #[cfg(any(feature = "alloc", test))]
        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_binary_uninit<V: SimdVector + Arith, Op: BinaryOp>(
            a: &[V::Scalar],
            b: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::binary_uninit::<V>(a, b, out, Op::apply) }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_binary_in_place<V: SimdVector + Arith, Op: BinaryOp>(
            a: &mut [V::Scalar],
            b: &[V::Scalar],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::binary_in_place::<V>(a, b, Op::apply) }
        }

        #[cfg(any(feature = "alloc", test))]
        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_binary_uninit<V: SimdVector + Arith, Op: BinaryOp>(
            a: &[V::Scalar],
            b: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::binary_uninit::<V>(a, b, out, Op::apply) }
        }

        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_binary_in_place<V: SimdVector + Arith, Op: BinaryOp>(
            a: &mut [V::Scalar],
            b: &[V::Scalar],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::binary_in_place::<V>(a, b, Op::apply) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TestFloat, assert_same};
    use std::vec::Vec;

    /// Lengths covering empty, tail-only, exact multiples and full + tail, for
    /// up to 16 lanes.
    const MAX_LEN: usize = 40;

    /// Finite values mixed with signed zeros, infinities, NaN and subnormals,
    /// so division hits `x / 0` and `inf / inf`.
    fn inputs<T: TestFloat>(len: usize) -> (Vec<T>, Vec<T>) {
        let a = (0..len).map(|i| match i % 7 {
            3 => T::SPECIAL[i % 8],
            _ => T::from_f64(i as f64 * 0.37 - 5.0),
        });
        let b = (0..len).map(|i| match i % 5 {
            1 => T::SPECIAL[i % 8],
            _ => T::from_f64(3.0 - i as f64 * 0.21),
        });
        (a.collect(), b.collect())
    }

    /// Plain loop the backends must match.
    fn reference<Op: BinaryOp, T: Arith>(a: &[T], b: &[T]) -> Vec<T> {
        a.iter().zip(b).map(|(&x, &y)| Op::apply(x, y)).collect()
    }

    fn check<Op: BinaryOp, T: TestFloat + ArithElement>(name: &str) {
        let name = std::format!("{name} {}", T::NAME);
        for len in 0..=MAX_LEN {
            let (a, b) = inputs::<T>(len);
            let want = reference::<Op, T>(&a, &b);

            let mut out = Vec::with_capacity(len);
            T::binary_uninit::<Op>(&a, &b, &mut out.spare_capacity_mut()[..len]);
            // SAFETY: `binary_uninit` initialised the first `len` elements.
            unsafe { out.set_len(len) };
            assert_same(&out, &want, &std::format!("{name}, len {len}"));

            let mut x = a.clone();
            T::binary_in_place::<Op>(&mut x, &b);
            assert_same(&x, &want, &std::format!("{name} in place, len {len}"));

            for (backend, x) in T::every_backend::<Op>(&a, &b) {
                assert_same(&x, &want, &std::format!("{name} {backend}, len {len}"));
            }
        }
    }

    fn check_all<Op: BinaryOp>(name: &str) {
        check::<Op, f32>(name);
        check::<Op, f64>(name);
    }

    #[test]
    fn add_matches_scalar() {
        check_all::<AddOp>("add");
    }

    #[test]
    fn sub_matches_scalar() {
        check_all::<SubOp>("sub");
    }

    #[test]
    fn mul_matches_scalar() {
        check_all::<MulOp>("mul");
    }

    #[test]
    fn div_matches_scalar() {
        check_all::<DivOp>("div");
    }
}
