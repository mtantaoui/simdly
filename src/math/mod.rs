//! Element-wise mathematical functions, through [`SliceMath`].
//!
//! Each function is one file implementing a lane-level trait (`Abs`, `Acos`,
//! …) for every lane type (`f32`, `f64` and the SIMD registers), and a
//! zero-sized [`UnaryOp`] names it, so every `(function, element, backend)`
//! triple compiles to its own inlined loop over the [`driver`].
//!
//! A single instruction ([`abs`], [`sqrt`]) is implemented per lane type. An
//! algorithm ([`acos`], [`asin`], [`atan`]) is written once, generic over [`Float`], and each
//! backend runs it with its own instructions.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(any(feature = "alloc", test))]
use core::mem::MaybeUninit;

mod abs;
mod acos;
mod asin;
mod atan;
mod float;
mod sqrt;

cfg_simd! {
    pub(crate) mod driver;
}

use abs::Abs;
use acos::Acos;
use asin::Asin;
use atan::Atan;
use float::Float;
use sqrt::Sqrt;

/// A lane type with every math function: a float or a SIMD register of
/// floats.
pub(crate) trait Math: Float + Acos + Asin + Atan {}

impl<T: Float + Acos + Asin + Atan> Math for T {}

/// An element-wise math function.
pub(crate) trait UnaryOp {
    /// Applies the function lane-wise.
    fn apply<V: Math>(v: V) -> V;
}

cfg_simd! {
    impl<F: UnaryOp, V: Math> driver::UnaryFn<V> for F {
        #[inline(always)]
        fn call(x: V) -> V {
            F::apply(x)
        }
    }
}

/// `|x|`.
pub(crate) struct AbsOp;

impl UnaryOp for AbsOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.abs()
    }
}

/// `acos(x)`.
pub(crate) struct AcosOp;

impl UnaryOp for AcosOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.acos()
    }
}

/// `asin(x)`.
pub(crate) struct AsinOp;

impl UnaryOp for AsinOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.asin()
    }
}

/// `atan(x)`.
pub(crate) struct AtanOp;

impl UnaryOp for AtanOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.atan()
    }
}

/// `√x`.
pub(crate) struct SqrtOp;

impl UnaryOp for SqrtOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.sqrt()
    }
}

/// An element type with math kernels: `f32` or `f64`.
pub(crate) trait MathElement: Math + Sized {
    /// `out[i] = F(x[i])`; on return all of `out` is initialised.
    ///
    /// # Panics
    ///
    /// If `x` and `out` have different lengths.
    #[cfg(any(feature = "alloc", test))]
    fn unary_uninit<F: UnaryOp>(x: &[Self], out: &mut [MaybeUninit<Self>]);

    /// `x[i] = F(x[i])`.
    fn unary_in_place<F: UnaryOp>(x: &mut [Self]);

    /// `F(x)` on every backend this CPU supports, scalar included, whatever
    /// dispatch would pick: `(backend, result)` pairs.
    #[cfg(test)]
    fn every_backend<F: UnaryOp>(x: &[Self]) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)>;
}

macro_rules! math_element {
    ($($T:ty => $avx512:ident, $avx2:ident, $neon:ident);* $(;)?) => {
        $(
            impl MathElement for $T {
                #[cfg(any(feature = "alloc", test))]
                #[inline]
                fn unary_uninit<F: UnaryOp>(x: &[Self], out: &mut [MaybeUninit<Self>]) {
                    assert_eq!(x.len(), out.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_unary_uninit::<x86::$avx512, F>(x, out),
                        avx2 => x86::avx2_unary_uninit::<x86::$avx2, F>(x, out),
                        neon => neon::unary_uninit::<neon::$neon, F>(x, out),
                        scalar => scalar_unary_uninit::<F, Self>(x, out),
                    }
                }

                #[inline]
                fn unary_in_place<F: UnaryOp>(x: &mut [Self]) {
                    dispatch! {
                        avx512 => x86::avx512_unary_in_place::<x86::$avx512, F>(x),
                        avx2 => x86::avx2_unary_in_place::<x86::$avx2, F>(x),
                        neon => neon::unary_in_place::<neon::$neon, F>(x),
                        scalar => scalar_unary_in_place::<F, Self>(x),
                    }
                }

                #[cfg(test)]
                fn every_backend<F: UnaryOp>(
                    x: &[Self],
                ) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)> {
                    let mut y = x.to_vec();
                    scalar_unary_in_place::<F, Self>(&mut y);
                    #[allow(unused_mut, reason = "only scalar on some targets")]
                    let mut results = std::vec![("scalar", y)];
                    cfg_x86!(if {
                        if crate::detect::has_avx2_fma() {
                            let mut y = x.to_vec();
                            // SAFETY: AVX2 + FMA checked just above.
                            unsafe { x86::avx2_unary_in_place::<x86::$avx2, F>(&mut y) };
                            results.push(("avx2", y));
                        }
                        if crate::detect::has_avx512f() {
                            let mut y = x.to_vec();
                            // SAFETY: AVX-512F checked just above.
                            unsafe { x86::avx512_unary_in_place::<x86::$avx512, F>(&mut y) };
                            results.push(("avx512", y));
                        }
                    } else {});
                    cfg_neon!(if {
                        let mut y = x.to_vec();
                        neon::unary_in_place::<neon::$neon, F>(&mut y);
                        results.push(("neon", y));
                    } else {});
                    results
                }
            }
        )*
    };
}

math_element!(f32 => F32x16, F32x8, F32x4; f64 => F64x8, F64x4, F64x2);

#[cfg(any(feature = "alloc", test))]
fn scalar_unary_uninit<F: UnaryOp, T: Math>(x: &[T], out: &mut [MaybeUninit<T>]) {
    for (o, &v) in out.iter_mut().zip(x) {
        o.write(F::apply(v));
    }
}

fn scalar_unary_in_place<F: UnaryOp, T: Math>(x: &mut [T]) {
    for v in x {
        *v = F::apply(*v);
    }
}

cfg_x86! {
    mod x86 {
        use super::{Math, UnaryOp, driver};
        use crate::simd::SimdVector;
        pub(super) use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        pub(super) use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};

        #[cfg(any(feature = "alloc", test))]
        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_unary_uninit<V: SimdVector + Math, F: UnaryOp>(
            x: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::unary_uninit::<V, F>(x, out) }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_unary_in_place<V: SimdVector + Math, F: UnaryOp>(x: &mut [V::Scalar]) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::unary_in_place::<V, F>(x) }
        }

        #[cfg(any(feature = "alloc", test))]
        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_unary_uninit<V: SimdVector + Math, F: UnaryOp>(
            x: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::unary_uninit::<V, F>(x, out) }
        }

        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_unary_in_place<V: SimdVector + Math, F: UnaryOp>(x: &mut [V::Scalar]) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::unary_in_place::<V, F>(x) }
        }
    }
}

cfg_neon! {
    mod neon {
        use super::{Math, UnaryOp, driver};
        use crate::simd::SimdVector;
        pub(super) use crate::simd::neon::{f32x4::F32x4, f64x2::F64x2};

        #[cfg(any(feature = "alloc", test))]
        #[inline]
        pub(super) fn unary_uninit<V: SimdVector + Math, F: UnaryOp>(
            x: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::unary_uninit::<V, F>(x, out) }
        }

        #[inline]
        pub(super) fn unary_in_place<V: SimdVector + Math, F: UnaryOp>(x: &mut [V::Scalar]) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::unary_in_place::<V, F>(x) }
        }
    }
}

/// Element-wise math functions on slices, using the fastest backend
/// available (see [`backend`](crate::backend)).
///
/// Implemented for `[f32]` and `[f64]`, so it also applies to `Vec`s, arrays
/// and anything else that dereferences to a slice. Every function documents
/// its error bound against the correctly rounded result.
///
/// The `_in_place` methods need no allocator; the others return a new `Vec`
/// and need the `alloc` feature.
///
/// ```
/// use simdly::SliceMath;
///
/// let mut a = [-1.5_f32, 0.0, 2.0];
/// a.abs_in_place();
/// assert_eq!(a, [1.5, 0.0, 2.0]);
/// ```
pub trait SliceMath<T>: crate::ops::sealed::Sealed {
    /// `|self[i]|`. Exact: it only clears the sign bit, so `-0.0` becomes
    /// `0.0` and NaN stays NaN.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![-1.0_f64, 2.0, -0.0];
    /// assert_eq!(a.abs(), [1.0, 2.0, 0.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn abs(&self) -> Vec<T>;

    /// `self[i] = |self[i]|`.
    fn abs_in_place(&mut self);

    /// `acos(self[i])` in radians, in `[0, π]`. Error ≤ 1 ULP; NaN outside
    /// `[−1, 1]`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![1.0_f64, 0.0, -1.0];
    /// assert_eq!(a.acos(), [0.0, core::f64::consts::FRAC_PI_2, core::f64::consts::PI]);
    /// ```
    #[cfg(feature = "alloc")]
    fn acos(&self) -> Vec<T>;

    /// `self[i] = acos(self[i])`.
    fn acos_in_place(&mut self);

    /// `asin(self[i])` in radians, in `[−π/2, π/2]`. Error ≤ 1 ULP; NaN
    /// outside `[−1, 1]`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![1.0_f32, 0.0, -1.0];
    /// assert_eq!(a.asin(), [core::f32::consts::FRAC_PI_2, 0.0, -core::f32::consts::FRAC_PI_2]);
    /// ```
    #[cfg(feature = "alloc")]
    fn asin(&self) -> Vec<T>;

    /// `self[i] = asin(self[i])`.
    fn asin_in_place(&mut self);

    /// `atan(self[i])` in radians, in `[−π/2, π/2]`. Error ≤ 1 ULP.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![1.0_f64, 0.0, f64::NEG_INFINITY];
    /// assert_eq!(a.atan(), [core::f64::consts::FRAC_PI_4, 0.0, -core::f64::consts::FRAC_PI_2]);
    /// ```
    #[cfg(feature = "alloc")]
    fn atan(&self) -> Vec<T>;

    /// `self[i] = atan(self[i])`.
    fn atan_in_place(&mut self);

    /// `√self[i]`, correctly rounded (0 ULP) on every backend. `√-0.0` is
    /// `-0.0`; negative values and NaN give NaN.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![4.0_f32, 2.0, -1.0];
    /// let r = a.sqrt();
    /// assert_eq!(r[..2], [2.0, 2.0_f32.sqrt()]);
    /// assert!(r[2].is_nan());
    /// ```
    #[cfg(feature = "alloc")]
    fn sqrt(&self) -> Vec<T>;

    /// `self[i] = √self[i]`.
    fn sqrt_in_place(&mut self);
}

macro_rules! impl_slice_math {
    ($($T:ty),*) => {
        $(
            impl SliceMath<$T> for [$T] {
                #[cfg(feature = "alloc")]
                #[inline]
                fn abs(&self) -> Vec<$T> {
                    unary_to_vec::<AbsOp, _>(self)
                }

                #[inline]
                fn abs_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<AbsOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn acos(&self) -> Vec<$T> {
                    unary_to_vec::<AcosOp, _>(self)
                }

                #[inline]
                fn acos_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<AcosOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn asin(&self) -> Vec<$T> {
                    unary_to_vec::<AsinOp, _>(self)
                }

                #[inline]
                fn asin_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<AsinOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn atan(&self) -> Vec<$T> {
                    unary_to_vec::<AtanOp, _>(self)
                }

                #[inline]
                fn atan_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<AtanOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn sqrt(&self) -> Vec<$T> {
                    unary_to_vec::<SqrtOp, _>(self)
                }

                #[inline]
                fn sqrt_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<SqrtOp>(self);
                }
            }
        )*
    };
}

impl_slice_math!(f32, f64);

#[cfg(feature = "alloc")]
fn unary_to_vec<F: UnaryOp, T: MathElement>(x: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(x.len());
    T::unary_uninit::<F>(x, &mut out.spare_capacity_mut()[..x.len()]);
    // SAFETY: `unary_uninit` initialised the first `x.len()` elements.
    unsafe { out.set_len(x.len()) };
    out
}
