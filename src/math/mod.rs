//! Element-wise mathematical functions, through [`SliceMath`].
//!
//! Each function is one file implementing a lane-level trait (`Abs`, `Acos`,
//! …) for every lane type (`f32`, `f64` and the SIMD registers), and a
//! zero-sized [`UnaryOp`] names it, so every `(function, element, backend)`
//! triple compiles to its own inlined loop over the [`driver`].
//!
//! A single instruction ([`abs`], [`sqrt`]) is implemented per lane type. An
//! algorithm ([`acos`], [`asin`], [`atan`], [`atan2`]) is written once, generic over [`Float`], and each
//! backend runs it with its own instructions.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(any(feature = "alloc", test))]
use core::mem::MaybeUninit;

mod abs;
mod acos;
mod asin;
mod atan;
mod atan2;
mod float;
mod sqrt;

cfg_simd! {
    pub(crate) mod driver;
}

use abs::Abs;
use acos::Acos;
use asin::Asin;
use atan::Atan;
use atan2::Atan2;
use float::Float;
use sqrt::Sqrt;

/// A lane type with every math function: a float or a SIMD register of
/// floats.
pub(crate) trait Math: Float + Acos + Asin + Atan + Atan2 {}

impl<T: Float + Acos + Asin + Atan + Atan2> Math for T {}

/// An element-wise math function.
pub(crate) trait UnaryOp {
    /// Applies the function lane-wise.
    fn apply<V: Math>(v: V) -> V;
}

/// An element-wise math function of two arguments, `f(y, x)`.
pub(crate) trait BinaryOp {
    /// Applies the function lane-wise.
    fn apply<V: Math>(y: V, x: V) -> V;
}

cfg_simd! {
    impl<F: UnaryOp, V: Math> driver::UnaryFn<V> for F {
        #[inline(always)]
        fn call(x: V) -> V {
            F::apply(x)
        }
    }

    /// A [`BinaryOp`] as a driver operation. A wrapper, as the arithmetic
    /// operations already implement `BinaryFn` for every type of theirs.
    pub(crate) struct Binary<F>(core::marker::PhantomData<F>);

    impl<F: BinaryOp, V: Math> driver::BinaryFn<V> for Binary<F> {
        #[inline(always)]
        fn call(y: V, x: V) -> V {
            F::apply(y, x)
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

/// `atan2(y, x)`.
pub(crate) struct Atan2Op;

impl BinaryOp for Atan2Op {
    #[inline(always)]
    fn apply<V: Math>(y: V, x: V) -> V {
        y.atan2(x)
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

    /// `out[i] = F(y[i], x[i])`; on return all of `out` is initialised.
    ///
    /// # Panics
    ///
    /// If `y`, `x` and `out` do not all have the same length.
    #[cfg(any(feature = "alloc", test))]
    fn binary_uninit<F: BinaryOp>(y: &[Self], x: &[Self], out: &mut [MaybeUninit<Self>]);

    /// `y[i] = F(y[i], x[i])`.
    ///
    /// # Panics
    ///
    /// If `y` and `x` have different lengths.
    fn binary_in_place<F: BinaryOp>(y: &mut [Self], x: &[Self]);

    /// [`every_backend`](Self::every_backend) for a [`BinaryOp`].
    #[cfg(test)]
    fn every_backend_binary<F: BinaryOp>(
        y: &[Self],
        x: &[Self],
    ) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)>;
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

                #[cfg(any(feature = "alloc", test))]
                #[inline]
                fn binary_uninit<F: BinaryOp>(y: &[Self], x: &[Self], out: &mut [MaybeUninit<Self>]) {
                    assert_eq!(y.len(), x.len(), "length mismatch");
                    assert_eq!(y.len(), out.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_binary_uninit::<x86::$avx512, F>(y, x, out),
                        avx2 => x86::avx2_binary_uninit::<x86::$avx2, F>(y, x, out),
                        neon => neon::binary_uninit::<neon::$neon, F>(y, x, out),
                        scalar => scalar_binary_uninit::<F, Self>(y, x, out),
                    }
                }

                #[inline]
                fn binary_in_place<F: BinaryOp>(y: &mut [Self], x: &[Self]) {
                    assert_eq!(y.len(), x.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_binary_in_place::<x86::$avx512, F>(y, x),
                        avx2 => x86::avx2_binary_in_place::<x86::$avx2, F>(y, x),
                        neon => neon::binary_in_place::<neon::$neon, F>(y, x),
                        scalar => scalar_binary_in_place::<F, Self>(y, x),
                    }
                }

                #[cfg(test)]
                fn every_backend_binary<F: BinaryOp>(
                    y: &[Self],
                    x: &[Self],
                ) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)> {
                    let mut r = y.to_vec();
                    scalar_binary_in_place::<F, Self>(&mut r, x);
                    #[allow(unused_mut, reason = "only scalar on some targets")]
                    let mut results = std::vec![("scalar", r)];
                    cfg_x86!(if {
                        if crate::detect::has_avx2_fma() {
                            let mut r = y.to_vec();
                            // SAFETY: AVX2 + FMA checked just above.
                            unsafe { x86::avx2_binary_in_place::<x86::$avx2, F>(&mut r, x) };
                            results.push(("avx2", r));
                        }
                        if crate::detect::has_avx512f() {
                            let mut r = y.to_vec();
                            // SAFETY: AVX-512F checked just above.
                            unsafe { x86::avx512_binary_in_place::<x86::$avx512, F>(&mut r, x) };
                            results.push(("avx512", r));
                        }
                    } else {});
                    cfg_neon!(if {
                        let mut r = y.to_vec();
                        neon::binary_in_place::<neon::$neon, F>(&mut r, x);
                        results.push(("neon", r));
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

#[cfg(any(feature = "alloc", test))]
fn scalar_binary_uninit<F: BinaryOp, T: Math>(y: &[T], x: &[T], out: &mut [MaybeUninit<T>]) {
    for ((o, &a), &b) in out.iter_mut().zip(y).zip(x) {
        o.write(F::apply(a, b));
    }
}

fn scalar_binary_in_place<F: BinaryOp, T: Math>(y: &mut [T], x: &[T]) {
    for (a, &b) in y.iter_mut().zip(x) {
        *a = F::apply(*a, b);
    }
}

cfg_x86! {
    mod x86 {
        use super::{Binary, BinaryOp, Math, UnaryOp, driver};
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

        #[cfg(any(feature = "alloc", test))]
        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_binary_uninit<V: SimdVector + Math, F: BinaryOp>(
            y: &[V::Scalar],
            x: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::binary_uninit::<V, Binary<F>>(y, x, out) }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_binary_in_place<V: SimdVector + Math, F: BinaryOp>(
            y: &mut [V::Scalar],
            x: &[V::Scalar],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::binary_in_place::<V, Binary<F>>(y, x) }
        }

        #[cfg(any(feature = "alloc", test))]
        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_binary_uninit<V: SimdVector + Math, F: BinaryOp>(
            y: &[V::Scalar],
            x: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::binary_uninit::<V, Binary<F>>(y, x, out) }
        }

        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_binary_in_place<V: SimdVector + Math, F: BinaryOp>(
            y: &mut [V::Scalar],
            x: &[V::Scalar],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::binary_in_place::<V, Binary<F>>(y, x) }
        }
    }
}

cfg_neon! {
    mod neon {
        use super::{Binary, BinaryOp, Math, UnaryOp, driver};
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

        #[cfg(any(feature = "alloc", test))]
        #[inline]
        pub(super) fn binary_uninit<V: SimdVector + Math, F: BinaryOp>(
            y: &[V::Scalar],
            x: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::binary_uninit::<V, Binary<F>>(y, x, out) }
        }

        #[inline]
        pub(super) fn binary_in_place<V: SimdVector + Math, F: BinaryOp>(
            y: &mut [V::Scalar],
            x: &[V::Scalar],
        ) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::binary_in_place::<V, Binary<F>>(y, x) }
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

    /// `atan2(self[i], x[i])`: the angle of the point `(x[i], self[i])`, in
    /// `[−π, π]`, as `f32::atan2`. Error ≤ 1 ULP; signed zeros and
    /// infinities follow C99 (`atan2(±0, −0) = ±π`, `atan2(±∞, ±∞) = ±π/4`
    /// or `±3π/4`, …).
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let y = vec![1.0_f64, 1.0, -1.0];
    /// let x = vec![1.0_f64, -1.0, 0.0];
    /// assert_eq!(y.atan2(&x), [core::f64::consts::FRAC_PI_4, 3.0 * core::f64::consts::FRAC_PI_4, -core::f64::consts::FRAC_PI_2]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `self` and `x` have different lengths.
    #[cfg(feature = "alloc")]
    fn atan2(&self, x: &[T]) -> Vec<T>;

    /// `self[i] = atan2(self[i], x[i])`.
    ///
    /// # Panics
    ///
    /// If `self` and `x` have different lengths.
    fn atan2_in_place(&mut self, x: &[T]);

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
                fn atan2(&self, x: &[$T]) -> Vec<$T> {
                    binary_to_vec::<Atan2Op, _>(self, x)
                }

                #[inline]
                fn atan2_in_place(&mut self, x: &[$T]) {
                    <$T as MathElement>::binary_in_place::<Atan2Op>(self, x);
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

#[cfg(feature = "alloc")]
fn binary_to_vec<F: BinaryOp, T: MathElement>(y: &[T], x: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(y.len());
    T::binary_uninit::<F>(y, x, &mut out.spare_capacity_mut()[..y.len()]);
    // SAFETY: `binary_uninit` initialised the first `y.len()` elements.
    unsafe { out.set_len(y.len()) };
    out
}
