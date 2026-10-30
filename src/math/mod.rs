//! Element-wise mathematical functions, through [`SliceMath`].
//!
//! Each function is one file implementing a lane-level trait (`Abs`, `Acos`,
//! …) for every lane type (`f32`, `f64` and the SIMD registers), and a
//! zero-sized [`UnaryOp`] names it, so every `(function, element, backend)`
//! triple compiles to its own inlined loop over the [`driver`].
//!
//! A single instruction ([`abs`], [`ceil`], [`floor`], [`sqrt`]) is implemented per lane type. An
//! algorithm ([`acos`], [`asin`], [`atan`], [`atan2`], [`cbrt`], [`exp`], [`hypot`] (and `hypot3`), [`ln`], [`trig`]) is written once, generic over [`Float`], and each
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
mod cbrt;
mod ceil;
mod exp;
mod float;
mod floor;
mod hypot;
mod ln;
mod sqrt;
pub(crate) mod trig;

cfg_simd! {
    pub(crate) mod driver;
}

use abs::Abs;
use acos::Acos;
use asin::Asin;
use atan::Atan;
use atan2::Atan2;
use cbrt::Cbrt;
use ceil::Ceil;
use exp::Exp;
use float::Float;
use floor::Floor;
use hypot::{Hypot, Hypot3};
use ln::Ln;
use sqrt::Sqrt;
use trig::Trig;

/// A lane type with every math function: a float or a SIMD register of
/// floats.
pub(crate) trait Math:
    Float + Acos + Asin + Atan + Atan2 + Cbrt + Ceil + Exp + Hypot + Hypot3 + Ln + Trig
{
}

impl<T: Float + Acos + Asin + Atan + Atan2 + Cbrt + Ceil + Exp + Hypot + Hypot3 + Ln + Trig> Math
    for T
{
}

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

/// An element-wise math function of three arguments, `f(x, y, z)`.
pub(crate) trait TernaryOp {
    /// Applies the function lane-wise.
    fn apply<V: Math>(x: V, y: V, z: V) -> V;
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

    impl<F: TernaryOp, V: Math> driver::TernaryFn<V> for F {
        #[inline(always)]
        fn call(x: V, y: V, z: V) -> V {
            F::apply(x, y, z)
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

/// `∛x`.
pub(crate) struct CbrtOp;

impl UnaryOp for CbrtOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.cbrt()
    }
}

/// `eˣ`.
pub(crate) struct ExpOp;

impl UnaryOp for ExpOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.exp()
    }
}

/// `⌈x⌉`.
pub(crate) struct CeilOp;

impl UnaryOp for CeilOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.ceil()
    }
}

/// `⌊x⌋`.
pub(crate) struct FloorOp;

impl UnaryOp for FloorOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.floor()
    }
}

/// `√(x² + y² + z²)`.
pub(crate) struct Hypot3Op;

impl TernaryOp for Hypot3Op {
    #[inline(always)]
    fn apply<V: Math>(x: V, y: V, z: V) -> V {
        x.hypot3(y, z)
    }
}

/// `√(x² + y²)`.
pub(crate) struct HypotOp;

impl BinaryOp for HypotOp {
    #[inline(always)]
    fn apply<V: Math>(x: V, y: V) -> V {
        x.hypot(y)
    }
}

/// `ln x`.
pub(crate) struct LnOp;

impl UnaryOp for LnOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.ln()
    }
}

/// `cos x`.
pub(crate) struct CosOp;

impl UnaryOp for CosOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.cos()
    }
}

/// `tan x`.
pub(crate) struct TanOp;

impl UnaryOp for TanOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.tan()
    }
}

/// `sin x`.
pub(crate) struct SinOp;

impl UnaryOp for SinOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.sin()
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

    /// `out[i] = F(x[i], y[i], z[i])`; on return all of `out` is
    /// initialised.
    ///
    /// # Panics
    ///
    /// If `x`, `y`, `z` and `out` do not all have the same length.
    #[cfg(feature = "alloc")]
    fn ternary_uninit<F: TernaryOp>(
        x: &[Self],
        y: &[Self],
        z: &[Self],
        out: &mut [MaybeUninit<Self>],
    );

    /// `x[i] = F(x[i], y[i], z[i])`.
    ///
    /// # Panics
    ///
    /// If `x`, `y` and `z` do not all have the same length.
    fn ternary_in_place<F: TernaryOp>(x: &mut [Self], y: &[Self], z: &[Self]);

    /// [`every_backend`](Self::every_backend) for a [`TernaryOp`].
    #[cfg(test)]
    fn every_backend_ternary<F: TernaryOp>(
        x: &[Self],
        y: &[Self],
        z: &[Self],
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

                #[cfg(feature = "alloc")]
                #[inline]
                fn ternary_uninit<F: TernaryOp>(
                    x: &[Self],
                    y: &[Self],
                    z: &[Self],
                    out: &mut [MaybeUninit<Self>],
                ) {
                    assert_eq!(x.len(), y.len(), "length mismatch");
                    assert_eq!(x.len(), z.len(), "length mismatch");
                    assert_eq!(x.len(), out.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_ternary_uninit::<x86::$avx512, F>(x, y, z, out),
                        avx2 => x86::avx2_ternary_uninit::<x86::$avx2, F>(x, y, z, out),
                        neon => neon::ternary_uninit::<neon::$neon, F>(x, y, z, out),
                        scalar => scalar_ternary_uninit::<F, Self>(x, y, z, out),
                    }
                }

                #[inline]
                fn ternary_in_place<F: TernaryOp>(x: &mut [Self], y: &[Self], z: &[Self]) {
                    assert_eq!(x.len(), y.len(), "length mismatch");
                    assert_eq!(x.len(), z.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_ternary_in_place::<x86::$avx512, F>(x, y, z),
                        avx2 => x86::avx2_ternary_in_place::<x86::$avx2, F>(x, y, z),
                        neon => neon::ternary_in_place::<neon::$neon, F>(x, y, z),
                        scalar => scalar_ternary_in_place::<F, Self>(x, y, z),
                    }
                }

                #[cfg(test)]
                fn every_backend_ternary<F: TernaryOp>(
                    x: &[Self],
                    y: &[Self],
                    z: &[Self],
                ) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)> {
                    let mut r = x.to_vec();
                    scalar_ternary_in_place::<F, Self>(&mut r, y, z);
                    #[allow(unused_mut, reason = "only scalar on some targets")]
                    let mut results = std::vec![("scalar", r)];
                    cfg_x86!(if {
                        if crate::detect::has_avx2_fma() {
                            let mut r = x.to_vec();
                            // SAFETY: AVX2 + FMA checked just above.
                            unsafe { x86::avx2_ternary_in_place::<x86::$avx2, F>(&mut r, y, z) };
                            results.push(("avx2", r));
                        }
                        if crate::detect::has_avx512f() {
                            let mut r = x.to_vec();
                            // SAFETY: AVX-512F checked just above.
                            unsafe { x86::avx512_ternary_in_place::<x86::$avx512, F>(&mut r, y, z) };
                            results.push(("avx512", r));
                        }
                    } else {});
                    cfg_neon!(if {
                        let mut r = x.to_vec();
                        neon::ternary_in_place::<neon::$neon, F>(&mut r, y, z);
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

#[cfg(feature = "alloc")]
fn scalar_ternary_uninit<F: TernaryOp, T: Math>(
    x: &[T],
    y: &[T],
    z: &[T],
    out: &mut [MaybeUninit<T>],
) {
    for (((o, &a), &b), &c) in out.iter_mut().zip(x).zip(y).zip(z) {
        o.write(F::apply(a, b, c));
    }
}

fn scalar_ternary_in_place<F: TernaryOp, T: Math>(x: &mut [T], y: &[T], z: &[T]) {
    for ((a, &b), &c) in x.iter_mut().zip(y).zip(z) {
        *a = F::apply(*a, b, c);
    }
}

cfg_x86! {
    mod x86 {
        use super::{Binary, BinaryOp, Math, TernaryOp, UnaryOp, driver};
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

        #[cfg(feature = "alloc")]
        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_ternary_uninit<V: SimdVector + Math, F: TernaryOp>(
            x: &[V::Scalar],
            y: &[V::Scalar],
            z: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::ternary_uninit::<V, F>(x, y, z, out) }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_ternary_in_place<V: SimdVector + Math, F: TernaryOp>(
            x: &mut [V::Scalar],
            y: &[V::Scalar],
            z: &[V::Scalar],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::ternary_in_place::<V, F>(x, y, z) }
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

        #[cfg(feature = "alloc")]
        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_ternary_uninit<V: SimdVector + Math, F: TernaryOp>(
            x: &[V::Scalar],
            y: &[V::Scalar],
            z: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::ternary_uninit::<V, F>(x, y, z, out) }
        }

        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_ternary_in_place<V: SimdVector + Math, F: TernaryOp>(
            x: &mut [V::Scalar],
            y: &[V::Scalar],
            z: &[V::Scalar],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::ternary_in_place::<V, F>(x, y, z) }
        }
    }
}

cfg_neon! {
    mod neon {
        use super::{Binary, BinaryOp, Math, TernaryOp, UnaryOp, driver};
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

        #[cfg(feature = "alloc")]
        #[inline]
        pub(super) fn ternary_uninit<V: SimdVector + Math, F: TernaryOp>(
            x: &[V::Scalar],
            y: &[V::Scalar],
            z: &[V::Scalar],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::ternary_uninit::<V, F>(x, y, z, out) }
        }

        #[inline]
        pub(super) fn ternary_in_place<V: SimdVector + Math, F: TernaryOp>(
            x: &mut [V::Scalar],
            y: &[V::Scalar],
            z: &[V::Scalar],
        ) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::ternary_in_place::<V, F>(x, y, z) }
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

    /// `∛self[i]`, defined for negative values too (`∛−8 = −2`). Error ≤ 1
    /// ULP; `±0`, `±∞` and NaN are returned as they are.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![27.0_f32, -8.0, 0.125];
    /// assert_eq!(a.cbrt(), [3.0, -2.0, 0.5]);
    /// ```
    #[cfg(feature = "alloc")]
    fn cbrt(&self) -> Vec<T>;

    /// `self[i] = ∛self[i]`.
    fn cbrt_in_place(&mut self);

    /// `cos self[i]`, in radians. Error ≤ 1 ULP for every finite input, huge
    /// ones included; NaN for `±∞`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let c = vec![0.0_f64, -0.0, core::f64::consts::PI].cos();
    /// assert_eq!(c, [1.0, 1.0, -1.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn cos(&self) -> Vec<T>;

    /// `self[i] = cos self[i]`.
    fn cos_in_place(&mut self);

    /// `e^self[i]`. Error ≤ 1 ULP; `+∞` above `ln(MAX)`, `+0` below the
    /// underflow threshold, subnormal results in between.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let r = vec![0.0_f64, f64::NEG_INFINITY, 1.0].exp();
    /// assert_eq!(r[..2], [1.0, 0.0]);
    /// // Within 1 ULP of e (here 0.67 ULP: the float just above `E`).
    /// let e = core::f64::consts::E;
    /// assert!((r[2] - e).abs() <= f64::EPSILON * 2.0);
    /// ```
    #[cfg(feature = "alloc")]
    fn exp(&self) -> Vec<T>;

    /// `self[i] = e^self[i]`.
    fn exp_in_place(&mut self);

    /// `⌈self[i]⌉`, the smallest integer not below it. Exact: `⌈−0.3⌉` is
    /// `−0.0`, `⌈0.3⌉` is `1.0`, and `±∞` and NaN are returned as they are.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![2.3_f32, -2.7, -0.3];
    /// let c = a.ceil();
    /// assert_eq!(c, [3.0, -2.0, -0.0]);
    /// assert!(c[2].is_sign_negative());
    /// ```
    #[cfg(feature = "alloc")]
    fn ceil(&self) -> Vec<T>;

    /// `self[i] = ⌈self[i]⌉`.
    fn ceil_in_place(&mut self);

    /// `⌊self[i]⌋`, the largest integer not above it. Exact: `⌊−0.0⌋` is
    /// `−0.0`, `⌊−0.3⌋` is `−1.0`, and `±∞` and NaN are returned as they are.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![2.7_f32, -2.7, -0.0];
    /// let f = a.floor();
    /// assert_eq!(f, [2.0, -3.0, -0.0]);
    /// assert!(f[2].is_sign_negative());
    /// ```
    #[cfg(feature = "alloc")]
    fn floor(&self) -> Vec<T>;

    /// `self[i] = ⌊self[i]⌋`.
    fn floor_in_place(&mut self);

    /// `√(self[i]² + y[i]²)`, without overflow or underflow in between, as
    /// `f32::hypot`. Error ≤ 1 ULP; `+∞` if either is infinite, even when the
    /// other is NaN.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let x = vec![3.0_f32, 1e30, f32::INFINITY];
    /// let y = vec![4.0_f32, 1e30, f32::NAN];
    /// let h = x.hypot(&y);
    /// assert_eq!(h[0], 5.0);
    /// assert!(h[1].is_finite()); // x² would overflow
    /// assert_eq!(h[2], f32::INFINITY);
    /// ```
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    #[cfg(feature = "alloc")]
    fn hypot(&self, y: &[T]) -> Vec<T>;

    /// `self[i] = √(self[i]² + y[i]²)`.
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    fn hypot_in_place(&mut self, y: &[T]);

    /// `√(self[i]² + y[i]² + z[i]²)`, without overflow or underflow in
    /// between. Error ≤ 1 ULP; `+∞` if any is infinite, even when another
    /// is NaN.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let x = vec![2.0_f32, 1e30, f32::INFINITY];
    /// let y = vec![3.0_f32, 1e30, 1.0];
    /// let z = vec![6.0_f32, 1e30, f32::NAN];
    /// let h = x.hypot3(&y, &z);
    /// assert_eq!(h[0], 7.0);
    /// assert!(h[1].is_finite()); // x² would overflow
    /// assert_eq!(h[2], f32::INFINITY);
    /// ```
    ///
    /// # Panics
    ///
    /// If `self`, `y` and `z` do not all have the same length.
    #[cfg(feature = "alloc")]
    fn hypot3(&self, y: &[T], z: &[T]) -> Vec<T>;

    /// `self[i] = √(self[i]² + y[i]² + z[i]²)`.
    ///
    /// # Panics
    ///
    /// If `self`, `y` and `z` do not all have the same length.
    fn hypot3_in_place(&mut self, y: &[T], z: &[T]);

    /// `ln self[i]`, the natural logarithm. Error ≤ 1 ULP; `ln(±0) = −∞`,
    /// NaN for negative values, `ln(+∞) = +∞`, `ln(1) = 0` exactly.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![1.0_f64, 0.0, 2.0];
    /// assert_eq!(a.ln(), [0.0, f64::NEG_INFINITY, core::f64::consts::LN_2]);
    /// ```
    #[cfg(feature = "alloc")]
    fn ln(&self) -> Vec<T>;

    /// `self[i] = ln self[i]`.
    fn ln_in_place(&mut self);

    /// `sin self[i]`, in radians. Error ≤ 1 ULP for every finite input, huge
    /// ones included; NaN for `±∞`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let s = vec![0.0_f64, core::f64::consts::FRAC_PI_2, -0.0].sin();
    /// assert_eq!(s, [0.0, 1.0, -0.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn sin(&self) -> Vec<T>;

    /// `self[i] = sin self[i]`.
    fn sin_in_place(&mut self);

    /// `tan self[i]`, in radians. Error ≤ 1 ULP for every finite input, huge
    /// ones included; NaN for `±∞`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// // FRAC_PI_4 is just below π/4: its tangent rounds to just below 1.
    /// let t = vec![0.0_f64, -0.0, core::f64::consts::FRAC_PI_4].tan();
    /// assert_eq!(t, [0.0, -0.0, 0.999_999_999_999_999_9]);
    /// ```
    #[cfg(feature = "alloc")]
    fn tan(&self) -> Vec<T>;

    /// `self[i] = tan self[i]`.
    fn tan_in_place(&mut self);

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
                fn cbrt(&self) -> Vec<$T> {
                    unary_to_vec::<CbrtOp, _>(self)
                }

                #[inline]
                fn cbrt_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<CbrtOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn cos(&self) -> Vec<$T> {
                    unary_to_vec::<CosOp, _>(self)
                }

                #[inline]
                fn cos_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<CosOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn exp(&self) -> Vec<$T> {
                    unary_to_vec::<ExpOp, _>(self)
                }

                #[inline]
                fn exp_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<ExpOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn ceil(&self) -> Vec<$T> {
                    unary_to_vec::<CeilOp, _>(self)
                }

                #[inline]
                fn ceil_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<CeilOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn floor(&self) -> Vec<$T> {
                    unary_to_vec::<FloorOp, _>(self)
                }

                #[inline]
                fn floor_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<FloorOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn hypot(&self, y: &[$T]) -> Vec<$T> {
                    binary_to_vec::<HypotOp, _>(self, y)
                }

                #[inline]
                fn hypot_in_place(&mut self, y: &[$T]) {
                    <$T as MathElement>::binary_in_place::<HypotOp>(self, y);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn hypot3(&self, y: &[$T], z: &[$T]) -> Vec<$T> {
                    ternary_to_vec::<Hypot3Op, _>(self, y, z)
                }

                #[inline]
                fn hypot3_in_place(&mut self, y: &[$T], z: &[$T]) {
                    <$T as MathElement>::ternary_in_place::<Hypot3Op>(self, y, z);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn ln(&self) -> Vec<$T> {
                    unary_to_vec::<LnOp, _>(self)
                }

                #[inline]
                fn ln_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<LnOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn sin(&self) -> Vec<$T> {
                    unary_to_vec::<SinOp, _>(self)
                }

                #[inline]
                fn sin_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<SinOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn tan(&self) -> Vec<$T> {
                    unary_to_vec::<TanOp, _>(self)
                }

                #[inline]
                fn tan_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<TanOp>(self);
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

#[cfg(feature = "alloc")]
fn ternary_to_vec<F: TernaryOp, T: MathElement>(x: &[T], y: &[T], z: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(x.len());
    T::ternary_uninit::<F>(x, y, z, &mut out.spare_capacity_mut()[..x.len()]);
    // SAFETY: `ternary_uninit` initialised the first `x.len()` elements.
    unsafe { out.set_len(x.len()) };
    out
}
