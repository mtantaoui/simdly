//! Element-wise mathematical functions, through [`SliceMath`].
//!
//! Each function is one file implementing a lane-level trait (`Abs`, `Acos`,
//! …) for every lane type (`f32`, `f64` and the SIMD registers), and a
//! zero-sized [`UnaryOp`] names it, so every `(function, element, backend)`
//! triple compiles to its own inlined loop over the [`driver`].
//!
//! A single instruction ([`abs`], [`ceil`], [`floor`], [`sqrt`]) is implemented per lane type. An
//! algorithm ([`acos`], [`asin`], [`atan`], [`atan2`], [`cbrt`], [`erf`], [`exp`], [`hypot`] (and `hypot3`, `hypot4`), [`hyperbolic`] (`sinh`, `cosh`, `tanh`), [`inverse_hyperbolic`] (`atanh`, `asinh`, `acosh`), [`ln`], [`log1p`], [`pow`] (`powf`, `powi`), [`sigmoid`], [`softplus`] (and `log_sigmoid`), [`trig`]) is written once, generic over [`Float`], and each
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
mod basic;
mod cbrt;
mod ceil;
mod erf;
mod exp;
mod float;
mod floor;
mod fmod;
mod hyperbolic;
mod hypot;
mod inverse_hyperbolic;
mod ln;
mod log1p;
#[cfg(feature = "rayon")]
pub(crate) mod par;
mod pow;
mod round;
mod sigmoid;
mod softplus;
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
use erf::Erf;
use exp::Exp;
use float::Float;
use floor::Floor;
use fmod::Fmod;
use hyperbolic::{Cosh, Sinh, Tanh};
use hypot::{Hypot, Norm};
use inverse_hyperbolic::{Acosh, Asinh, Atanh};
use ln::Ln;
use log1p::Log1p;
use pow::Pow;
use round::{Round, RoundTiesEven, Trunc};
use sigmoid::Sigmoid;
use softplus::{LogSigmoid, Softplus};
use sqrt::Sqrt;
use trig::Trig;

/// A lane type with every math function: a float or a SIMD register of
/// floats.
pub(crate) trait Math:
    Float
    + Acos
    + Asin
    + Atan
    + Atan2
    + Cbrt
    + Ceil
    + Exp
    + Hypot
    + Norm
    + Ln
    + Log1p
    + LogSigmoid
    + Sigmoid
    + Softplus
    + Sinh
    + Cosh
    + Tanh
    + Atanh
    + Asinh
    + Acosh
    + Erf
    + Pow
    + Trunc
    + RoundTiesEven
    + Round
    + Fmod
    + Trig
{
}

impl<
    T: Float
        + Acos
        + Asin
        + Atan
        + Atan2
        + Cbrt
        + Ceil
        + Exp
        + Hypot
        + Norm
        + Ln
        + Log1p
        + LogSigmoid
        + Sigmoid
        + Softplus
        + Sinh
        + Cosh
        + Tanh
        + Atanh
        + Asinh
        + Acosh
        + Erf
        + Pow
        + Trunc
        + RoundTiesEven
        + Round
        + Fmod
        + Trig,
> Math for T
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

/// An element-wise math function of `N` arguments.
pub(crate) trait NaryOp<const N: usize> {
    /// Applies the function lane-wise.
    fn apply<V: Math>(v: [V; N]) -> V;
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

    impl<F: NaryOp<N>, V: Math, const N: usize> driver::NaryFn<V, N> for F {
        #[inline(always)]
        fn call(v: [V; N]) -> V {
            F::apply(v)
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

/// `xʸ`, element-wise.
pub(crate) struct PowOp;

impl BinaryOp for PowOp {
    #[inline(always)]
    fn apply<V: Math>(x: V, y: V) -> V {
        let zero = x.constant(<V::Elem as float::Element>::from_f32(0.0));
        x.pow(y, zero)
    }
}

/// `xʸ` with `y = yh + yl` the same in every lane: `powf_scalar` (`yl = 0`)
/// and `powi`, whose exponent `f32` may not hold exactly.
pub(crate) struct PowScalarOp;

impl NaryOp<3> for PowScalarOp {
    #[inline(always)]
    fn apply<V: Math>([x, yh, yl]: [V; 3]) -> V {
        x.pow(yh, yl)
    }
}

/// `x^N` for a small integer `N`, by binary powering.
pub(crate) struct PowiOp<const N: i32>;

impl<const N: i32> UnaryOp for PowiOp<N> {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.powi_small::<N>()
    }
}

/// `x[i] = x[i]^n` with [`PowiOp`] for `n` in `3..=8` and `−8..=−2`;
/// `false`, `x` untouched, otherwise.
fn powi_small_in_place<T: MathElement>(x: &mut [T], n: i32) -> bool {
    macro_rules! small {
        ($($n:literal)*) => {
            match n {
                $($n => T::unary_in_place::<PowiOp<$n>>(x),)*
                _ => return false,
            }
        };
    }
    small!(3 4 5 6 7 8 -2 -3 -4 -5 -6 -7 -8);
    true
}

/// [`powi_small_in_place`] into a new vector.
#[cfg(feature = "alloc")]
fn powi_small_to_vec<T: MathElement>(x: &[T], n: i32) -> Option<Vec<T>> {
    macro_rules! small {
        ($($n:literal)*) => {
            match n {
                $($n => Some(unary_to_vec::<PowiOp<$n>, T>(x)),)*
                _ => None,
            }
        };
    }
    small!(3 4 5 6 7 8 -2 -3 -4 -5 -6 -7 -8)
}

/// `x` rounded toward zero.
pub(crate) struct TruncOp;

impl UnaryOp for TruncOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.trunc()
    }
}

/// `x` rounded to nearest, halves away from zero.
pub(crate) struct RoundOp;

impl UnaryOp for RoundOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.round()
    }
}

/// `x` rounded to nearest, halves to even.
pub(crate) struct RoundTiesEvenOp;

impl UnaryOp for RoundTiesEvenOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.round_ties_even()
    }
}

/// `−x`.
pub(crate) struct NegOp;

impl UnaryOp for NegOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        -v
    }
}

/// `1/x`.
pub(crate) struct RecipOp;

impl UnaryOp for RecipOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.constant(<V::Elem as float::Element>::from_f32(1.0)) / v
    }
}

/// `−1`, `0` or `1`.
pub(crate) struct SignOp;

impl UnaryOp for SignOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        basic::sign(v)
    }
}

/// `x` clamped to `[min, max]`, the bounds the same in every lane.
pub(crate) struct ClampOp;

impl NaryOp<3> for ClampOp {
    #[inline(always)]
    fn apply<V: Math>([x, min, max]: [V; 3]) -> V {
        basic::clamp(x, min, max)
    }
}

/// `x % y`, element-wise.
pub(crate) struct FmodOp;

impl BinaryOp for FmodOp {
    #[inline(always)]
    fn apply<V: Math>(x: V, y: V) -> V {
        x.fmod(y)
    }
}

/// Python's `x % y`, element-wise.
pub(crate) struct RemainderOp;

impl BinaryOp for RemainderOp {
    #[inline(always)]
    fn apply<V: Math>(x: V, y: V) -> V {
        x.remainder(y)
    }
}

/// `x % y`, `y` the same in every lane (the third operand is unused).
pub(crate) struct FmodScalarOp;

impl NaryOp<3> for FmodScalarOp {
    #[inline(always)]
    fn apply<V: Math>([x, y, _]: [V; 3]) -> V {
        x.fmod(y)
    }
}

/// Python's `x % y`, `y` the same in every lane (the third operand is
/// unused).
pub(crate) struct RemainderScalarOp;

impl NaryOp<3> for RemainderScalarOp {
    #[inline(always)]
    fn apply<V: Math>([x, y, _]: [V; 3]) -> V {
        x.remainder(y)
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

impl NaryOp<3> for Hypot3Op {
    #[inline(always)]
    fn apply<V: Math>(v: [V; 3]) -> V {
        V::norm(v)
    }
}

/// `√(x² + y² + z² + w²)`.
pub(crate) struct Hypot4Op;

impl NaryOp<4> for Hypot4Op {
    #[inline(always)]
    fn apply<V: Math>(v: [V; 4]) -> V {
        V::norm(v)
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

/// `ln(1 + x)`.
pub(crate) struct Log1pOp;

impl UnaryOp for Log1pOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.log1p()
    }
}

/// `σ(x) = 1/(1 + e⁻ˣ)`.
pub(crate) struct SigmoidOp;

impl UnaryOp for SigmoidOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.sigmoid()
    }
}

/// `ln σ(x) = −ln(1 + e⁻ˣ)`.
pub(crate) struct LogSigmoidOp;

impl UnaryOp for LogSigmoidOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.log_sigmoid()
    }
}

/// `ln(1 + eˣ)`.
pub(crate) struct SoftplusOp;

impl UnaryOp for SoftplusOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.softplus()
    }
}

/// `sinh x`.
pub(crate) struct SinhOp;

impl UnaryOp for SinhOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.sinh()
    }
}

/// `cosh x`.
pub(crate) struct CoshOp;

impl UnaryOp for CoshOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.cosh()
    }
}

/// `tanh x`.
pub(crate) struct TanhOp;

impl UnaryOp for TanhOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.tanh()
    }
}

/// `atanh x`.
pub(crate) struct AtanhOp;

impl UnaryOp for AtanhOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.atanh()
    }
}

/// `asinh x`.
pub(crate) struct AsinhOp;

impl UnaryOp for AsinhOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.asinh()
    }
}

/// `acosh x`.
pub(crate) struct AcoshOp;

impl UnaryOp for AcoshOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.acosh()
    }
}

/// `erf x`.
pub(crate) struct ErfOp;

impl UnaryOp for ErfOp {
    #[inline(always)]
    fn apply<V: Math>(v: V) -> V {
        v.erf()
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

    /// `out[i] = F([src[0][i], .., src[N-1][i]])`; on return all of `out`
    /// is initialised.
    ///
    /// # Panics
    ///
    /// If the slices and `out` do not all have the same length.
    #[cfg(feature = "alloc")]
    fn nary_uninit<F: NaryOp<N>, const N: usize>(src: [&[Self]; N], out: &mut [MaybeUninit<Self>]);

    /// `x[i] = F([x[i], rest[0][i], .., rest[M-1][i]])`, with `N = M + 1`.
    ///
    /// # Panics
    ///
    /// If the slices do not all have the same length.
    fn nary_in_place<F: NaryOp<N>, const N: usize, const M: usize>(
        x: &mut [Self],
        rest: [&[Self]; M],
    );

    /// [`every_backend`](Self::every_backend) for a [`NaryOp`].
    #[cfg(test)]
    fn every_backend_nary<F: NaryOp<N>, const N: usize, const M: usize>(
        x: &[Self],
        rest: [&[Self]; M],
    ) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)>;

    /// `out[i] = F([x[i], b[0], b[1]])`; on return all of `out` is
    /// initialised.
    ///
    /// # Panics
    ///
    /// If `x` and `out` have different lengths.
    #[cfg(feature = "alloc")]
    fn scalars_uninit<F: NaryOp<3>>(x: &[Self], b: [Self; 2], out: &mut [MaybeUninit<Self>]);

    /// `x[i] = F([x[i], b[0], b[1]])`.
    fn scalars_in_place<F: NaryOp<3>>(x: &mut [Self], b: [Self; 2]);

    /// [`every_backend`](Self::every_backend) for [`scalars_in_place`](Self::scalars_in_place).
    #[cfg(test)]
    fn every_backend_scalars<F: NaryOp<3>>(
        x: &[Self],
        b: [Self; 2],
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
                fn nary_uninit<F: NaryOp<N>, const N: usize>(
                    src: [&[Self]; N],
                    out: &mut [MaybeUninit<Self>],
                ) {
                    for x in src {
                        assert_eq!(x.len(), out.len(), "length mismatch");
                    }
                    dispatch! {
                        avx512 => x86::avx512_nary_uninit::<x86::$avx512, F, N>(src, out),
                        avx2 => x86::avx2_nary_uninit::<x86::$avx2, F, N>(src, out),
                        neon => neon::nary_uninit::<neon::$neon, F, N>(src, out),
                        scalar => scalar_nary_uninit::<F, Self, N>(src, out),
                    }
                }

                #[inline]
                fn nary_in_place<F: NaryOp<N>, const N: usize, const M: usize>(
                    x: &mut [Self],
                    rest: [&[Self]; M],
                ) {
                    const { assert!(M + 1 == N, "N must be M + 1") };
                    for r in rest {
                        assert_eq!(x.len(), r.len(), "length mismatch");
                    }
                    dispatch! {
                        avx512 => x86::avx512_nary_in_place::<x86::$avx512, F, N, M>(x, rest),
                        avx2 => x86::avx2_nary_in_place::<x86::$avx2, F, N, M>(x, rest),
                        neon => neon::nary_in_place::<neon::$neon, F, N, M>(x, rest),
                        scalar => scalar_nary_in_place::<F, Self, N, M>(x, rest),
                    }
                }

                #[cfg(test)]
                fn every_backend_nary<F: NaryOp<N>, const N: usize, const M: usize>(
                    x: &[Self],
                    rest: [&[Self]; M],
                ) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)> {
                    let mut r = x.to_vec();
                    scalar_nary_in_place::<F, Self, N, M>(&mut r, rest);
                    #[allow(unused_mut, reason = "only scalar on some targets")]
                    let mut results = std::vec![("scalar", r)];
                    cfg_x86!(if {
                        if crate::detect::has_avx2_fma() {
                            let mut r = x.to_vec();
                            // SAFETY: AVX2 + FMA checked just above.
                            unsafe { x86::avx2_nary_in_place::<x86::$avx2, F, N, M>(&mut r, rest) };
                            results.push(("avx2", r));
                        }
                        if crate::detect::has_avx512f() {
                            let mut r = x.to_vec();
                            // SAFETY: AVX-512F checked just above.
                            unsafe { x86::avx512_nary_in_place::<x86::$avx512, F, N, M>(&mut r, rest) };
                            results.push(("avx512", r));
                        }
                    } else {});
                    cfg_neon!(if {
                        let mut r = x.to_vec();
                        neon::nary_in_place::<neon::$neon, F, N, M>(&mut r, rest);
                        results.push(("neon", r));
                    } else {});
                    results
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn scalars_uninit<F: NaryOp<3>>(x: &[Self], b: [Self; 2], out: &mut [MaybeUninit<Self>]) {
                    assert_eq!(x.len(), out.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_scalars_uninit::<x86::$avx512, F>(x, b, out),
                        avx2 => x86::avx2_scalars_uninit::<x86::$avx2, F>(x, b, out),
                        neon => neon::scalars_uninit::<neon::$neon, F>(x, b, out),
                        scalar => scalar_scalars_uninit::<F, Self>(x, b, out),
                    }
                }

                #[inline]
                fn scalars_in_place<F: NaryOp<3>>(x: &mut [Self], b: [Self; 2]) {
                    dispatch! {
                        avx512 => x86::avx512_scalars_in_place::<x86::$avx512, F>(x, b),
                        avx2 => x86::avx2_scalars_in_place::<x86::$avx2, F>(x, b),
                        neon => neon::scalars_in_place::<neon::$neon, F>(x, b),
                        scalar => scalar_scalars_in_place::<F, Self>(x, b),
                    }
                }

                #[cfg(test)]
                fn every_backend_scalars<F: NaryOp<3>>(
                    x: &[Self],
                    b: [Self; 2],
                ) -> std::vec::Vec<(&'static str, std::vec::Vec<Self>)> {
                    let mut r = x.to_vec();
                    scalar_scalars_in_place::<F, Self>(&mut r, b);
                    #[allow(unused_mut, reason = "only scalar on some targets")]
                    let mut results = std::vec![("scalar", r)];
                    cfg_x86!(if {
                        if crate::detect::has_avx2_fma() {
                            let mut r = x.to_vec();
                            // SAFETY: AVX2 + FMA checked just above.
                            unsafe { x86::avx2_scalars_in_place::<x86::$avx2, F>(&mut r, b) };
                            results.push(("avx2", r));
                        }
                        if crate::detect::has_avx512f() {
                            let mut r = x.to_vec();
                            // SAFETY: AVX-512F checked just above.
                            unsafe { x86::avx512_scalars_in_place::<x86::$avx512, F>(&mut r, b) };
                            results.push(("avx512", r));
                        }
                    } else {});
                    cfg_neon!(if {
                        let mut r = x.to_vec();
                        neon::scalars_in_place::<neon::$neon, F>(&mut r, b);
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
fn scalar_nary_uninit<F: NaryOp<N>, T: Math, const N: usize>(
    src: [&[T]; N],
    out: &mut [MaybeUninit<T>],
) {
    for (i, o) in out.iter_mut().enumerate() {
        o.write(F::apply(src.map(|x| x[i])));
    }
}

#[cfg(feature = "alloc")]
fn scalar_scalars_uninit<F: NaryOp<3>, T: Math>(
    x: &[T],
    [b0, b1]: [T; 2],
    out: &mut [MaybeUninit<T>],
) {
    for (o, &v) in out.iter_mut().zip(x) {
        o.write(F::apply([v, b0, b1]));
    }
}

fn scalar_scalars_in_place<F: NaryOp<3>, T: Math>(x: &mut [T], [b0, b1]: [T; 2]) {
    for v in x {
        *v = F::apply([*v, b0, b1]);
    }
}

fn scalar_nary_in_place<F: NaryOp<N>, T: Math, const N: usize, const M: usize>(
    x: &mut [T],
    rest: [&[T]; M],
) {
    for (i, a) in x.iter_mut().enumerate() {
        let mut v = [*a; N];
        for (k, r) in rest.iter().enumerate() {
            v[k + 1] = r[i];
        }
        *a = F::apply(v);
    }
}

cfg_x86! {
    mod x86 {
        use super::{Binary, BinaryOp, Math, NaryOp, UnaryOp, driver};
        use crate::simd::SimdVector;
        pub(super) use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        pub(super) use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};

        #[cfg(feature = "alloc")]
        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_scalars_uninit<V: SimdVector + Math, F: NaryOp<3>>(
            x: &[V::Scalar],
            b: [V::Scalar; 2],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::scalars_uninit::<V, F>(x, b, out) }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_scalars_in_place<V: SimdVector + Math, F: NaryOp<3>>(
            x: &mut [V::Scalar],
            b: [V::Scalar; 2],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::scalars_in_place::<V, F>(x, b) }
        }

        #[cfg(feature = "alloc")]
        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_scalars_uninit<V: SimdVector + Math, F: NaryOp<3>>(
            x: &[V::Scalar],
            b: [V::Scalar; 2],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::scalars_uninit::<V, F>(x, b, out) }
        }

        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_scalars_in_place<V: SimdVector + Math, F: NaryOp<3>>(
            x: &mut [V::Scalar],
            b: [V::Scalar; 2],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::scalars_in_place::<V, F>(x, b) }
        }

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
        pub(super) fn avx2_nary_uninit<V: SimdVector + Math, F: NaryOp<N>, const N: usize>(
            src: [&[V::Scalar]; N],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::nary_uninit::<V, F, N>(src, out) }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_nary_in_place<
            V: SimdVector + Math,
            F: NaryOp<N>,
            const N: usize,
            const M: usize,
        >(
            x: &mut [V::Scalar],
            rest: [&[V::Scalar]; M],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, and `V` is an
            // AVX2 register type.
            unsafe { driver::nary_in_place::<V, F, N, M>(x, rest) }
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
        pub(super) fn avx512_nary_uninit<V: SimdVector + Math, F: NaryOp<N>, const N: usize>(
            src: [&[V::Scalar]; N],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::nary_uninit::<V, F, N>(src, out) }
        }

        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_nary_in_place<
            V: SimdVector + Math,
            F: NaryOp<N>,
            const N: usize,
            const M: usize,
        >(
            x: &mut [V::Scalar],
            rest: [&[V::Scalar]; M],
        ) {
            // SAFETY: AVX-512F is enabled in this function, and `V` is an
            // AVX-512 register type.
            unsafe { driver::nary_in_place::<V, F, N, M>(x, rest) }
        }
    }
}

cfg_neon! {
    mod neon {
        use super::{Binary, BinaryOp, Math, NaryOp, UnaryOp, driver};
        use crate::simd::SimdVector;
        pub(super) use crate::simd::neon::{f32x4::F32x4, f64x2::F64x2};

        #[cfg(feature = "alloc")]
        #[inline]
        pub(super) fn scalars_uninit<V: SimdVector + Math, F: NaryOp<3>>(
            x: &[V::Scalar],
            b: [V::Scalar; 2],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::scalars_uninit::<V, F>(x, b, out) }
        }

        #[inline]
        pub(super) fn scalars_in_place<V: SimdVector + Math, F: NaryOp<3>>(
            x: &mut [V::Scalar],
            b: [V::Scalar; 2],
        ) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::scalars_in_place::<V, F>(x, b) }
        }

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
        pub(super) fn nary_uninit<V: SimdVector + Math, F: NaryOp<N>, const N: usize>(
            src: [&[V::Scalar]; N],
            out: &mut [core::mem::MaybeUninit<V::Scalar>],
        ) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::nary_uninit::<V, F, N>(src, out) }
        }

        #[inline]
        pub(super) fn nary_in_place<
            V: SimdVector + Math,
            F: NaryOp<N>,
            const N: usize,
            const M: usize,
        >(
            x: &mut [V::Scalar],
            rest: [&[V::Scalar]; M],
        ) {
            // SAFETY: NEON is enabled at compile time, and `V` is a NEON
            // register type.
            unsafe { driver::nary_in_place::<V, F, N, M>(x, rest) }
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

    /// `self[i]^y[i]`. Error ≤ 1 ULP, overflow and subnormal results
    /// included; C99's special cases, as `f64::powf`: `x⁰ = 1` and `1ʸ = 1`
    /// even for NaN, a negative `x` gives NaN unless `y` is an integer,
    /// whose parity then sets the sign.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let x = vec![2.0_f64, -2.0, -8.0, 0.0, f64::NAN];
    /// let y = vec![0.5, 3.0, 1.0 / 3.0, -1.0, 0.0];
    /// let p = x.powf(&y);
    /// assert_eq!(p[..2], [core::f64::consts::SQRT_2, -8.0]);
    /// assert!(p[2].is_nan()); // (−8)^(1/3): the exponent is not an integer
    /// assert_eq!(p[3..], [f64::INFINITY, 1.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    #[cfg(feature = "alloc")]
    fn powf(&self, y: &[T]) -> Vec<T>;

    /// `self[i] = self[i]^y[i]`.
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    fn powf_in_place(&mut self, y: &[T]);

    /// `self[i]^y`, one exponent for every element: [`powf`](Self::powf)
    /// without a slice of `y`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let x = vec![4.0_f32, 9.0, 0.0, -8.0];
    /// let r = x.powf_scalar(0.5);
    /// assert_eq!(r[..3], [2.0, 3.0, 0.0]);
    /// assert!(r[3].is_nan());
    /// ```
    #[cfg(feature = "alloc")]
    fn powf_scalar(&self, y: T) -> Vec<T>;

    /// `self[i] = self[i]^y`.
    fn powf_scalar_in_place(&mut self, y: T);

    /// `self[i] % y[i]`, as Rust's `%` and C's `fmod`: `x − n·y` with
    /// `n = trunc(x/y)`, exact, with the sign of `x`. `x % ±0`, `±∞ % y`
    /// and NaN give NaN; `x % ±∞ = x` for finite `x`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let x = vec![7.5_f64, -7.5, 1e300, -6.0];
    /// let y = vec![2.0, 2.0, 3.0, 3.0];
    /// assert_eq!(x.fmod(&y), [1.5, -1.5, 1e300 % 3.0, -0.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    #[cfg(feature = "alloc")]
    fn fmod(&self, y: &[T]) -> Vec<T>;

    /// `self[i] = self[i] % y[i]`.
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    fn fmod_in_place(&mut self, y: &[T]);

    /// `self[i] % y`, one divisor for every element: [`fmod`](Self::fmod).
    #[cfg(feature = "alloc")]
    fn fmod_scalar(&self, y: T) -> Vec<T>;

    /// `self[i] = self[i] % y`.
    fn fmod_scalar_in_place(&mut self, y: T);

    /// Python's and PyTorch's `self[i] % y[i]`, burn's `remainder`: the
    /// sign of `y`, `x − ⌊x/y⌋·y`. Computed as burn does, `r = x % y` (exact,
    /// [`fmod`](Self::fmod)), then `r + y` where `r ≠ 0` and the signs of `r`
    /// and `y` differ; that sum rounds.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let x = vec![7.5_f32, -7.5, 7.5, -6.0];
    /// let y = vec![2.0, 2.0, -2.0, 3.0];
    /// assert_eq!(x.remainder(&y), [1.5, 0.5, -0.5, -0.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    #[cfg(feature = "alloc")]
    fn remainder(&self, y: &[T]) -> Vec<T>;

    /// `self[i] = remainder(self[i], y[i])`.
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    fn remainder_in_place(&mut self, y: &[T]);

    /// [`remainder`](Self::remainder) with one divisor for every element.
    #[cfg(feature = "alloc")]
    fn remainder_scalar(&self, y: T) -> Vec<T>;

    /// `self[i] = remainder(self[i], y)`.
    fn remainder_scalar_in_place(&mut self, y: T);

    /// `self[i]^n` for an integer `n`. Error ≤ 1 ULP for every `n`, where
    /// repeated multiplication (as `f32::powi`) loses about `log₂|n|` ULPs:
    /// for `|n| ≤ 8`, binary powering in two-part arithmetic, rounded once;
    /// beyond, [`powf`](Self::powf)'s `e^(n·ln x)`, `n` exact even past
    /// `f32`'s 2²⁴. `n` = 0, 1, 2 and −1 give `1`, `x`, `x·x` and `1/x`,
    /// each rounded once.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let x = vec![2.0_f64, -2.0, 0.0, 1.0 + f64::EPSILON];
    /// let r = x.powi(-3);
    /// assert_eq!(r[..3], [0.125, -0.125, f64::INFINITY]);
    /// // (1 + 2⁻²³)^(2²⁴ + 1) ≈ e²: the exponent is not an f32, the result is.
    /// let r = [1.0_f32 + f32::EPSILON].powi(16_777_217);
    /// assert!((r[0] - 7.389_056).abs() <= 7.389_056 * f32::EPSILON);
    /// ```
    #[cfg(feature = "alloc")]
    fn powi(&self, n: i32) -> Vec<T>;

    /// `self[i] = self[i]^n`.
    fn powi_in_place(&mut self, n: i32);

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

    /// `self[i]` rounded toward zero, as `f32::trunc`: exact, signed zeros
    /// kept (`trunc(−0.7) = −0.0`); `±∞` and NaN as they are.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![2.7_f64, -2.7, -0.3, f64::INFINITY];
    /// assert_eq!(a.trunc(), [2.0, -2.0, -0.0, f64::INFINITY]);
    /// ```
    #[cfg(feature = "alloc")]
    fn trunc(&self) -> Vec<T>;

    /// `self[i] = trunc(self[i])`.
    fn trunc_in_place(&mut self);

    /// `self[i]` rounded to nearest, halves away from zero, as `f32::round`:
    /// exact, signed zeros kept; `±∞` and NaN as they are.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![2.5_f32, -2.5, 0.5, 1.4999999];
    /// assert_eq!(a.round(), [3.0, -3.0, 1.0, 1.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn round(&self) -> Vec<T>;

    /// `self[i] = round(self[i])`.
    fn round_in_place(&mut self);

    /// `self[i]` rounded to nearest, halves to even, as
    /// `f32::round_ties_even` (and burn's `round`): exact, signed zeros kept;
    /// `±∞` and NaN as they are.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![2.5_f32, -2.5, 3.5, -0.5];
    /// assert_eq!(a.round_ties_even(), [2.0, -2.0, 4.0, -0.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn round_ties_even(&self) -> Vec<T>;

    /// `self[i] = round_ties_even(self[i])`.
    fn round_ties_even_in_place(&mut self);

    /// `−self[i]`: the sign bit flipped, NaN included.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![1.5_f64, -0.0];
    /// assert_eq!(a.neg(), [-1.5, 0.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn neg(&self) -> Vec<T>;

    /// `self[i] = −self[i]`.
    fn neg_in_place(&mut self);

    /// `1/self[i]`, rounded once, as `1.0 / x`: `±∞` for `±0`, `±0` for
    /// `±∞`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![4.0_f32, -0.0, f32::INFINITY];
    /// assert_eq!(a.recip(), [0.25, f32::NEG_INFINITY, 0.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn recip(&self) -> Vec<T>;

    /// `self[i] = 1/self[i]`.
    fn recip_in_place(&mut self);

    /// The sign of `self[i]`, as burn's and PyTorch's: `−1`, `0` or `1`;
    /// `0` for `±0` and, as burn's, for NaN.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![-3.0_f64, -0.0, 0.0, 2.0, f64::NAN];
    /// assert_eq!(a.sign(), [-1.0, 0.0, 0.0, 1.0, 0.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn sign(&self) -> Vec<T>;

    /// `self[i] = sign(self[i])`.
    fn sign_in_place(&mut self);

    /// `self[i]` clamped to `[min, max]`, as `f32::clamp`: `min` below
    /// `min`, `max` above `max`, `self[i]` otherwise, NaN included.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![-2.0_f32, 0.5, 7.0, f32::NAN];
    /// let c = a.clamp(0.0, 1.0);
    /// assert_eq!(c[..3], [0.0, 0.5, 1.0]);
    /// assert!(c[3].is_nan());
    /// ```
    ///
    /// # Panics
    ///
    /// If `min > max`, or either is NaN, as `f32::clamp`; checked once, not
    /// per element.
    #[cfg(feature = "alloc")]
    fn clamp(&self, min: T, max: T) -> Vec<T>;

    /// `self[i] = clamp(self[i], min, max)`.
    ///
    /// # Panics
    ///
    /// As [`clamp`](Self::clamp).
    fn clamp_in_place(&mut self, min: T, max: T);

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

    /// `√(self[i]² + y[i]² + z[i]² + w[i]²)`, as [`hypot3`](Self::hypot3):
    /// the norm of a quaternion or a 4-vector.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let x = vec![1.0_f64, 1e300, -0.0];
    /// let y = vec![2.0_f64, 1e300, f64::NAN];
    /// let z = vec![4.0_f64, 1e300, f64::NEG_INFINITY];
    /// let w = vec![10.0_f64, 1e300, 0.0];
    /// let h = x.hypot4(&y, &z, &w);
    /// assert_eq!(h[0], 11.0);
    /// assert_eq!(h[1], 2e300); // x² would overflow
    /// assert_eq!(h[2], f64::INFINITY);
    /// ```
    ///
    /// # Panics
    ///
    /// If `self`, `y`, `z` and `w` do not all have the same length.
    #[cfg(feature = "alloc")]
    fn hypot4(&self, y: &[T], z: &[T], w: &[T]) -> Vec<T>;

    /// `self[i] = √(self[i]² + y[i]² + z[i]² + w[i]²)`.
    ///
    /// # Panics
    ///
    /// If `self`, `y`, `z` and `w` do not all have the same length.
    fn hypot4_in_place(&mut self, y: &[T], z: &[T], w: &[T]);

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

    /// `ln(1 + self[i])`, accurate where `1 + self[i]` would round: near 0,
    /// `log1p(x) ≈ x`. Error ≤ 1 ULP; `log1p(±0) = ±0`, `log1p(−1) = −∞`,
    /// NaN below `−1`, `log1p(+∞) = +∞`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f64, 1.0, -1.0, 1e-20];
    /// // `(1.0 + 1e-20).ln()` would be 0.
    /// assert_eq!(a.log1p(), [0.0, core::f64::consts::LN_2, f64::NEG_INFINITY, 1e-20]);
    /// ```
    #[cfg(feature = "alloc")]
    fn log1p(&self) -> Vec<T>;

    /// `self[i] = ln(1 + self[i])`.
    fn log1p_in_place(&mut self);

    /// `σ(self[i]) = 1/(1 + e^−self[i])`, the logistic function, in
    /// `[0, 1]`. Error ≤ 1 ULP, subnormal results included: `σ(x) ≈ eˣ`
    /// far below 0 rounds to `+0` only past `exp`'s underflow threshold.
    /// `σ(±0) = 0.5`, `σ(+∞) = 1`, `σ(−∞) = +0`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f32, f32::INFINITY, f32::NEG_INFINITY, 1.0];
    /// let s = a.sigmoid();
    /// assert_eq!(s[..3], [0.5, 1.0, 0.0]);
    /// assert_eq!(s[3], 1.0 / (1.0 + (-1.0_f32).exp()));
    /// ```
    #[cfg(feature = "alloc")]
    fn sigmoid(&self) -> Vec<T>;

    /// `self[i] = σ(self[i])`.
    fn sigmoid_in_place(&mut self);

    /// `ln σ(self[i]) = −ln(1 + e^−self[i])`, the log of the logistic
    /// function, in `[−∞, 0]`, without overflow or cancellation: `≈ x` far
    /// below 0, `≈ −e⁻ˣ` far above. Error ≤ 1 ULP;
    /// `log_sigmoid(0) = −ln 2`, `log_sigmoid(−∞) = −∞`,
    /// `log_sigmoid(+∞) = −0`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f64, -1000.0, 1000.0];
    /// // `σ(-1000.0).ln()` would be −∞.
    /// assert_eq!(a.log_sigmoid(), [-core::f64::consts::LN_2, -1000.0, -0.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn log_sigmoid(&self) -> Vec<T>;

    /// `self[i] = ln σ(self[i])`.
    fn log_sigmoid_in_place(&mut self);

    /// `softplus(self[i]) = ln(1 + e^self[i])`, a smooth `max(x, 0)`, in
    /// `[0, ∞]`, without overflow or cancellation: `≈ eˣ` far below 0, `x`
    /// far above. Error ≤ 1 ULP, subnormal results included;
    /// `softplus(0) = ln 2`, `softplus(−∞) = +0`, `softplus(+∞) = +∞`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f32, 1000.0, -1000.0];
    /// // `(1.0 + 1000.0_f32.exp()).ln()` would be ∞.
    /// assert_eq!(a.softplus(), [core::f32::consts::LN_2, 1000.0, 0.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn softplus(&self) -> Vec<T>;

    /// `self[i] = ln(1 + e^self[i])`.
    fn softplus_in_place(&mut self);

    /// `sinh self[i] = (e^self[i] − e^−self[i])/2`, the hyperbolic sine.
    /// Error ≤ 1 ULP, without cancellation near 0 or overflow between
    /// `eˣ`'s and `sinh`'s own: `sinh(±0) = ±0`, `sinh(±∞) = ±∞`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f32, -0.0, 1e-20, 89.0];
    /// // `(e^89 − e^−89)/2` would overflow at `e^89`.
    /// let s = a.sinh();
    /// assert_eq!(s[..3], [0.0, -0.0, 1e-20]);
    /// assert!(s[3].is_finite());
    /// ```
    #[cfg(feature = "alloc")]
    fn sinh(&self) -> Vec<T>;

    /// `self[i] = sinh self[i]`.
    fn sinh_in_place(&mut self);

    /// `cosh self[i] = (e^self[i] + e^−self[i])/2`, the hyperbolic cosine,
    /// in `[1, ∞]`. Error ≤ 1 ULP; `cosh(±0) = 1`, `cosh(±∞) = +∞`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f64, -0.0, f64::NEG_INFINITY];
    /// assert_eq!(a.cosh(), [1.0, 1.0, f64::INFINITY]);
    /// ```
    #[cfg(feature = "alloc")]
    fn cosh(&self) -> Vec<T>;

    /// `self[i] = cosh self[i]`.
    fn cosh_in_place(&mut self);

    /// `tanh self[i] = sinh x / cosh x`, the hyperbolic tangent, in
    /// `[−1, 1]`. Error ≤ 1 ULP, without cancellation near 0 or overflow
    /// far from it; `tanh(±0) = ±0`, `tanh(±∞) = ±1`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f32, -0.0, 1e-20, 100.0, f32::NEG_INFINITY];
    /// // `(e^100 − e^−100)/(e^100 + e^−100)` would be ∞/∞ = NaN.
    /// assert_eq!(a.tanh(), [0.0, -0.0, 1e-20, 1.0, -1.0]);
    /// ```
    #[cfg(feature = "alloc")]
    fn tanh(&self) -> Vec<T>;

    /// `self[i] = tanh self[i]`.
    fn tanh_in_place(&mut self);

    /// `atanh self[i] = ½·ln((1 + x)/(1 − x))`, the inverse hyperbolic
    /// tangent. Error ≤ 1 ULP; `atanh(±0) = ±0`, `atanh(±1) = ±∞`, NaN
    /// beyond `±1`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f64, -0.0, 1.0, -1.0, 2.0, 1e-20];
    /// let t = a.atanh();
    /// assert_eq!(t[..4], [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY]);
    /// assert!(t[4].is_nan());
    /// assert_eq!(t[5], 1e-20);
    /// ```
    #[cfg(feature = "alloc")]
    fn atanh(&self) -> Vec<T>;

    /// `self[i] = atanh self[i]`.
    fn atanh_in_place(&mut self);

    /// `asinh self[i] = ln(x + √(x² + 1))`, the inverse hyperbolic sine.
    /// Error ≤ 1 ULP, without cancellation for `x < 0` or overflow of `x²`;
    /// `asinh(±0) = ±0`, `asinh(±∞) = ±∞`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f32, -0.0, 1e-20, -1e30];
    /// let s = a.asinh();
    /// assert_eq!(s[..3], [0.0, -0.0, 1e-20]);
    /// // `(x + (x * x + 1.0).sqrt()).ln()` would be +∞: `x²` overflows.
    /// // asinh(−10³⁰) ≈ −ln(2·10³⁰).
    /// assert!((s[3] + 69.770_7).abs() < 1e-4);
    /// ```
    #[cfg(feature = "alloc")]
    fn asinh(&self) -> Vec<T>;

    /// `self[i] = asinh self[i]`.
    fn asinh_in_place(&mut self);

    /// `acosh self[i] = ln(x + √(x² − 1))`, the inverse hyperbolic cosine,
    /// for `x ≥ 1`. Error ≤ 1 ULP; `acosh(1) = +0`, `acosh(+∞) = +∞`, NaN
    /// below 1.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![1.0_f64, f64::INFINITY, 0.5];
    /// let c = a.acosh();
    /// assert_eq!(c[..2], [0.0, f64::INFINITY]);
    /// assert!(c[2].is_nan());
    /// ```
    #[cfg(feature = "alloc")]
    fn acosh(&self) -> Vec<T>;

    /// `self[i] = acosh self[i]`.
    fn acosh_in_place(&mut self);

    /// `erf self[i] = (2/√π)·∫₀ˣ e^−t² dt`, the error function, in
    /// `[−1, 1]`. Error ≤ 1 ULP; `erf(±0) = ±0`, `erf(±∞) = ±1`.
    ///
    /// ```
    /// use simdly::SliceMath;
    ///
    /// let a = vec![0.0_f64, -0.0, f64::INFINITY, -10.0, 1e-300];
    /// let e = a.erf();
    /// assert_eq!(e[..4], [0.0, -0.0, 1.0, -1.0]);
    /// // erf x ≈ (2/√π)·x near 0.
    /// assert_eq!(e[4], 1e-300 * core::f64::consts::FRAC_2_SQRT_PI);
    /// ```
    #[cfg(feature = "alloc")]
    fn erf(&self) -> Vec<T>;

    /// `self[i] = erf self[i]`.
    fn erf_in_place(&mut self);

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
                fn powf_in_place(&mut self, y: &[$T]) {
                    <$T as MathElement>::binary_in_place::<PowOp>(self, y);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn powf(&self, y: &[$T]) -> Vec<$T> {
                    binary_to_vec::<PowOp, _>(self, y)
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn powf_scalar(&self, y: $T) -> Vec<$T> {
                    scalars_to_vec::<PowScalarOp, _>(self, [y, 0.0])
                }

                #[inline]
                fn powf_scalar_in_place(&mut self, y: $T) {
                    <$T as MathElement>::scalars_in_place::<PowScalarOp>(self, [y, 0.0]);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn fmod(&self, y: &[$T]) -> Vec<$T> {
                    binary_to_vec::<FmodOp, _>(self, y)
                }

                #[inline]
                fn fmod_in_place(&mut self, y: &[$T]) {
                    <$T as MathElement>::binary_in_place::<FmodOp>(self, y);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn fmod_scalar(&self, y: $T) -> Vec<$T> {
                    scalars_to_vec::<FmodScalarOp, _>(self, [y, 0.0])
                }

                #[inline]
                fn fmod_scalar_in_place(&mut self, y: $T) {
                    <$T as MathElement>::scalars_in_place::<FmodScalarOp>(self, [y, 0.0]);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn remainder(&self, y: &[$T]) -> Vec<$T> {
                    binary_to_vec::<RemainderOp, _>(self, y)
                }

                #[inline]
                fn remainder_in_place(&mut self, y: &[$T]) {
                    <$T as MathElement>::binary_in_place::<RemainderOp>(self, y);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn remainder_scalar(&self, y: $T) -> Vec<$T> {
                    scalars_to_vec::<RemainderScalarOp, _>(self, [y, 0.0])
                }

                #[inline]
                fn remainder_scalar_in_place(&mut self, y: $T) {
                    <$T as MathElement>::scalars_in_place::<RemainderScalarOp>(self, [y, 0.0]);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn powi(&self, n: i32) -> Vec<$T> {
                    if let Some(out) = powi_small_to_vec(self, n) {
                        return out;
                    }
                    let mut out = self.to_vec();
                    out.powi_in_place(n);
                    out
                }

                #[inline]
                fn powi_in_place(&mut self, n: i32) {
                    match n {
                        0 => self.fill(1.0),
                        1 => {}
                        2 => self.iter_mut().for_each(|v| *v *= *v),
                        -1 => self.iter_mut().for_each(|v| *v = 1.0 / *v),
                        _ if powi_small_in_place(self, n) => {}
                        _ => {
                            // n = hi + lo, both exact; lo = 0 for f64, and
                            // |lo| ≤ 64 for f32.
                            let hi = n as $T;
                            let lo = (i64::from(n) - hi as i64) as $T;
                            <$T as MathElement>::scalars_in_place::<PowScalarOp>(self, [hi, lo]);
                        }
                    }
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
                fn trunc(&self) -> Vec<$T> {
                    unary_to_vec::<TruncOp, _>(self)
                }

                #[inline]
                fn trunc_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<TruncOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn round(&self) -> Vec<$T> {
                    unary_to_vec::<RoundOp, _>(self)
                }

                #[inline]
                fn round_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<RoundOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn round_ties_even(&self) -> Vec<$T> {
                    unary_to_vec::<RoundTiesEvenOp, _>(self)
                }

                #[inline]
                fn round_ties_even_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<RoundTiesEvenOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn neg(&self) -> Vec<$T> {
                    unary_to_vec::<NegOp, _>(self)
                }

                #[inline]
                fn neg_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<NegOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn recip(&self) -> Vec<$T> {
                    unary_to_vec::<RecipOp, _>(self)
                }

                #[inline]
                fn recip_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<RecipOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn sign(&self) -> Vec<$T> {
                    unary_to_vec::<SignOp, _>(self)
                }

                #[inline]
                fn sign_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<SignOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn clamp(&self, min: $T, max: $T) -> Vec<$T> {
                    check_clamp_bounds(min, max);
                    scalars_to_vec::<ClampOp, _>(self, [min, max])
                }

                #[inline]
                fn clamp_in_place(&mut self, min: $T, max: $T) {
                    check_clamp_bounds(min, max);
                    <$T as MathElement>::scalars_in_place::<ClampOp>(self, [min, max]);
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
                    nary_to_vec::<Hypot3Op, _, 3>([self, y, z])
                }

                #[inline]
                fn hypot3_in_place(&mut self, y: &[$T], z: &[$T]) {
                    <$T as MathElement>::nary_in_place::<Hypot3Op, 3, 2>(self, [y, z]);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn hypot4(&self, y: &[$T], z: &[$T], w: &[$T]) -> Vec<$T> {
                    nary_to_vec::<Hypot4Op, _, 4>([self, y, z, w])
                }

                #[inline]
                fn hypot4_in_place(&mut self, y: &[$T], z: &[$T], w: &[$T]) {
                    <$T as MathElement>::nary_in_place::<Hypot4Op, 4, 3>(self, [y, z, w]);
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
                fn log1p(&self) -> Vec<$T> {
                    unary_to_vec::<Log1pOp, _>(self)
                }

                #[inline]
                fn log1p_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<Log1pOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn sigmoid(&self) -> Vec<$T> {
                    unary_to_vec::<SigmoidOp, _>(self)
                }

                #[inline]
                fn sigmoid_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<SigmoidOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn log_sigmoid(&self) -> Vec<$T> {
                    unary_to_vec::<LogSigmoidOp, _>(self)
                }

                #[inline]
                fn log_sigmoid_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<LogSigmoidOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn softplus(&self) -> Vec<$T> {
                    unary_to_vec::<SoftplusOp, _>(self)
                }

                #[inline]
                fn softplus_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<SoftplusOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn sinh(&self) -> Vec<$T> {
                    unary_to_vec::<SinhOp, _>(self)
                }

                #[inline]
                fn sinh_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<SinhOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn cosh(&self) -> Vec<$T> {
                    unary_to_vec::<CoshOp, _>(self)
                }

                #[inline]
                fn cosh_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<CoshOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn tanh(&self) -> Vec<$T> {
                    unary_to_vec::<TanhOp, _>(self)
                }

                #[inline]
                fn tanh_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<TanhOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn atanh(&self) -> Vec<$T> {
                    unary_to_vec::<AtanhOp, _>(self)
                }

                #[inline]
                fn atanh_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<AtanhOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn asinh(&self) -> Vec<$T> {
                    unary_to_vec::<AsinhOp, _>(self)
                }

                #[inline]
                fn asinh_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<AsinhOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn acosh(&self) -> Vec<$T> {
                    unary_to_vec::<AcoshOp, _>(self)
                }

                #[inline]
                fn acosh_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<AcoshOp>(self);
                }

                #[cfg(feature = "alloc")]
                #[inline]
                fn erf(&self) -> Vec<$T> {
                    unary_to_vec::<ErfOp, _>(self)
                }

                #[inline]
                fn erf_in_place(&mut self) {
                    <$T as MathElement>::unary_in_place::<ErfOp>(self);
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

/// `f32::clamp`'s panic, once per slice: `min > max` or a NaN bound.
#[inline]
#[track_caller]
pub(crate) fn check_clamp_bounds<T: PartialOrd + core::fmt::Debug>(min: T, max: T) {
    assert!(
        min <= max,
        "clamp: min > max, or a bound is NaN: min = {min:?}, max = {max:?}"
    );
}

#[cfg(feature = "alloc")]
fn scalars_to_vec<F: NaryOp<3>, T: MathElement>(x: &[T], b: [T; 2]) -> Vec<T> {
    let mut out = Vec::with_capacity(x.len());
    T::scalars_uninit::<F>(x, b, &mut out.spare_capacity_mut()[..x.len()]);
    // SAFETY: `scalars_uninit` initialised the first `x.len()` elements.
    unsafe { out.set_len(x.len()) };
    out
}

#[cfg(feature = "alloc")]
fn nary_to_vec<F: NaryOp<N>, T: MathElement, const N: usize>(src: [&[T]; N]) -> Vec<T> {
    let len = src[0].len();
    let mut out = Vec::with_capacity(len);
    T::nary_uninit::<F, N>(src, &mut out.spare_capacity_mut()[..len]);
    // SAFETY: `nary_uninit` initialised the first `len` elements.
    unsafe { out.set_len(len) };
    out
}
