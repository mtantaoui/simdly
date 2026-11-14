//! [`ParSliceMath`]: [`SliceMath`](super::SliceMath) on several threads.
//!
//! A slice longer than [`PARALLEL_SIMD_THRESHOLD`] is cut into chunks of
//! [`PARALLEL_CHUNK_SIZE`] elements, and rayon runs the single-threaded
//! function on each. The chunk size is a multiple of every register width, so
//! only the last chunk has a masked tail, and the results are the same, bit
//! for bit, as the single-threaded ones.

use alloc::vec::Vec;

use rayon::prelude::*;

use super::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, Atan2Op, AtanOp, AtanhOp, BinaryOp, CbrtOp, CeilOp,
    CosOp, CoshOp, ErfOp, ExpOp, FloorOp, Hypot3Op, Hypot4Op, HypotOp, LnOp, Log1pOp, LogSigmoidOp,
    MathElement, NaryOp, PowOp, PowScalarOp, SigmoidOp, SinOp, SinhOp, SoftplusOp, SqrtOp, TanOp,
    TanhOp, UnaryOp, binary_to_vec, nary_to_vec, scalars_to_vec, unary_to_vec,
};
use crate::{PARALLEL_CHUNK_SIZE, PARALLEL_SIMD_THRESHOLD};

/// The math functions of [`SliceMath`](crate::SliceMath) on several threads,
/// for `[f32]` and `[f64]` (with the `rayon` feature).
///
/// Slices up to [`PARALLEL_SIMD_THRESHOLD`] elements run on the calling
/// thread, as `SliceMath` does: below it, starting threads costs more than
/// it saves. The results are the same, bit for bit, as `SliceMath`'s, and so
/// are the panics on slices of different lengths.
///
/// ```
/// use simdly::ParSliceMath;
///
/// let x: Vec<f32> = (0..1_000_000).map(|i| i as f32 * 1e-3).collect();
/// let s = x.par_sin();
/// assert_eq!(s[0], 0.0);
/// ```
pub trait ParSliceMath<T>: crate::ops::sealed::Sealed {
    /// [`SliceMath::abs`](crate::SliceMath::abs) on several threads.
    fn par_abs(&self) -> Vec<T>;
    /// [`SliceMath::abs_in_place`](crate::SliceMath::abs_in_place) on several threads.
    fn par_abs_in_place(&mut self);

    /// [`SliceMath::acos`](crate::SliceMath::acos) on several threads.
    fn par_acos(&self) -> Vec<T>;
    /// [`SliceMath::acos_in_place`](crate::SliceMath::acos_in_place) on several threads.
    fn par_acos_in_place(&mut self);

    /// [`SliceMath::asin`](crate::SliceMath::asin) on several threads.
    fn par_asin(&self) -> Vec<T>;
    /// [`SliceMath::asin_in_place`](crate::SliceMath::asin_in_place) on several threads.
    fn par_asin_in_place(&mut self);

    /// [`SliceMath::atan`](crate::SliceMath::atan) on several threads.
    fn par_atan(&self) -> Vec<T>;
    /// [`SliceMath::atan_in_place`](crate::SliceMath::atan_in_place) on several threads.
    fn par_atan_in_place(&mut self);

    /// [`SliceMath::atan2`](crate::SliceMath::atan2) on several threads.
    ///
    /// # Panics
    ///
    /// If `self` and `x` have different lengths.
    fn par_atan2(&self, x: &[T]) -> Vec<T>;
    /// [`SliceMath::atan2_in_place`](crate::SliceMath::atan2_in_place) on several threads.
    ///
    /// # Panics
    ///
    /// If `self` and `x` have different lengths.
    fn par_atan2_in_place(&mut self, x: &[T]);

    /// [`SliceMath::powf`](crate::SliceMath::powf) on several threads.
    fn par_powf(&self, y: &[T]) -> Vec<T>;
    /// [`SliceMath::powf_in_place`](crate::SliceMath::powf_in_place) on several threads.
    fn par_powf_in_place(&mut self, y: &[T]);

    /// [`SliceMath::powf_scalar`](crate::SliceMath::powf_scalar) on several threads.
    fn par_powf_scalar(&self, y: T) -> Vec<T>;
    /// [`SliceMath::powf_scalar_in_place`](crate::SliceMath::powf_scalar_in_place) on several threads.
    fn par_powf_scalar_in_place(&mut self, y: T);

    /// [`SliceMath::powi`](crate::SliceMath::powi) on several threads.
    fn par_powi(&self, n: i32) -> Vec<T>;
    /// [`SliceMath::powi_in_place`](crate::SliceMath::powi_in_place) on several threads.
    fn par_powi_in_place(&mut self, n: i32);

    /// [`SliceMath::cbrt`](crate::SliceMath::cbrt) on several threads.
    fn par_cbrt(&self) -> Vec<T>;
    /// [`SliceMath::cbrt_in_place`](crate::SliceMath::cbrt_in_place) on several threads.
    fn par_cbrt_in_place(&mut self);

    /// [`SliceMath::cos`](crate::SliceMath::cos) on several threads.
    fn par_cos(&self) -> Vec<T>;
    /// [`SliceMath::cos_in_place`](crate::SliceMath::cos_in_place) on several threads.
    fn par_cos_in_place(&mut self);

    /// [`SliceMath::exp`](crate::SliceMath::exp) on several threads.
    fn par_exp(&self) -> Vec<T>;
    /// [`SliceMath::exp_in_place`](crate::SliceMath::exp_in_place) on several threads.
    fn par_exp_in_place(&mut self);

    /// [`SliceMath::ceil`](crate::SliceMath::ceil) on several threads.
    fn par_ceil(&self) -> Vec<T>;
    /// [`SliceMath::ceil_in_place`](crate::SliceMath::ceil_in_place) on several threads.
    fn par_ceil_in_place(&mut self);

    /// [`SliceMath::floor`](crate::SliceMath::floor) on several threads.
    fn par_floor(&self) -> Vec<T>;
    /// [`SliceMath::floor_in_place`](crate::SliceMath::floor_in_place) on several threads.
    fn par_floor_in_place(&mut self);

    /// [`SliceMath::hypot`](crate::SliceMath::hypot) on several threads.
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    fn par_hypot(&self, y: &[T]) -> Vec<T>;
    /// [`SliceMath::hypot_in_place`](crate::SliceMath::hypot_in_place) on several threads.
    ///
    /// # Panics
    ///
    /// If `self` and `y` have different lengths.
    fn par_hypot_in_place(&mut self, y: &[T]);

    /// [`SliceMath::hypot3`](crate::SliceMath::hypot3) on several threads.
    ///
    /// # Panics
    ///
    /// If the slices do not all have the same length.
    fn par_hypot3(&self, y: &[T], z: &[T]) -> Vec<T>;
    /// [`SliceMath::hypot3_in_place`](crate::SliceMath::hypot3_in_place) on several threads.
    ///
    /// # Panics
    ///
    /// If the slices do not all have the same length.
    fn par_hypot3_in_place(&mut self, y: &[T], z: &[T]);

    /// [`SliceMath::hypot4`](crate::SliceMath::hypot4) on several threads.
    ///
    /// # Panics
    ///
    /// If the slices do not all have the same length.
    fn par_hypot4(&self, y: &[T], z: &[T], w: &[T]) -> Vec<T>;
    /// [`SliceMath::hypot4_in_place`](crate::SliceMath::hypot4_in_place) on several threads.
    ///
    /// # Panics
    ///
    /// If the slices do not all have the same length.
    fn par_hypot4_in_place(&mut self, y: &[T], z: &[T], w: &[T]);

    /// [`SliceMath::ln`](crate::SliceMath::ln) on several threads.
    fn par_ln(&self) -> Vec<T>;
    /// [`SliceMath::ln_in_place`](crate::SliceMath::ln_in_place) on several threads.
    fn par_ln_in_place(&mut self);

    /// [`SliceMath::log1p`](crate::SliceMath::log1p) on several threads.
    fn par_log1p(&self) -> Vec<T>;
    /// [`SliceMath::log1p_in_place`](crate::SliceMath::log1p_in_place) on several threads.
    fn par_log1p_in_place(&mut self);

    /// [`SliceMath::sigmoid`](crate::SliceMath::sigmoid) on several threads.
    fn par_sigmoid(&self) -> Vec<T>;
    /// [`SliceMath::sigmoid_in_place`](crate::SliceMath::sigmoid_in_place) on several threads.
    fn par_sigmoid_in_place(&mut self);

    /// [`SliceMath::log_sigmoid`](crate::SliceMath::log_sigmoid) on several threads.
    fn par_log_sigmoid(&self) -> Vec<T>;
    /// [`SliceMath::log_sigmoid_in_place`](crate::SliceMath::log_sigmoid_in_place) on several threads.
    fn par_log_sigmoid_in_place(&mut self);

    /// [`SliceMath::softplus`](crate::SliceMath::softplus) on several threads.
    fn par_softplus(&self) -> Vec<T>;
    /// [`SliceMath::softplus_in_place`](crate::SliceMath::softplus_in_place) on several threads.
    fn par_softplus_in_place(&mut self);

    /// [`SliceMath::sinh`](crate::SliceMath::sinh) on several threads.
    fn par_sinh(&self) -> Vec<T>;
    /// [`SliceMath::sinh_in_place`](crate::SliceMath::sinh_in_place) on several threads.
    fn par_sinh_in_place(&mut self);

    /// [`SliceMath::cosh`](crate::SliceMath::cosh) on several threads.
    fn par_cosh(&self) -> Vec<T>;
    /// [`SliceMath::cosh_in_place`](crate::SliceMath::cosh_in_place) on several threads.
    fn par_cosh_in_place(&mut self);

    /// [`SliceMath::tanh`](crate::SliceMath::tanh) on several threads.
    fn par_tanh(&self) -> Vec<T>;
    /// [`SliceMath::tanh_in_place`](crate::SliceMath::tanh_in_place) on several threads.
    fn par_tanh_in_place(&mut self);

    /// [`SliceMath::atanh`](crate::SliceMath::atanh) on several threads.
    fn par_atanh(&self) -> Vec<T>;
    /// [`SliceMath::atanh_in_place`](crate::SliceMath::atanh_in_place) on several threads.
    fn par_atanh_in_place(&mut self);

    /// [`SliceMath::asinh`](crate::SliceMath::asinh) on several threads.
    fn par_asinh(&self) -> Vec<T>;
    /// [`SliceMath::asinh_in_place`](crate::SliceMath::asinh_in_place) on several threads.
    fn par_asinh_in_place(&mut self);

    /// [`SliceMath::acosh`](crate::SliceMath::acosh) on several threads.
    fn par_acosh(&self) -> Vec<T>;
    /// [`SliceMath::acosh_in_place`](crate::SliceMath::acosh_in_place) on several threads.
    fn par_acosh_in_place(&mut self);

    /// [`SliceMath::erf`](crate::SliceMath::erf) on several threads.
    fn par_erf(&self) -> Vec<T>;
    /// [`SliceMath::erf_in_place`](crate::SliceMath::erf_in_place) on several threads.
    fn par_erf_in_place(&mut self);

    /// [`SliceMath::sin`](crate::SliceMath::sin) on several threads.
    fn par_sin(&self) -> Vec<T>;
    /// [`SliceMath::sin_in_place`](crate::SliceMath::sin_in_place) on several threads.
    fn par_sin_in_place(&mut self);

    /// [`SliceMath::tan`](crate::SliceMath::tan) on several threads.
    fn par_tan(&self) -> Vec<T>;
    /// [`SliceMath::tan_in_place`](crate::SliceMath::tan_in_place) on several threads.
    fn par_tan_in_place(&mut self);

    /// [`SliceMath::sqrt`](crate::SliceMath::sqrt) on several threads.
    fn par_sqrt(&self) -> Vec<T>;
    /// [`SliceMath::sqrt_in_place`](crate::SliceMath::sqrt_in_place) on several threads.
    fn par_sqrt_in_place(&mut self);
}

/// One `par_` pair per unary function: `(name, name_in_place, op)`.
macro_rules! par_unary {
    ($T:ty: $($f:ident $f_in_place:ident $Op:ident),* $(,)?) => {
        $(
            #[inline]
            fn $f(&self) -> Vec<$T> {
                par_unary_to_vec::<$Op, _>(self)
            }

            #[inline]
            fn $f_in_place(&mut self) {
                par_unary_in_place::<$Op, _>(self);
            }
        )*
    };
}

macro_rules! impl_par_slice_math {
    ($($T:ty),*) => {
        $(
            impl ParSliceMath<$T> for [$T] {
                par_unary!($T:
                    par_abs par_abs_in_place AbsOp,
                    par_acos par_acos_in_place AcosOp,
                    par_asin par_asin_in_place AsinOp,
                    par_atan par_atan_in_place AtanOp,
                    par_cbrt par_cbrt_in_place CbrtOp,
                    par_cos par_cos_in_place CosOp,
                    par_exp par_exp_in_place ExpOp,
                    par_ceil par_ceil_in_place CeilOp,
                    par_floor par_floor_in_place FloorOp,
                    par_ln par_ln_in_place LnOp,
                    par_log1p par_log1p_in_place Log1pOp,
                    par_sigmoid par_sigmoid_in_place SigmoidOp,
                    par_log_sigmoid par_log_sigmoid_in_place LogSigmoidOp,
                    par_softplus par_softplus_in_place SoftplusOp,
                    par_sinh par_sinh_in_place SinhOp,
                    par_cosh par_cosh_in_place CoshOp,
                    par_tanh par_tanh_in_place TanhOp,
                    par_atanh par_atanh_in_place AtanhOp,
                    par_asinh par_asinh_in_place AsinhOp,
                    par_acosh par_acosh_in_place AcoshOp,
                    par_erf par_erf_in_place ErfOp,
                    par_sin par_sin_in_place SinOp,
                    par_tan par_tan_in_place TanOp,
                    par_sqrt par_sqrt_in_place SqrtOp,
                );

                #[inline]
                fn par_atan2(&self, x: &[$T]) -> Vec<$T> {
                    par_binary_to_vec::<Atan2Op, _>(self, x)
                }

                #[inline]
                fn par_powf(&self, y: &[$T]) -> Vec<$T> {
                    par_binary_to_vec::<PowOp, _>(self, y)
                }

                #[inline]
                fn par_powf_in_place(&mut self, y: &[$T]) {
                    par_binary_in_place::<PowOp, _>(self, y);
                }

                #[inline]
                fn par_powf_scalar(&self, y: $T) -> Vec<$T> {
                    par_scalars_to_vec::<PowScalarOp, _>(self, [y, 0.0])
                }

                #[inline]
                fn par_powf_scalar_in_place(&mut self, y: $T) {
                    par_scalars_in_place::<PowScalarOp, _>(self, [y, 0.0]);
                }

                #[inline]
                fn par_powi(&self, n: i32) -> Vec<$T> {
                    if self.len() <= PARALLEL_SIMD_THRESHOLD {
                        return crate::SliceMath::powi(self, n);
                    }
                    let mut out = self.to_vec();
                    out.par_powi_in_place(n);
                    out
                }

                #[inline]
                fn par_powi_in_place(&mut self, n: i32) {
                    if self.len() <= PARALLEL_SIMD_THRESHOLD {
                        return crate::SliceMath::powi_in_place(self, n);
                    }
                    self.par_chunks_mut(PARALLEL_CHUNK_SIZE)
                        .for_each(|c| crate::SliceMath::powi_in_place(c, n));
                }

                #[inline]
                fn par_atan2_in_place(&mut self, x: &[$T]) {
                    par_binary_in_place::<Atan2Op, _>(self, x);
                }

                #[inline]
                fn par_hypot(&self, y: &[$T]) -> Vec<$T> {
                    par_binary_to_vec::<HypotOp, _>(self, y)
                }

                #[inline]
                fn par_hypot_in_place(&mut self, y: &[$T]) {
                    par_binary_in_place::<HypotOp, _>(self, y);
                }

                #[inline]
                fn par_hypot3(&self, y: &[$T], z: &[$T]) -> Vec<$T> {
                    par_nary_to_vec::<Hypot3Op, _, 3>([self, y, z])
                }

                #[inline]
                fn par_hypot3_in_place(&mut self, y: &[$T], z: &[$T]) {
                    par_nary_in_place::<Hypot3Op, _, 3, 2>(self, [y, z]);
                }

                #[inline]
                fn par_hypot4(&self, y: &[$T], z: &[$T], w: &[$T]) -> Vec<$T> {
                    par_nary_to_vec::<Hypot4Op, _, 4>([self, y, z, w])
                }

                #[inline]
                fn par_hypot4_in_place(&mut self, y: &[$T], z: &[$T], w: &[$T]) {
                    par_nary_in_place::<Hypot4Op, _, 4, 3>(self, [y, z, w]);
                }
            }
        )*
    };
}

impl_par_slice_math!(f32, f64);

// Each helper hands short slices, and slices of different lengths (for the
// panic), to the single-threaded function.

fn par_unary_to_vec<F: UnaryOp, T: MathElement + Send + Sync>(x: &[T]) -> Vec<T> {
    if x.len() <= PARALLEL_SIMD_THRESHOLD {
        return unary_to_vec::<F, T>(x);
    }
    let mut out = Vec::with_capacity(x.len());
    out.spare_capacity_mut()[..x.len()]
        .par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .zip(x.par_chunks(PARALLEL_CHUNK_SIZE))
        .for_each(|(out, x)| T::unary_uninit::<F>(x, out));
    // SAFETY: the chunks cover the first `x.len()` elements, and
    // `unary_uninit` initialised each of them.
    unsafe { out.set_len(x.len()) };
    out
}

fn par_unary_in_place<F: UnaryOp, T: MathElement + Send>(x: &mut [T]) {
    if x.len() <= PARALLEL_SIMD_THRESHOLD {
        return T::unary_in_place::<F>(x);
    }
    x.par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .for_each(T::unary_in_place::<F>);
}

fn par_binary_to_vec<F: BinaryOp, T: MathElement + Send + Sync>(y: &[T], x: &[T]) -> Vec<T> {
    if y.len() <= PARALLEL_SIMD_THRESHOLD || y.len() != x.len() {
        return binary_to_vec::<F, T>(y, x);
    }
    let mut out = Vec::with_capacity(y.len());
    out.spare_capacity_mut()[..y.len()]
        .par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .zip(y.par_chunks(PARALLEL_CHUNK_SIZE))
        .zip(x.par_chunks(PARALLEL_CHUNK_SIZE))
        .for_each(|((out, y), x)| T::binary_uninit::<F>(y, x, out));
    // SAFETY: the chunks cover the first `y.len()` elements, and
    // `binary_uninit` initialised each of them.
    unsafe { out.set_len(y.len()) };
    out
}

fn par_binary_in_place<F: BinaryOp, T: MathElement + Send + Sync>(y: &mut [T], x: &[T]) {
    if y.len() <= PARALLEL_SIMD_THRESHOLD || y.len() != x.len() {
        return T::binary_in_place::<F>(y, x);
    }
    y.par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .zip(x.par_chunks(PARALLEL_CHUNK_SIZE))
        .for_each(|(y, x)| T::binary_in_place::<F>(y, x));
}

fn par_scalars_to_vec<F: NaryOp<3>, T: MathElement + Send + Sync>(x: &[T], b: [T; 2]) -> Vec<T> {
    if x.len() <= PARALLEL_SIMD_THRESHOLD {
        return scalars_to_vec::<F, T>(x, b);
    }
    let mut out = Vec::with_capacity(x.len());
    out.spare_capacity_mut()[..x.len()]
        .par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .zip(x.par_chunks(PARALLEL_CHUNK_SIZE))
        .for_each(|(out, x)| T::scalars_uninit::<F>(x, b, out));
    // SAFETY: the chunks cover the first `x.len()` elements, and
    // `scalars_uninit` initialised each of them.
    unsafe { out.set_len(x.len()) };
    out
}

fn par_scalars_in_place<F: NaryOp<3>, T: MathElement + Send + Sync>(x: &mut [T], b: [T; 2]) {
    if x.len() <= PARALLEL_SIMD_THRESHOLD {
        return T::scalars_in_place::<F>(x, b);
    }
    x.par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .for_each(|x| T::scalars_in_place::<F>(x, b));
}

fn par_nary_to_vec<F: NaryOp<N>, T: MathElement + Send + Sync, const N: usize>(
    src: [&[T]; N],
) -> Vec<T> {
    let len = src[0].len();
    if len <= PARALLEL_SIMD_THRESHOLD || src.iter().any(|s| s.len() != len) {
        return nary_to_vec::<F, T, N>(src);
    }
    let mut out = Vec::with_capacity(len);
    out.spare_capacity_mut()[..len]
        .par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .enumerate()
        .for_each(|(i, out)| {
            let start = i * PARALLEL_CHUNK_SIZE;
            T::nary_uninit::<F, N>(src.map(|s| &s[start..start + out.len()]), out);
        });
    // SAFETY: the chunks cover the first `len` elements, and `nary_uninit`
    // initialised each of them.
    unsafe { out.set_len(len) };
    out
}

fn par_nary_in_place<F: NaryOp<N>, T: MathElement + Send + Sync, const N: usize, const M: usize>(
    x: &mut [T],
    rest: [&[T]; M],
) {
    if x.len() <= PARALLEL_SIMD_THRESHOLD || rest.iter().any(|r| r.len() != x.len()) {
        return T::nary_in_place::<F, N, M>(x, rest);
    }
    x.par_chunks_mut(PARALLEL_CHUNK_SIZE)
        .enumerate()
        .for_each(|(i, x)| {
            let start = i * PARALLEL_CHUNK_SIZE;
            T::nary_in_place::<F, N, M>(x, rest.map(|r| &r[start..start + x.len()]));
        });
}

#[cfg(test)]
mod tests {
    use super::ParSliceMath;
    use crate::testing::{TestFloat, assert_same};
    use crate::{PARALLEL_CHUNK_SIZE, PARALLEL_SIMD_THRESHOLD, SliceMath};
    use std::vec::Vec;

    /// Long enough for several chunks and a short last one.
    const LEN: usize = PARALLEL_SIMD_THRESHOLD + 3 * PARALLEL_CHUNK_SIZE + 13;

    fn input<T: TestFloat>(len: usize, seed: f64) -> Vec<T> {
        (0..len)
            .map(|i| T::from_f64(((i as f64 + seed) * 0.37).sin() * 3.0))
            .collect()
    }

    /// Every `par_` function against its `SliceMath` one, at `len`.
    fn check<T: TestFloat>(len: usize)
    where
        [T]: SliceMath<T> + ParSliceMath<T>,
    {
        let (x, y, z, w) = (
            input::<T>(len, 0.0),
            input(len, 1.0),
            input(len, 2.0),
            input(len, 3.0),
        );
        type Pair<T> = (
            fn(&[T]) -> Vec<T>,
            fn(&[T]) -> Vec<T>,
            fn(&mut [T]),
            fn(&mut [T]),
        );
        let unary: [(&str, Pair<T>); 24] = [
            (
                "abs",
                (
                    |x| x.abs(),
                    |x| x.par_abs(),
                    |x| x.abs_in_place(),
                    |x| x.par_abs_in_place(),
                ),
            ),
            (
                "acos",
                (
                    |x| x.acos(),
                    |x| x.par_acos(),
                    |x| x.acos_in_place(),
                    |x| x.par_acos_in_place(),
                ),
            ),
            (
                "asin",
                (
                    |x| x.asin(),
                    |x| x.par_asin(),
                    |x| x.asin_in_place(),
                    |x| x.par_asin_in_place(),
                ),
            ),
            (
                "atan",
                (
                    |x| x.atan(),
                    |x| x.par_atan(),
                    |x| x.atan_in_place(),
                    |x| x.par_atan_in_place(),
                ),
            ),
            (
                "cbrt",
                (
                    |x| x.cbrt(),
                    |x| x.par_cbrt(),
                    |x| x.cbrt_in_place(),
                    |x| x.par_cbrt_in_place(),
                ),
            ),
            (
                "ceil",
                (
                    |x| x.ceil(),
                    |x| x.par_ceil(),
                    |x| x.ceil_in_place(),
                    |x| x.par_ceil_in_place(),
                ),
            ),
            (
                "cos",
                (
                    |x| x.cos(),
                    |x| x.par_cos(),
                    |x| x.cos_in_place(),
                    |x| x.par_cos_in_place(),
                ),
            ),
            (
                "exp",
                (
                    |x| x.exp(),
                    |x| x.par_exp(),
                    |x| x.exp_in_place(),
                    |x| x.par_exp_in_place(),
                ),
            ),
            (
                "floor",
                (
                    |x| x.floor(),
                    |x| x.par_floor(),
                    |x| x.floor_in_place(),
                    |x| x.par_floor_in_place(),
                ),
            ),
            (
                "ln",
                (
                    |x| x.ln(),
                    |x| x.par_ln(),
                    |x| x.ln_in_place(),
                    |x| x.par_ln_in_place(),
                ),
            ),
            (
                "log1p",
                (
                    |x| x.log1p(),
                    |x| x.par_log1p(),
                    |x| x.log1p_in_place(),
                    |x| x.par_log1p_in_place(),
                ),
            ),
            (
                "sigmoid",
                (
                    |x| x.sigmoid(),
                    |x| x.par_sigmoid(),
                    |x| x.sigmoid_in_place(),
                    |x| x.par_sigmoid_in_place(),
                ),
            ),
            (
                "log_sigmoid",
                (
                    |x| x.log_sigmoid(),
                    |x| x.par_log_sigmoid(),
                    |x| x.log_sigmoid_in_place(),
                    |x| x.par_log_sigmoid_in_place(),
                ),
            ),
            (
                "softplus",
                (
                    |x| x.softplus(),
                    |x| x.par_softplus(),
                    |x| x.softplus_in_place(),
                    |x| x.par_softplus_in_place(),
                ),
            ),
            (
                "sinh",
                (
                    |x| x.sinh(),
                    |x| x.par_sinh(),
                    |x| x.sinh_in_place(),
                    |x| x.par_sinh_in_place(),
                ),
            ),
            (
                "cosh",
                (
                    |x| x.cosh(),
                    |x| x.par_cosh(),
                    |x| x.cosh_in_place(),
                    |x| x.par_cosh_in_place(),
                ),
            ),
            (
                "tanh",
                (
                    |x| x.tanh(),
                    |x| x.par_tanh(),
                    |x| x.tanh_in_place(),
                    |x| x.par_tanh_in_place(),
                ),
            ),
            (
                "atanh",
                (
                    |x| x.atanh(),
                    |x| x.par_atanh(),
                    |x| x.atanh_in_place(),
                    |x| x.par_atanh_in_place(),
                ),
            ),
            (
                "asinh",
                (
                    |x| x.asinh(),
                    |x| x.par_asinh(),
                    |x| x.asinh_in_place(),
                    |x| x.par_asinh_in_place(),
                ),
            ),
            (
                "acosh",
                (
                    |x| x.acosh(),
                    |x| x.par_acosh(),
                    |x| x.acosh_in_place(),
                    |x| x.par_acosh_in_place(),
                ),
            ),
            (
                "erf",
                (
                    |x| x.erf(),
                    |x| x.par_erf(),
                    |x| x.erf_in_place(),
                    |x| x.par_erf_in_place(),
                ),
            ),
            (
                "sin",
                (
                    |x| x.sin(),
                    |x| x.par_sin(),
                    |x| x.sin_in_place(),
                    |x| x.par_sin_in_place(),
                ),
            ),
            (
                "sqrt",
                (
                    |x| x.sqrt(),
                    |x| x.par_sqrt(),
                    |x| x.sqrt_in_place(),
                    |x| x.par_sqrt_in_place(),
                ),
            ),
            (
                "tan",
                (
                    |x| x.tan(),
                    |x| x.par_tan(),
                    |x| x.tan_in_place(),
                    |x| x.par_tan_in_place(),
                ),
            ),
        ];
        for (name, (seq, par, seq_in, par_in)) in unary {
            let what = std::format!("par_{name} {} len {len}", T::NAME);
            assert_same(&par(&x), &seq(&x), &what);
            let (mut a, mut b) = (x.clone(), x.clone());
            seq_in(&mut a);
            par_in(&mut b);
            assert_same(&b, &a, &std::format!("{what} in place"));
        }

        let what = |f: &str| std::format!("par_{f} {} len {len}", T::NAME);
        assert_same(&y.par_atan2(&x), &y.atan2(&x), &what("atan2"));
        assert_same(&x.par_powf(&y), &x.powf(&y), &what("powf"));
        let p = T::from_f64(1.7);
        assert_same(
            &x.par_powf_scalar(p),
            &x.powf_scalar(p),
            &what("powf_scalar"),
        );
        for n in [-3, 7, 19] {
            assert_same(&x.par_powi(n), &x.powi(n), &what("powi"));
        }
        assert_same(&x.par_hypot(&y), &x.hypot(&y), &what("hypot"));
        assert_same(&x.par_hypot3(&y, &z), &x.hypot3(&y, &z), &what("hypot3"));
        assert_same(
            &x.par_hypot4(&y, &z, &w),
            &x.hypot4(&y, &z, &w),
            &what("hypot4"),
        );

        let in_place = |seq: fn(&mut [T]), par: fn(&mut [T]), f: &str| {
            let (mut a, mut b) = (x.clone(), x.clone());
            seq(&mut a);
            par(&mut b);
            assert_same(&b, &a, &std::format!("{} in place", what(f)));
        };
        // `fn` pointers can't capture `y`, `z`, `w`: recompute them inside.
        in_place(
            |x| x.atan2_in_place(&input(x.len(), 1.0)),
            |x| x.par_atan2_in_place(&input(x.len(), 1.0)),
            "atan2",
        );
        in_place(
            |x| x.powf_in_place(&input(x.len(), 1.0)),
            |x| x.par_powf_in_place(&input(x.len(), 1.0)),
            "powf",
        );
        in_place(
            |x| x.powf_scalar_in_place(T::from_f64(1.7)),
            |x| x.par_powf_scalar_in_place(T::from_f64(1.7)),
            "powf_scalar",
        );
        in_place(|x| x.powi_in_place(-3), |x| x.par_powi_in_place(-3), "powi");
        in_place(|x| x.powi_in_place(19), |x| x.par_powi_in_place(19), "powi");
        in_place(
            |x| x.hypot_in_place(&input(x.len(), 1.0)),
            |x| x.par_hypot_in_place(&input(x.len(), 1.0)),
            "hypot",
        );
        in_place(
            |x| x.hypot3_in_place(&input(x.len(), 1.0), &input(x.len(), 2.0)),
            |x| x.par_hypot3_in_place(&input(x.len(), 1.0), &input(x.len(), 2.0)),
            "hypot3",
        );
        in_place(
            |x| {
                x.hypot4_in_place(
                    &input(x.len(), 1.0),
                    &input(x.len(), 2.0),
                    &input(x.len(), 3.0),
                )
            },
            |x| {
                x.par_hypot4_in_place(
                    &input(x.len(), 1.0),
                    &input(x.len(), 2.0),
                    &input(x.len(), 3.0),
                )
            },
            "hypot4",
        );
    }

    /// Bit for bit equal to `SliceMath`, above and below the threshold.
    #[test]
    fn par_matches_sequential() {
        for len in [0, 1, 100, PARALLEL_SIMD_THRESHOLD, LEN] {
            check::<f32>(len);
            check::<f64>(len);
        }
    }

    #[test]
    #[should_panic]
    fn par_length_mismatch_panics() {
        let x = std::vec![1.0_f32; LEN];
        let _ = x.par_hypot(&x[1..]);
    }

    #[test]
    #[should_panic]
    fn par_nary_length_mismatch_panics() {
        let mut x = std::vec![1.0_f64; LEN];
        let y = x.clone();
        x.par_hypot3_in_place(&y, &y[1..]);
    }
}
