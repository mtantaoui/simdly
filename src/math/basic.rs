//! `−x`, `1/x`, `sign x` and `clamp(x, min, max)`, written once over
//! [`Float`]: one instruction or two per lane, exact (`1/x` rounded once,
//! as `std`'s).
//!
//! - `sign` is burn's and PyTorch's on numbers, `−1`, `0` or `1`, with `0`
//!   for `±0` and, as burn's, for NaN.
//! - `clamp` is `f32::clamp`'s comparisons: `x < min` gives `min`, `x > max`
//!   gives `max`, anything else (NaN, a signed zero equal to a bound) `x`.

use super::atan::k;
use super::float::Float;

/// `−1`, `0` or `1` by the sign of `x`; `0` for `±0` and NaN.
#[inline(always)]
pub(crate) fn sign<V: Float>(x: V) -> V {
    let [zero, one] = [k(x, 0.0), k(x, 1.0)];
    V::select(x.gt(zero), one, V::select(x.lt(zero), -one, zero))
}

/// `min` below `min`, `max` above `max`, `x` otherwise (NaN included).
#[inline(always)]
pub(crate) fn clamp<V: Float>(x: V, min: V, max: V) -> V {
    V::select(x.lt(min), min, V::select(x.gt(max), max, x))
}

#[cfg(test)]
mod tests {
    use crate::math::{ClampOp, MathElement, NegOp, RecipOp, SignOp, UnaryOp};
    use crate::testing::{TestFloat, assert_same, samples};
    use std::vec::Vec;

    fn inputs<T: TestFloat>() -> Vec<T> {
        let max = if T::NAME == "f32" {
            f32::MAX.into()
        } else {
            f64::MAX
        };
        let mut x = samples(50_000, max, &[1.0, 2.0]);
        x.extend(
            [
                0.0,
                -0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                -f64::NAN,
            ]
            .map(T::from_f64),
        );
        x
    }

    /// `F` bit for bit `reference` (computed in `T`), on every backend.
    fn check<T: TestFloat + MathElement, F: UnaryOp>(name: &str, reference: impl Fn(T) -> T) {
        let x: Vec<T> = inputs();
        let want: Vec<T> = x.iter().map(|&v| reference(v)).collect();
        for (backend, got) in T::every_backend::<F>(&x) {
            assert_same(&got, &want, &std::format!("{name} {} {backend}", T::NAME));
        }
    }

    #[test]
    fn neg_recip_sign() {
        check::<f32, NegOp>("neg", |v| -v);
        check::<f64, NegOp>("neg", |v| -v);
        check::<f32, RecipOp>("recip", |v| 1.0 / v);
        check::<f64, RecipOp>("recip", |v| 1.0 / v);
        let sign32 = |v: f32| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
        let sign64 = |v: f64| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
        check::<f32, SignOp>("sign", sign32);
        check::<f64, SignOp>("sign", sign64);
        // NaN keeps its payload's sign through `neg`.
        let n = f64::every_backend::<NegOp>(&[f64::NAN]);
        for (backend, got) in n {
            assert!(got[0].is_sign_negative(), "neg NaN {backend}");
        }
    }

    /// `f32::clamp`'s results, signed zeros and NaN included.
    #[test]
    fn clamp_matches_std() {
        for (lo, hi) in [
            (-1.0, 1.0),
            (0.0, 0.0),
            (-0.0, 0.0),
            (2.5, 1e30),
            (f64::NEG_INFINITY, 0.0),
        ] {
            let x: Vec<f64> = inputs();
            let want: Vec<f64> = x.iter().map(|v| v.clamp(lo, hi)).collect();
            for (backend, got) in f64::every_backend_scalars::<ClampOp>(&x, [lo, hi]) {
                assert_same(
                    &got,
                    &want,
                    &std::format!("clamp f64 [{lo}, {hi}] {backend}"),
                );
            }
            let x: Vec<f32> = inputs();
            let (lo, hi) = (lo as f32, hi as f32);
            let want: Vec<f32> = x.iter().map(|v| v.clamp(lo, hi)).collect();
            for (backend, got) in f32::every_backend_scalars::<ClampOp>(&x, [lo, hi]) {
                assert_same(
                    &got,
                    &want,
                    &std::format!("clamp f32 [{lo}, {hi}] {backend}"),
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "min > max")]
    fn clamp_panics_on_reversed_bounds() {
        use crate::SliceMath;
        [1.0_f32].clamp_in_place(1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "NaN")]
    fn clamp_panics_on_nan_bound() {
        use crate::SliceMath;
        [1.0_f64].clamp_in_place(f64::NAN, 0.0);
    }
}
