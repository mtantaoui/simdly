//! `fmod(x, y) = x % y` and `remainder(x, y)`, written once over [`Float`].
//!
//! - **`fmod`** is Rust's `%` and C's `fmod`: `x − n·y` with `n = trunc(x/y)`,
//!   exact, the sign of `x`. No ISA has it, so it is long division in
//!   chunks: with `a = |x|` and `b = |y|`, `b' = b·2^e` (exact) with `e` such
//!   that `a/b' < 2^(p−1)`, `q = trunc(a/b')` is the integer quotient or one
//!   too large (the division rounds up at most to the next integer), and
//!   `a − q·b'` is exact in one fused multiply-add: a multiple of `b'`'s ULP
//!   below `b'` in magnitude. Adding `b'` back where it went negative leaves
//!   `a mod b'` in `[0, b')`, and since `b'` is a multiple of `b`,
//!   `a mod b = (a mod b') mod b`. Each pass removes `p − 2` bits of the
//!   exponent gap: one pass while `a/b < 2^(p−1)`, at most about 40 (`f64`)
//!   for `10^300 % 10^−300`; the loop runs while some lane needs it.
//! - **`remainder`** is Python's and PyTorch's `%`, burn's
//!   `float_remainder`: `r = x % y`, then `r + y` where `r ≠ 0` and its sign
//!   differs from `y`'s, the sign of `y`. That sum rounds, as burn's.
//!
//! `fmod(x, ±0)`, `fmod(±∞, y)` and NaN give NaN; `fmod(x, ±∞) = x` for
//! finite `x`; `fmod(±0, y) = ±0`.

use super::atan::{k, ks};
use super::float::{Element, Float};
use super::ln;
use super::round::Trunc;

/// `⌊log₂ v⌋` for positive finite `v`, subnormals included.
#[inline(always)]
fn ilogb<V: Float>(v: V) -> V
where
    V::Elem: ln::Kernel,
{
    let [tiny_scale, tiny_k] = ks(v, <V::Elem as ln::Kernel>::TINY);
    let tiny = v.lt(v.constant(V::Elem::MIN_POSITIVE));
    let (_, e) = V::select(tiny, v * tiny_scale, v).frexp();
    e + V::select(tiny, tiny_k, k(v, 0.0))
}

/// `a mod b` for finite `a ≥ 0` and finite `b > 0`, exactly.
#[inline(always)]
fn fmod_abs<V: Float + Trunc>(a: V, b: V) -> V
where
    V::Elem: Kernel,
{
    let [zero, half, one] = [k(a, 0.0), k(a, 0.5), k(a, 1.0)];
    let [lo, hi, sup, sdown] = ks(a, V::Elem::DEKKER);
    let gap = a.constant(V::Elem::GAP);
    let round = a.constant(V::Elem::ROUND);
    let eb = ilogb(b);
    let mut r = a;
    loop {
        let more = r.ge(b);
        if !V::any(more) {
            return r;
        }
        // b' = b·2^e ≤ r, r/b' < 2^(p−1); 2^e in two normal factors, as `b'`
        // may be far from `b` (`e` reaches about 2100 for `f64`).
        let e = ilogb(r) - eb - gap;
        let e = V::select(e.gt(zero), e, zero);
        let e1 = e.mul_add(half, round) - round;
        let bs = b.ldexp(e1).ldexp(e - e1);
        let q = (r / bs).trunc();
        // r − q·b', exact (`residual`). The scalar backend's is Dekker's,
        // whose partial products must stay normal: near either end of the
        // range, r and b' are scaled by a power of 2 first, and the result
        // (a multiple of b''s ULP) back, both exactly.
        let up = V::select(r.lt(lo), sup, V::select(bs.gt(hi), sdown, one));
        let down = V::select(r.lt(lo), sdown, V::select(bs.gt(hi), sup, one));
        let t = q.residual(bs * up, r * up) * down;
        let t = V::select(t.lt(zero), t + bs, t);
        r = V::select(more, t, r);
    }
}

/// `x % y`, as Rust's and C's `fmod`.
#[inline(always)]
fn fmod<V: Float + Trunc>(x: V, y: V) -> V
where
    V::Elem: Kernel,
{
    let [zero, one] = [k(x, 0.0), k(x, 1.0)];
    let [inf, nan] = [k(x, f32::INFINITY), k(x, f32::NAN)];
    let (a, b) = (x.abs(), y.abs());

    // Finite x, finite nonzero y: the long division; anything else runs it
    // on (0, 1), at no cost, and is selected below.
    let ok = V::select(
        a.lt(inf),
        V::select(b.gt(zero), V::select(b.lt(inf), one, zero), zero),
        zero,
    )
    .eq(one);
    let r = fmod_abs(V::select(ok, a, zero), V::select(ok, b, one));
    let r = V::select(x.is_sign_negative(), -r, r);

    // fmod(x, ±∞) = x for finite x; NaN for x = ±∞, y = ±0 or NaN.
    let x_by_inf = V::select(a.lt(inf), V::select(b.eq(inf), one, zero), zero).eq(one);
    V::select(ok, r, V::select(x_by_inf, x, nan))
}

/// Python's `x % y`: [`fmod`], moved by `y` to `y`'s sign.
#[inline(always)]
fn remainder<V: Float + Trunc>(x: V, y: V) -> V
where
    V::Elem: Kernel,
{
    let zero = k(x, 0.0);
    let r = fmod(x, y);
    // r ≠ 0 and (r < 0) ≠ (y < 0): exactly one of them is negative.
    let neg = V::select(r.lt(zero), k(x, 1.0), zero) + V::select(y.lt(zero), k(x, 1.0), zero);
    let differ = V::select(r.eq(zero), zero, neg).eq(k(x, 1.0));
    V::select(differ, r + y, r)
}

/// A lane type with `x % y` and Python's `x % y`.
pub(crate) trait Fmod {
    fn fmod(self, y: Self) -> Self;
    fn remainder(self, y: Self) -> Self;
}

impl<V: Float + Trunc> Fmod for V
where
    V::Elem: Kernel,
{
    #[inline(always)]
    fn fmod(self, y: Self) -> Self {
        fmod(self, y)
    }

    #[inline(always)]
    fn remainder(self, y: Self) -> Self {
        remainder(self, y)
    }
}

/// `fmod`'s constant, per element type.
pub(crate) trait Kernel: ln::Kernel {
    /// `p − 2`: each pass of the long division removes this many bits of
    /// the exponent gap, keeping the quotient below `2^(p−1)`.
    const GAP: Self;
    /// `[lo, hi, 2^k, 2^−k]`: below `lo` (`r`) or above `hi` (`b'`), the
    /// scalar backend's Dekker product would underflow or overflow, and
    /// the pass runs scaled by `2^±k`.
    const DEKKER: [Self; 4];
}

impl Kernel for f32 {
    const GAP: f32 = 22.0;
    // 2^−60, 2^100, 2^60, 2^−60
    const DEKKER: [f32; 4] = [
        f32::from_bits(67 << 23),
        f32::from_bits(227 << 23),
        f32::from_bits(187 << 23),
        f32::from_bits(67 << 23),
    ];
}

impl Kernel for f64 {
    const GAP: f64 = 51.0;
    // 2^−800, 2^900, 2^400, 2^−400
    const DEKKER: [f64; 4] = [
        f64::from_bits(223 << 52),
        f64::from_bits(1923 << 52),
        f64::from_bits(1423 << 52),
        f64::from_bits(623 << 52),
    ];
}

#[cfg(test)]
mod tests {
    use crate::math::{FmodOp, FmodScalarOp, MathElement, RemainderOp, RemainderScalarOp};
    use crate::testing::{TestFloat, assert_same};
    use std::vec::Vec;

    /// burn's `remainder_float`, on `std`'s `%`.
    fn burn_remainder<
        T: TestFloat + core::ops::Rem<Output = T> + core::ops::Add<Output = T> + PartialOrd,
    >(
        a: T,
        b: T,
    ) -> T {
        let zero = T::from_f64(0.0);
        let r = a % b;
        if r != zero && (r < zero) != (b < zero) {
            r + b
        } else {
            r
        }
    }

    /// Pairs over every binade of both operands, subnormals included, both
    /// signs, plus ±0, ±∞ and NaN on either side: exponent gaps of every
    /// size, from none to `MAX / MIN_SUBNORMAL`.
    fn pairs<T: TestFloat>(n: usize) -> (Vec<T>, Vec<T>) {
        let mut s = 0x9e37_79b9_7f4a_7c15_u64;
        let mut bits = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let any = |b: u64| {
            if T::NAME == "f32" {
                T::from_bits64(b & 0xffff_ffff)
            } else {
                T::from_bits64(b)
            }
        };
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for i in 0..n {
            let (a, b) = (any(bits()), any(bits()));
            // A third of the pairs with exponents close, as most uses.
            let b = if i % 3 == 0 {
                T::from_f64(a.to_f64() / (1.0 + (bits() % 1000) as f64 * 0.37))
            } else {
                b
            };
            x.push(a);
            y.push(b);
        }
        let special = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1e-310,
            f64::MAX,
        ];
        for &a in &special {
            for &b in &special {
                x.push(T::from_f64(a));
                y.push(T::from_f64(b));
            }
        }
        (x, y)
    }

    fn check<T>()
    where
        T: TestFloat
            + MathElement
            + core::ops::Rem<Output = T>
            + core::ops::Add<Output = T>
            + PartialOrd,
    {
        let (x, y) = pairs::<T>(200_000);
        let want: Vec<T> = x.iter().zip(&y).map(|(&a, &b)| a % b).collect();
        for (backend, got) in T::every_backend_binary::<FmodOp>(&x, &y) {
            assert_same(&got, &want, &std::format!("fmod {} {backend}", T::NAME));
        }
        let want: Vec<T> = x
            .iter()
            .zip(&y)
            .map(|(&a, &b)| burn_remainder(a, b))
            .collect();
        for (backend, got) in T::every_backend_binary::<RemainderOp>(&x, &y) {
            assert_same(
                &got,
                &want,
                &std::format!("remainder {} {backend}", T::NAME),
            );
        }
        // One divisor for the slice.
        for d in [3.0, -0.7, 1e-30, 1e30] {
            let d = T::from_f64(d);
            let want: Vec<T> = x.iter().map(|&a| a % d).collect();
            for (backend, got) in
                T::every_backend_scalars::<FmodScalarOp>(&x, [d, T::from_f64(0.0)])
            {
                assert_same(
                    &got,
                    &want,
                    &std::format!("fmod_scalar {} {backend}", T::NAME),
                );
            }
            let want: Vec<T> = x.iter().map(|&a| burn_remainder(a, d)).collect();
            for (backend, got) in
                T::every_backend_scalars::<RemainderScalarOp>(&x, [d, T::from_f64(0.0)])
            {
                assert_same(
                    &got,
                    &want,
                    &std::format!("remainder_scalar {} {backend}", T::NAME),
                );
            }
        }
    }

    /// Bit for bit `%` and burn's `remainder_float`, signed zeros, infinities
    /// and NaN included.
    #[test]
    fn fmod_and_remainder_match() {
        check::<f32>();
        check::<f64>();
    }
}
