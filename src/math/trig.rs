//! `sin x` and `cos x`, ≤ 1 ULP on every backend, for every finite input: musl's kernels
//! and argument reduction (from FreeBSD's `k_sin.c`, `k_cos.c`, `k_sinf.c`,
//! `k_cosf.c`, `e_rem_pio2.c`, `e_rem_pio2f.c`), written once over
//! [`Float`].
//!
//! `x = n·π/2 + y`, `|y| ≤ π/4`, then `sin x` is `±sin y` or `±cos y` by
//! `n mod 4` (`cos x = sin(x + π/2)`: `n + 1`); a register computes both
//! polynomials and selects per lane.
//!
//! - **`f32`** runs in `f64` ([`Widen`]), as musl's `sinf`: `π/2` in two
//!   parts (25 + 53 bits) reduces exactly enough for `|x| < 2^28·π/2`, and
//!   short `f64` polynomials give the result before it is rounded to `f32`.
//! - **`f64`**: `π/2` in three parts with tails, musl's second round for
//!   every lane and its third where more than 49 bits cancelled, `y` kept as
//!   `y0 + y1`, for `|x| < 2^20·π/2`.
//! - **Larger `|x|`**: lanes beyond those limits are reduced exactly by
//!   [`reduce`] (2/π to 1 408 bits, integer arithmetic), one at a time
//!   ([`Float::map_lanes`]), only when a register holds one.
//!
//! The algorithms and constants come from FreeBSD's msun, under this notice:
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```

#![allow(clippy::excessive_precision, reason = "fdlibm's constants, verbatim")]

use super::atan::{k, ks};
use super::float::{Element, Float, WideFn, Widen, two_sum};

/// A lane type with a sine, a cosine and a tangent.
pub(crate) trait Trig {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
}

/// `F` for the lane types of one element type.
pub(crate) trait TrigKernel<V> {
    fn trig<F: TrigFn>(x: V) -> V;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Trig for V
where
    V::Elem: TrigKernel<V>,
{
    #[inline(always)]
    fn sin(self) -> Self {
        V::Elem::trig::<SinFn>(self)
    }

    #[inline(always)]
    fn cos(self) -> Self {
        V::Elem::trig::<CosFn>(self)
    }

    #[inline(always)]
    fn tan(self) -> Self {
        V::Elem::trig::<TanFn>(self)
    }
}

impl<V: Widen> TrigKernel<V> for f32 {
    #[inline(always)]
    fn trig<F: TrigFn>(x: V) -> V {
        x.via_f64::<TrigF32<F>>()
    }
}

impl<V: Float<Elem = f64>> TrigKernel<V> for f64 {
    #[inline(always)]
    fn trig<F: TrigFn>(x: V) -> V {
        trig_f64::<V, F>(x)
    }
}

/// `sin`, `cos` or `tan` of `x = n·π/2 + y`, from `n` and `y`.
pub(crate) trait TrigFn {
    /// `f(±0) = ±0`, which the polynomials turn into `+0`.
    const ODD: bool;
    /// For `f32` results, from `y` in `f64`.
    fn from_f32<V: Float<Elem = f64>>(n: V, y: V) -> V;
    /// For `f64` results, from `y = y0 + y1`.
    fn from_f64<V: Float<Elem = f64>>(n: V, y0: V, y1: V) -> V;
}

pub(crate) struct SinFn;
pub(crate) struct CosFn;
pub(crate) struct TanFn;

impl TrigFn for SinFn {
    const ODD: bool = true;

    #[inline(always)]
    fn from_f32<V: Float<Elem = f64>>(n: V, y: V) -> V {
        by_quadrant::<_, false>(n, sin_f32_kernel(y), cos_f32_kernel(y))
    }

    #[inline(always)]
    fn from_f64<V: Float<Elem = f64>>(n: V, y0: V, y1: V) -> V {
        by_quadrant::<_, false>(n, sin_f64_kernel(y0, y1), cos_f64_kernel(y0, y1))
    }
}

impl TrigFn for CosFn {
    const ODD: bool = false;

    #[inline(always)]
    fn from_f32<V: Float<Elem = f64>>(n: V, y: V) -> V {
        by_quadrant::<_, true>(n, sin_f32_kernel(y), cos_f32_kernel(y))
    }

    #[inline(always)]
    fn from_f64<V: Float<Elem = f64>>(n: V, y0: V, y1: V) -> V {
        by_quadrant::<_, true>(n, sin_f64_kernel(y0, y1), cos_f64_kernel(y0, y1))
    }
}

/// `tan x` is `tan y` for even `n`, `−1/tan y` for odd.
impl TrigFn for TanFn {
    const ODD: bool = true;

    #[inline(always)]
    fn from_f32<V: Float<Elem = f64>>(n: V, y: V) -> V {
        let t = tan_f32_kernel(y);
        V::select(is_odd(n), k(y, -1.0) / t, t)
    }

    #[inline(always)]
    fn from_f64<V: Float<Elem = f64>>(n: V, y0: V, y1: V) -> V {
        tan_f64_kernel(is_odd(n), y0, y1)
    }
}

/// Lanes where the integer `n` is odd.
#[inline(always)]
fn is_odd<V: Float<Elem = f64>>(n: V) -> V::Mask {
    let [half, two, one] = ks(n, [0.5, 2.0, 1.0]);
    (n - two * (n * half).floor()).eq(one)
}

/// `2/π`, rounded.
const INV_PIO2: f64 = core::f64::consts::FRAC_2_PI;
/// `1.5 · 2^52`: adding and subtracting it rounds to an integer.
const TOINT: f64 = 6_755_399_441_055_744.0;

/// Per lane, `sin x` from `x = n·π/2 + y`, `s = sin y`, `c = cos y`: `s`
/// if `n mod 4 = 0`, `c` if 1, `−s` if 2, `−c` if 3; `cos x` with `n + 1`.
#[inline(always)]
fn by_quadrant<V: Float, const COS: bool>(n: V, s: V, c: V) -> V {
    let [quarter, one, two, four] = ks(n, [0.25, 1.0, 2.0, 4.0].map(V::Elem::from_f32));
    let n = if COS { n + one } else { n };
    let q = n - four * (n * quarter).floor();
    V::select(
        q.eq(one),
        c,
        V::select(q.eq(two), -s, V::select(q.lt(one), s, -c)),
    )
}

/// `F` of an `f32` input, in `f64`: the `f32` kernel.
struct TrigF32<F>(core::marker::PhantomData<F>);

impl<F: TrigFn> WideFn for TrigF32<F> {
    #[inline(always)]
    fn call<W: Float<Elem = f64>>(x: W) -> W {
        // musl e_rem_pio2f.c: π/2 = PIO2_1 (25 bits) + PIO2_1T.
        const PIO2_1: f64 = 1.570_796_310_901_641_845_70e0;
        const PIO2_1T: f64 = 1.589_325_477_352_819_669_16e-8;
        /// `2^28·π/2` as an `f32` (`0x4dc90fdb`): from here on,
        /// [`reduce_f32_big`].
        const LIMIT: f64 = 421_657_440.0;

        let [inv_pio2, toint, pio2_1, pio2_1t, limit] =
            ks(x, [INV_PIO2, TOINT, PIO2_1, PIO2_1T, LIMIT]);
        let n = x.mul_add(inv_pio2, toint) - toint;
        let y = (-n).mul_add(pio2_1t, (-n).mul_add(pio2_1, x));
        let ax = x.abs();
        let big = ax.ge(limit);
        let (n, y) = if W::any(big) {
            let (nb, yb) = reduce_f32_big(x, ax);
            (W::select(big, nb, n), W::select(big, yb, y))
        } else {
            (n, y)
        };
        let r = F::from_f32(n, y);
        special::<_, F>(x, ax, r)
    }
}

/// `r`, except `x` where `x = ±0` and `F` is odd, and NaN where `x` is `±∞`
/// or NaN.
#[inline(always)]
fn special<V: Float, F: TrigFn>(x: V, ax: V, r: V) -> V {
    let r = if F::ODD {
        V::select(x.eq(k(x, 0.0)), x, r)
    } else {
        r
    };
    nan_unless_finite(x, ax, r)
}

/// `r`, except NaN where `x` is `±∞` or NaN.
#[inline(always)]
fn nan_unless_finite<V: Float>(x: V, ax: V, r: V) -> V {
    V::select(ax.lt(k(x, f32::INFINITY)), r, k(x, f32::NAN))
}

/// `r`, with the lanes where `large` is set replaced by `slow(x)`, computed
/// only if there are any.
#[inline(always)]
fn finish<V: Float>(x: V, large: V::Mask, r: V, slow: fn(V::Elem) -> V::Elem) -> V {
    if V::any(large) {
        V::select(large, x.map_lanes(slow), r)
    } else {
        r
    }
}

/// musl k_sinf.c, in `f64`: `sin y` for `|y| ≤ π/4`, to more than `f32`'s
/// precision.
#[inline(always)]
fn sin_f32_kernel<V: Float<Elem = f64>>(y: V) -> V {
    let [s1, s2, s3, s4] = ks(
        y,
        [
            -0.166_666_666_416_265_235_595,
            0.008_333_329_385_889_463_175_6,
            -0.000_198_393_348_360_966_317_347,
            0.000_002_718_311_493_989_821_906_4,
        ],
    );
    let z = y * y;
    let w = z * z;
    let r = z.mul_add(s4, s3);
    let s = z * y;
    (s * w).mul_add(r, s.mul_add(z.mul_add(s2, s1), y))
}

/// musl k_cosf.c, in `f64`: `cos y` for `|y| ≤ π/4`.
#[inline(always)]
fn cos_f32_kernel<V: Float<Elem = f64>>(y: V) -> V {
    let [c0, c1, c2, c3, one] = ks(
        y,
        [
            -0.499_999_997_251_031_003_120,
            0.041_666_623_323_739_063_189_4,
            -0.001_388_676_377_460_992_946_92,
            0.000_024_390_448_796_277_409_065_4,
            1.0,
        ],
    );
    let z = y * y;
    let w = z * z;
    let r = z.mul_add(c3, c2);
    (w * z).mul_add(r, w.mul_add(c1, z.mul_add(c0, one)))
}

/// `π/2 = C1 + C2 + C3` to `2^−163`, each the next 53 bits.
const PIO2_C: [f64; 3] = [
    f64::from_bits(0x3ff9_21fb_5444_2d18),
    f64::from_bits(0x3c91_a626_3314_5c07),
    f64::from_bits(0xb91f_1976_b7ed_8fbc),
];

/// `x − n·π/2` as `y0 + y1`, for `|x| < 2^33` and `n = round(x·2/π)` (to
/// within `2^−19`, so `|y| ≤ π/4 + 2^−18`): each `n·Ci` split into two
/// floats exactly (`residual`), and the sum kept in double-double. The error is about `2^(k − 157)` for
/// `|x| < 2^k`, and `|y| ≥ 2^−59` for `f64` inputs in `[2^20, 2^64)`
/// (continued fractions of `2/π`): `2^−124` against `2^−119` needed.
#[inline(always)]
fn reduce_mid<V: Float<Elem = f64>>(x: V, n: V) -> (V, V) {
    let [c1, c2, c3] = ks(x, PIO2_C);
    let p1 = n * c1;
    let e1 = -n.residual(c1, p1); // n·c1 − p1
    let t = x - p1; // exact: x and p1 within a factor 2 (Sterbenz)
    let p2 = n * c2;
    let e2 = -n.residual(c2, p2);
    let (a, ea) = two_sum(t, -e1);
    let (b, eb) = two_sum(a, -p2);
    let lo = ((ea + eb) - e2) - n * c3;
    let y0 = b + lo;
    (y0, lo - (y0 - b))
}

/// `x = n·π/2 + y` for an `f32` value `|x| ≥ 2^28·π/2`, to `f32`'s needs,
/// in vector arithmetic.
///
/// With `x = m·2^e` (`m` a 24-bit integer) and `G ≤ e`, `M = x·2^−G` is an
/// integer, so `x·2/π ≡ M·A (mod 4)` with `A = 2^G·(2/π) mod 4`. Two
/// groups, `G = 5` and `55`, cover every `f32` from `2^28.65` up with
/// `M < 2^74`. `A` in three parts, each product split exactly
/// (`residual`) and each part reduced mod 4 ([`rem4`]), summed in
/// double-double: the error is below `2^−58`, and `|y| ≥ 2^−29.2` for every
/// `f32` (at `7.729179e28`).
#[inline(always)]
fn reduce_f32_big<V: Float<Elem = f64>>(x: V, ax: V) -> (V, V) {
    // G = 5, then G = 55 from x ≥ 2^78 (e ≥ 55).
    const SCALE: [f64; 2] = [1.0 / 32.0, 1.0 / 36_028_797_018_963_968.0];
    const A1: [f64; 2] = [
        f64::from_bits(0x3fd7_cc1b_7272_20a9),
        f64::from_bits(0x4004_a7f0_9d5f_47d5),
    ];
    const A2: [f64; 2] = [
        f64::from_bits(0x3c73_f84e_afa3_ea6a),
        f64::from_bits(0xbc96_447e_493a_d4ce),
    ];
    const A3: [f64; 2] = [
        f64::from_bits(0xb911_1f92_4eb5_3362),
        f64::from_bits(0x393e_21c8_20ff_28b2),
    ];
    let [t55, toint, pio2] = ks(
        x,
        [
            302_231_454_903_657_293_676_544.0,
            TOINT,
            core::f64::consts::FRAC_PI_2,
        ],
    );
    let g55 = ax.ge(t55);
    let (a1, a2, a3) = (pick2(x, g55, A1), pick2(x, g55, A2), pick2(x, g55, A3));
    // Signed: any representative mod 4 will do.
    let m = x * pick2(x, g55, SCALE);

    let p1 = m * a1;
    let e1 = m.residual(a1, p1); // p1 − m·a1
    let p2 = m * a2;
    let e2 = m.residual(a2, p2);
    let (s, ea) = two_sum(rem4(p1), -rem4(e1));
    let (s, eb) = two_sum(s, rem4(p2));
    let lo = ((ea + eb) - e2) + m * a3;
    let n = (s + toint) - toint;
    (n, ((s - n) + lo) * pio2)
}

/// Per lane, `c[1]` where `hi`, else `c[0]`.
#[inline(always)]
fn pick2<V: Float<Elem = f64>>(x: V, hi: V::Mask, c: [f64; 2]) -> V {
    let [c0, c1] = ks(x, c);
    V::select(hi, c1, c0)
}

/// `v` minus the nearest multiple of 4, in `[−2, 2]`, exactly: a small `v`
/// stays as it is (`[0, 4)` would round `−2^−11` to `4`'s precision).
#[inline(always)]
fn rem4<V: Float<Elem = f64>>(v: V) -> V {
    let [quarter, half, four] = ks(v, [0.25, 0.5, 4.0]);
    v - four * v.mul_add(quarter, half).floor()
}

/// 128 bits of [`TWO_OVER_PI`] from bit offset `off` (`b_(off+1)` first).
fn two_over_pi_128(off: usize) -> u128 {
    let (word, shift) = (off / 64, (off % 64) as u32);
    let at = |i: usize| {
        if shift == 0 {
            TWO_OVER_PI[word + i]
        } else {
            (TWO_OVER_PI[word + i] << shift) | (TWO_OVER_PI[word + i + 1] >> (64 - shift))
        }
    };
    u128::from(at(0)) << 64 | u128::from(at(1))
}

/// The `f64` kernel.
#[inline(always)]
fn trig_f64<V: Float<Elem = f64>, F: TrigFn>(x: V) -> V {
    // musl e_rem_pio2.c: π/2 = PIO2_1 + PIO2_2 + PIO2_3 + PIO2_3T, each
    // PIO2_i with trailing zeros so that n·PIO2_i is exact.
    const PIO2_1: f64 = 1.570_796_326_734_125_614_17e0;
    const PIO2_2: f64 = 6.077_100_506_303_965_976_60e-11;
    const PIO2_2T: f64 = 2.022_266_248_795_950_631_54e-21;
    const PIO2_3: f64 = 2.022_266_248_711_166_455_80e-21;
    const PIO2_3T: f64 = 8.478_427_660_368_899_569_97e-32;
    /// `2^20·π/2` (`0x413921fb` high word): from here on, [`reduce_mid`].
    const LIMIT: f64 = f64::from_bits(0x4139_21fb_0000_0000);
    /// `2^33`: from here on, [`reduce`] (see [`reduce_mid`]).
    const LIMIT_MID: f64 = 8_589_934_592.0;

    let [
        inv_pio2,
        toint,
        p1,
        p2,
        p2t,
        p3,
        p3t,
        limit,
        limit_mid,
        two_m49,
    ] = ks(
        x,
        [
            INV_PIO2,
            TOINT,
            PIO2_1,
            PIO2_2,
            PIO2_2T,
            PIO2_3,
            PIO2_3T,
            LIMIT,
            LIMIT_MID,
            f64::from_bits((1023 - 49) << 52),
        ],
    );
    let n = x.mul_add(inv_pio2, toint) - toint;
    // π/2 = p1 + p2 + p2t = p1 + p2 + p3 + p3t, each n·p_i exact. Round 2
    // (musl's) keeps its subtraction's error in `w`, so it is safe for
    // every lane.
    let t = (-n).mul_add(p1, x);
    let w = n * p2;
    let r = t - w;
    let w = n.mul_add(p2t, -((t - r) - w));
    let y0 = r - w;
    let y1 = (r - y0) - w;
    // Round 3 drops round 2's error, which is exact only after heavy
    // cancellation: musl's condition, more than 49 bits lost. Rare: only
    // computed for registers that need it.
    let deep = y0.abs().lt(x.abs() * two_m49);
    let (y0, y1) = if V::any(deep) {
        let w3 = n * p3;
        let r3 = r - w3;
        let w3 = n.mul_add(p3t, -((r - r3) - w3));
        let y0_3 = r3 - w3;
        let y1_3 = (r3 - y0_3) - w3;
        (V::select(deep, y0_3, y0), V::select(deep, y1_3, y1))
    } else {
        (y0, y1)
    };

    let ax = x.abs();
    let mid = ax.ge(limit);
    let (y0, y1) = if V::any(mid) {
        let (m0, m1) = reduce_mid(x, n);
        (V::select(mid, m0, y0), V::select(mid, m1, y1))
    } else {
        (y0, y1)
    };
    let r = F::from_f64(n, y0, y1);
    let r = finish(x, ax.ge(limit_mid), r, f64_large::<F>);
    special::<_, F>(x, ax, r)
}

/// musl k_sin.c: `sin(y0 + y1)` for `|y0 + y1| ≤ π/4`.
#[inline(always)]
fn sin_f64_kernel<V: Float<Elem = f64>>(y0: V, y1: V) -> V {
    let [s1, s2, s3, s4, s5, s6, half] = ks(
        y0,
        [
            -1.666_666_666_666_663_243_48e-1,
            8.333_333_333_322_489_461_24e-3,
            -1.984_126_982_985_794_931_34e-4,
            2.755_731_370_707_006_767_89e-6,
            -2.505_076_025_340_686_341_95e-8,
            1.589_690_995_211_550_102_21e-10,
            0.5,
        ],
    );
    let z = y0 * y0;
    let w = z * z;
    let r = (z * w).mul_add(z.mul_add(s6, s5), z.mul_add(z.mul_add(s4, s3), s2));
    let v = z * y0;
    y0 - ((z * half.mul_add(y1, -(v * r)) - y1) - v * s1)
}

/// musl k_cos.c: `cos(y0 + y1)` for `|y0 + y1| ≤ π/4`.
#[inline(always)]
fn cos_f64_kernel<V: Float<Elem = f64>>(y0: V, y1: V) -> V {
    let [c1, c2, c3, c4, c5, c6, half, one] = ks(
        y0,
        [
            4.166_666_666_666_660_190_37e-2,
            -1.388_888_888_887_410_957_49e-3,
            2.480_158_728_947_672_941_78e-5,
            -2.755_731_435_139_066_330_35e-7,
            2.087_572_321_298_174_827_90e-9,
            -1.135_964_755_778_819_482_65e-11,
            0.5,
            1.0,
        ],
    );
    let z = y0 * y0;
    let w = z * z;
    let r = z * z.mul_add(z.mul_add(c3, c2), c1) + (w * w) * z.mul_add(z.mul_add(c6, c5), c4);
    let hz = half * z;
    let w = one - hz;
    w + (((one - w) - hz) + z.mul_add(r, -(y0 * y1)))
}

/// `F` of a lane beyond the `f64` path's reduction.
fn f64_large<F: TrigFn>(x: f64) -> f64 {
    let (n, y0, y1) = reduce(x);
    F::from_f64(n, y0, y1)
}

/// musl k_tanf.c, in `f64`: `tan y` for `|y| ≤ π/4`.
#[inline(always)]
fn tan_f32_kernel<V: Float<Elem = f64>>(y: V) -> V {
    let [t0, t1, t2, t3, t4, t5] = ks(
        y,
        [
            0.333_331_395_030_791_399_758,
            0.133_392_002_712_976_742_718,
            0.053_381_237_844_567_039_352_3,
            0.024_528_318_116_654_727_887_3,
            0.002_974_357_433_599_673_049_27,
            0.009_465_647_849_436_731_667_28,
        ],
    );
    let z = y * y;
    let r = z.mul_add(t5, t4);
    let t = z.mul_add(t3, t2);
    let w = z * z;
    let s = z * y;
    let u = z.mul_add(t1, t0);
    (s * w).mul_add(w.mul_add(r, t), s.mul_add(u, y))
}

/// musl k_tan.c: `tan(y0 + y1)` for `|y0 + y1| ≤ π/4`, or `−1/tan` where
/// `odd`.
#[inline(always)]
fn tan_f64_kernel<V: Float<Elem = f64>>(odd: V::Mask, y0: V, y1: V) -> V {
    let [t0, t1, t2, t3, t4, t5, t6] = ks(
        y0,
        [
            3.333_333_333_333_340_919_86e-1,
            1.333_333_333_332_012_426_99e-1,
            5.396_825_397_622_605_213_77e-2,
            2.186_948_829_485_954_245_99e-2,
            8.863_239_823_599_300_057_37e-3,
            3.592_079_107_591_312_353_56e-3,
            1.456_209_454_325_290_255_16e-3,
        ],
    );
    let [t7, t8, t9, t10, t11, t12] = ks(
        y0,
        [
            5.880_412_408_202_640_968_74e-4,
            2.464_631_348_184_699_068_12e-4,
            7.817_944_429_395_570_923_00e-5,
            7.140_724_913_826_081_903_05e-5,
            -1.855_863_748_552_754_566_54e-5,
            2.590_730_518_636_337_128_84e-5,
        ],
    );
    let [pio4, pio4lo, big_at, zero, one, two] = ks(
        y0,
        [
            core::f64::consts::FRAC_PI_4,
            3.061_616_997_868_383_017_93e-17,
            // 0.6744, musl's `0x3FE59428` high word.
            f64::from_bits(0x3fe5_9428_0000_0000),
            0.0,
            1.0,
            2.0,
        ],
    );
    // Above 0.6744, tan y from tan(π/4 − |y|).
    let neg = y0.is_sign_negative();
    let ay = y0.abs();
    let big = ay.ge(big_at);
    let x = V::select(big, (pio4 - ay) + (pio4lo - V::select(neg, -y1, y1)), y0);
    let y = V::select(big, zero, y1);

    let z = x * x;
    let w = z * z;
    let r = w.mul_add(
        w.mul_add(w.mul_add(w.mul_add(w.mul_add(t11, t9), t7), t5), t3),
        t1,
    );
    let v = z * w.mul_add(
        w.mul_add(w.mul_add(w.mul_add(w.mul_add(t12, t10), t8), t6), t4),
        t2,
    );
    let s = z * x;
    let r = s.mul_add(t0, z.mul_add(s.mul_add(r + v, y), y));
    let w = x + r;
    if !V::any(big) && !V::any(odd) {
        return w;
    }

    // One division per lane, for whichever form it needs: `w²/(w ± 1)`
    // (big) or `−1/w` (odd).
    let sg = V::select(odd, -one, one);
    let q = V::select(big, w * w, -one) / V::select(big, w + sg, w);
    // Big: (1 − tan)/(1 + tan) form, `±1` by parity, then the sign back.
    let vb = sg - two * (x + (r - q));
    let big_r = V::select(neg, -vb, vb);
    // Odd: −1/(x + r) to 2 ULP rounded is too coarse: `a = −1/w`, then one
    // correction with `1 + a·w` exact (`residual`) and `w`'s rounding error.
    let e = (x - w) + r;
    let t = q.mul_add(e, (-q).residual(w, one));
    let inv = q.mul_add(t, q);
    V::select(big, big_r, V::select(odd, inv, w))
}

/// The binary expansion of `2/π`, 64 bits per word, most significant first:
/// `2/π = 0.a2f9836e4e441529fc27…` (computed at 1 600 bits).
const TWO_OVER_PI: [u64; 22] = [
    0xa2f9_836e_4e44_1529,
    0xfc27_57d1_f534_ddc0,
    0xdb62_9599_3c43_9041,
    0xfe51_63ab_debb_c561,
    0xb724_6e3a_424d_d2e0,
    0x0649_2eea_09d1_921c,
    0xfe1d_eb1c_b129_a73e,
    0xe882_35f5_2ebb_4484,
    0xe99c_7026_b45f_7e41,
    0x3991_d639_8353_39f4,
    0x9c84_5f8b_bdf9_283b,
    0x1ff8_97ff_de05_980f,
    0xef2f_118b_5a0a_6d1f,
    0x6d36_7ecf_27cb_09b7,
    0x4f46_3f66_9e5f_ea2d,
    0x7527_bac7_ebe5_f17b,
    0x3d07_39f7_8a52_92ea,
    0x6bfb_5fb1_1f8d_5d08,
    0x5603_3046_fc7b_6bab,
    0xf0cf_bc20_9af4_361d,
    0xa9e3_9161_5ee6_1b08,
    0x6599_855f_14a0_6840,
];

/// `p << sh` for a 256-bit `p` (little-endian limbs), `sh < 256`.
fn shl256(p: [u64; 4], sh: u32) -> [u64; 4] {
    let (w, b) = ((sh / 64) as usize, sh % 64);
    let mut r = [0; 4];
    for (src, out) in r.iter_mut().skip(w).enumerate() {
        let carry = if b > 0 && src > 0 {
            p[src - 1] >> (64 - b)
        } else {
            0
        };
        *out = (p[src] << b) | carry;
    }
    r
}

/// `2^k` for `−1022 ≤ k ≤ 1023`.
fn exp2i(k: i32) -> f64 {
    f64::from_bits(((1023 + k) as u64) << 52)
}

/// `x = n·π/2 + y0 + y1` exactly enough for any finite `x`: `n` (mod 4, as
/// a float) and `|y0 + y1| ≤ π/4` to `2^−136` (Payne–Hanek): `2^−75`
/// relative at worst, as `|y| ≥ 2^−61` for every `f64`.
///
/// With `|x| = m·2^e`, `m` a 53-bit integer, the bits of `2/π` that would
/// only add multiples of 4 to `x·2/π` are skipped; a 192-bit window of the
/// rest times `m` (256 bits) holds the quadrant and the remainder.
pub(crate) fn reduce(x: f64) -> (f64, f64, f64) {
    if !x.is_finite() || x.abs() <= core::f64::consts::FRAC_PI_4 {
        return (0.0, x, 0.0);
    }
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32; // |x| > π/4: normal
    let m = (bits & ((1 << 52) - 1)) | (1 << 52);
    let e = exp - 1075; // |x| = m · 2^e, e ≥ −53

    // Bits b_k0 … b_(k0+191) of 2/π (b_k worth 2^−k); earlier ones give
    // multiples of 4.
    let k0 = (e - 1).max(1);
    let hi = two_over_pi_128(k0 as usize - 1);
    let w = [
        two_over_pi_128(k0 as usize + 127) >> 64,
        hi & u128::from(u64::MAX),
        hi >> 64,
    ];

    // p = m · w, 256 bits, little-endian 64-bit limbs.
    let mut p = [0_u64; 4];
    let mut carry = 0_u128;
    for i in 0..3 {
        let t = u128::from(m) * w[i] + carry;
        p[i] = t as u64;
        carry = t >> 64;
    }
    p[3] = carry as u64;

    // x·2/π mod 4 = p · 2^−s, s = k0 + 191 − e in [190, 245]: the quadrant
    // is bits s, s + 1; the fraction, moved to the top, is a signed
    // fixed-point number in [−½, ½), negative when rounding n up.
    let s = (k0 + 191 - e) as u32;
    let q = shl256(p, 254 - s);
    let mut n = (q[3] >> 62) as u32;
    let mut r = shl256(q, 2);
    let neg = r[3] >> 63 == 1;
    if neg {
        n += 1;
        // −r = !r + 1; the 1, worth 2^−256, does not matter.
        r = r.map(|l| !l);
    }

    // |f| = r · 2^−256 as hi + lo: its leading 106 bits. |f| ≥ 2^−62 for
    // every f64, but stay general.
    let lz = match r.iter().rposition(|&l| l != 0) {
        None => return (f64::from(n & 3), 0.0, 0.0),
        Some(i) => 64 * (3 - i as u32) + r[i].leading_zeros(),
    };
    let t = shl256(r, lz);
    let f_hi = (t[3] >> 11) as i64 as f64 * exp2i(-53 - lz as i32);
    let lo = ((t[3] & 0x7ff) << 42) | (t[2] >> 22);
    let f_lo = lo as i64 as f64 * exp2i(-106 - lz as i32);

    // y = f · π/2, in double-double.
    const PIO2_HI: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
    const PIO2_LO: f64 = f64::from_bits(0x3c91_a626_3314_5c07);
    let ph = f_hi * PIO2_HI;
    let pe = -f_hi.residual(PIO2_HI, ph); // f_hi·PIO2_HI − ph, exactly
    let pl = pe + (f_hi * PIO2_LO + f_lo * PIO2_HI);
    let y0 = ph + pl;
    let y1 = pl - (y0 - ph);
    let (y0, y1) = if neg { (-y0, -y1) } else { (y0, y1) };

    if x < 0.0 {
        (f64::from((4 - n) & 3), -y0, -y1)
    } else {
        (f64::from(n & 3), y0, y1)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{CosOp, MathElement, SinOp, TanOp, UnaryOp};
    use crate::testing::dd::{cos_dd, sin_dd, tan_dd};
    use crate::testing::{TestFloat, assert_exact, assert_ulps_dd, assert_ulps_every_f32, samples};
    use std::vec::Vec;

    /// `f64` inputs close to a multiple of `π/2`. The first five are below
    /// `2^20·π/2` and within `2^−70` (relative), where round 2 cancels:
    /// without round 3, `sin` of the third and `cos` of the fourth are off
    /// by 1.06 ULP, `tan` of the fourth by 2.17.
    const NEAR_PIO2: [f64; 7] = [
        642_615.918_884_445_8,
        1_285_231.837_768_891_6,
        826_882.894_388_101_5,
        413_441.447_194_050_76,
        321_307.959_442_222_9,
        // The closest in [2^20, 2^33), for the vector reduction above
        // 2^20·π/2: 2^−59 and 2^−53.5 away.
        14_461_176.670_278_38,
        5_848_238_277.541_742,
    ];

    /// Every binade up to the largest float (huge inputs included), and the
    /// neighbours of the reduction limits, of `π/4`, `π/2`, `π` and of
    /// [`NEAR_PIO2`].
    fn inputs<T: TestFloat>() -> Vec<T> {
        let max = if T::NAME == "f32" {
            f32::MAX.into()
        } else {
            f64::MAX
        };
        let limits = [421_657_440.0, f64::from_bits(0x4139_21fb_0000_0000)];
        let mut around = std::vec![
            core::f64::consts::FRAC_PI_4,
            core::f64::consts::FRAC_PI_2,
            core::f64::consts::PI
        ];
        around.extend(limits);
        around.extend(NEAR_PIO2);
        samples::<T>(150_000, max, &around)
            .into_iter()
            .filter(|v| v.to_f64().is_finite())
            .collect()
    }

    #[test]
    fn sin_within_1_ulp() {
        assert_ulps_dd::<f32, SinOp>("sin", &inputs(), sin_dd, 1.0);
        assert_ulps_dd::<f64, SinOp>("sin", &inputs(), sin_dd, 1.0);
    }

    #[test]
    fn cos_within_1_ulp() {
        assert_ulps_dd::<f32, CosOp>("cos", &inputs(), cos_dd, 1.0);
        assert_ulps_dd::<f64, CosOp>("cos", &inputs(), cos_dd, 1.0);
    }

    #[test]
    fn tan_within_1_ulp() {
        assert_ulps_dd::<f32, TanOp>("tan", &inputs(), tan_dd, 1.0);
        assert_ulps_dd::<f64, TanOp>("tan", &inputs(), tan_dd, 1.0);
    }

    /// The exact reduction, through the references, to `2^−74` (its bound,
    /// see [`reduce`](super::reduce)) against
    /// 3000-bit values: huge inputs, the `f64` input closest to an odd
    /// multiple of `π/2`, [`NEAR_PIO2`], and one where `sin` cancels to
    /// `−6.4e−11`.
    #[test]
    fn reference_is_precise() {
        let huge = 6_381_956_970_095_103.0 * 2f64.powi(797);
        let cases = [
            (1e22, -0.852_200_849_767_188_8, 0.523_214_785_395_139),
            (1e300, -0.817_881_912_115_908_5, -0.575_386_111_957_549_1),
            (
                f64::MAX,
                0.004_961_954_789_184_062,
                -0.999_987_689_426_559_9,
            ),
            (huge, 1.0, -4.687_165_924_254_628e-19),
            (1_647_099.329_165_285_5, -6.420_676_210_313_675e-11, 1.0),
            (NEAR_PIO2[0], 8.859_201_669_192_259e-17, -1.0),
            (NEAR_PIO2[1], -1.771_840_333_838_451_7e-16, 1.0),
            (NEAR_PIO2[2], -4.551_307_990_356_309e-16, -1.0),
            (NEAR_PIO2[3], 1.0, -2.275_653_995_178_154_4e-16),
            (NEAR_PIO2[4], -1.0, -4.429_600_834_596_129e-17),
            (NEAR_PIO2[5], -1.0, -1.698_503_829_898_600_4e-18),
            (NEAR_PIO2[6], 1.0, 7.992_064_984_149_294e-17),
            (1e6, -0.349_993_502_171_292_94, 0.936_752_127_533_144_7),
            (3.0, 0.141_120_008_059_867_2, -0.989_992_496_600_445_4),
            (0.75, 0.681_638_760_023_334_1, 0.731_688_868_873_820_9),
        ];
        let tan = [
            -1.628_778_225_606_898_8,
            1.421_448_823_874_724_5,
            -0.004_962_015_874_444_895,
            -2.133_485_385_753_704e18,
            -6.420_676_210_313_675e-11,
            -8.859_201_669_192_259e-17,
            -1.771_840_333_838_451_7e-16,
            4.551_307_990_356_309e-16,
            -4_394_341_152_560_466.0,
            2.257_539_758_864_469_6e16,
            5.887_534_560_694_51e17,
            1.251_241_077_222_602e16,
            -0.373_624_453_987_599,
            -0.142_546_543_074_277_8,
            0.931_596_459_944_072_5,
        ];
        // The values minus their rounding to f64, for double-double checks.
        let lo: [[f64; 3]; 15] = [
            [
                -6.780_682_589_677_328_4e-18,
                -4.714_320_107_657_516_4e-17,
                -1.011_493_425_412_603_4e-16,
            ],
            [
                -4.781_358_374_403_26e-17,
                2.677_076_191_878_706_8e-17,
                -7.511_471_696_020_065e-17,
            ],
            [
                -2.504_937_767_649_410_4e-19,
                -2.603_289_026_721_674_8e-17,
                2.410_586_573_485_649_5e-19,
            ],
            [
                -1.098_476_220_074_687e-37,
                4.372_055_742_938_273_3e-36,
                92.325_147_336_655_52,
            ],
            [
                1.570_165_989_482_554e-27,
                -2.061_254_149_884_398_8e-21,
                1.570_033_643_027_718_3e-27,
            ],
            [
                4.311_411_351_827_589e-33,
                3.924_272_710_770_946e-33,
                -4.311_411_351_827_589e-33,
            ],
            [
                -8.622_822_703_655_177e-33,
                -1.569_709_084_308_378_3e-32,
                -8.622_822_703_655_18e-33,
            ],
            [
                6.219_871_892_327_475e-33,
                1.035_720_221_154_059e-31,
                -6.219_871_892_327_428e-33,
            ],
            [
                -2.589_300_552_885_147_5e-32,
                3.109_935_946_163_732e-33,
                -0.082_935_409_337_699_41,
            ],
            [
                9.810_681_776_927_364e-34,
                -2.155_705_675_913_794_5e-33,
                1.066_290_386_052_789_3,
            ],
            [
                1.442_457_630_090_106_8e-36,
                3.029_174_338_658_756e-36,
                55.313_769_241_792,
            ],
            [
                -3.193_655_135_543_262e-33,
                -6.027_957_256_327_615e-33,
                0.750_601_744_556_653_1,
            ],
            [
                -1.595_284_880_932_396_8e-17,
                4.637_088_260_214_747e-17,
                -3.578_451_556_475_479e-18,
            ],
            [
                8.577_269_787_017_502e-18,
                -4.206_026_156_609_973_4e-17,
                -1.387_034_906_787_784_3e-18,
            ],
            [
                4.410_467_313_197_903e-17,
                -1.047_582_430_651_276_8e-17,
                -1.354_738_139_659_303_6e-17,
            ],
        ];
        for (((x, sin, cos), tan), lo) in cases.into_iter().zip(tan).zip(lo) {
            for (name, f, want) in [
                ("sin_dd", sin_dd as fn(f64) -> (f64, f64), (sin, lo[0])),
                ("cos_dd", cos_dd, (cos, lo[1])),
                ("tan_dd", tan_dd, (tan, lo[2])),
            ] {
                let (hi, lo) = f(x);
                let err = ((hi - want.0) + (lo - want.1)).abs();
                assert!(
                    err <= want.0.abs() * 2f64.powi(-74),
                    "{name}({x:e}): {err:e}"
                );
            }
        }
    }

    /// [`reduce_f32_big`](super::reduce_f32_big) at the six `f32` inputs
    /// closest to a multiple of `π/2`, against 3000-bit values: `n mod 4`,
    /// and `y` to `2^−50`.
    #[test]
    fn f32_reduction_is_precise() {
        for (x, n, y) in [
            (7.729_178_919_452_902e28, 1, 1.614_769_798_247_621_1e-9),
            (21_999_384_576.0, 1, 2.012_646_031_918_552_8e-9),
            (1.545_835_783_890_580_4e29, 2, 3.229_539_596_495_242_2e-9),
            (43_998_769_152.0, 2, 4.025_292_063_837_105_5e-9),
            (4.638_183_382_783_466e25, 1, 6.467_856_984_570_355e-9),
            (3.091_671_567_781_160_8e29, 0, 6.459_079_192_990_484_5e-9),
        ] {
            for (x, n, y) in [(x, n, y), (-x, 4 - n, -y)] {
                let (got_n, got_y) = super::reduce_f32_big(x, f64::abs(x));
                assert_eq!((got_n as i64).rem_euclid(4), n % 4, "n of {x:e}");
                let err = (got_y - y).abs();
                assert!(err <= y.abs() * 2f64.powi(-50), "y of {x:e}: {err:e}");
            }
        }
    }

    /// `F` of huge `f32` inputs, which take the exact reduction, on every
    /// backend.
    fn huge_f32<F: UnaryOp>(name: &str, cases: [(f32, f32); 6]) {
        for (backend, got) in f32::every_backend::<F>(&cases.map(|c| c.0)) {
            for (i, (x, want)) in cases.iter().enumerate() {
                assert_eq!(got[i], *want, "{name}({x:e}) on {backend}");
            }
        }
    }

    #[test]
    fn huge_f32_inputs() {
        // The last: `cos` is −1.6e−9, which a reduction mod 4 into [0, 4)
        // got wrong by 2.7 ULP.
        let x = [
            1e38,
            f32::MAX,
            1e10,
            421_657_440.0,
            12_345_678.0,
            7.729_179e28,
        ];
        let sin = [
            0.989_164_5,
            -0.521_876_5,
            -0.487_506_03,
            -0.739_739_9,
            -0.903_541_15,
            1.0,
        ];
        let cos = [
            0.146_811_68,
            0.853_021,
            0.873_119_65,
            0.672_892_9,
            -0.428_501_34,
            -1.614_769_8e-9,
        ];
        huge_f32::<SinOp>("sin", core::array::from_fn(|i| (x[i], sin[i])));
        huge_f32::<CosOp>("cos", core::array::from_fn(|i| (x[i], cos[i])));
        let tan = [
            6.737_641_3,
            -0.611_797_9,
            -0.558_349_6,
            -1.099_342_6,
            2.108_607_5,
            -6.192_833e8,
        ];
        huge_f32::<TanOp>("tan", core::array::from_fn(|i| (x[i], tan[i])));
    }

    #[test]
    fn special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let sin = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (inf, nan),
            (-inf, nan),
            (nan, nan),
        ];
        let cos = [(0.0, 1.0), (-0.0, 1.0), (inf, nan), (-inf, nan), (nan, nan)];
        assert_exact::<f64, SinOp>("sin", &sin);
        assert_exact::<f32, SinOp>("sin", &sin.map(|(x, y)| (x as f32, y as f32)));
        assert_exact::<f64, CosOp>("cos", &cos);
        assert_exact::<f32, CosOp>("cos", &cos.map(|(x, y)| (x as f32, y as f32)));
        assert_exact::<f64, TanOp>("tan", &sin);
        assert_exact::<f32, TanOp>("tan", &sin.map(|(x, y)| (x as f32, y as f32)));
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive` (against the
    /// platform's `f64` `sin`, `cos` and `tan`).
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn sin_f32_exhaustive() {
        assert_ulps_every_f32::<SinOp>("sin", f64::sin, 1.0);
    }

    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn cos_f32_exhaustive() {
        assert_ulps_every_f32::<CosOp>("cos", f64::cos, 1.0);
    }

    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn tan_f32_exhaustive() {
        assert_ulps_every_f32::<TanOp>("tan", f64::tan, 1.0);
    }
}
