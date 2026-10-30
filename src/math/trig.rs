//! `sin x`, ≤ 1 ULP on every backend, for every finite input: musl's kernels
//! and argument reduction (from FreeBSD's `k_sin.c`, `k_cos.c`, `k_sinf.c`,
//! `k_cosf.c`, `e_rem_pio2.c`, `e_rem_pio2f.c`), written once over
//! [`Float`].
//!
//! `x = n·π/2 + y`, `|y| ≤ π/4`, then `sin x` is `±sin y` or `±cos y` by
//! `n mod 4`; a register computes both polynomials and selects per lane.
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
use super::float::{Element, Float, WideFn, Widen};

/// A lane type with a sine.
pub(crate) trait Sin {
    fn sin(self) -> Self;
}

/// `sin` for the lane types of one element type.
pub(crate) trait SinKernel<V> {
    fn sin(x: V) -> V;
}

/// No closures or `impl Fn`: see [`acos`](super::acos).
impl<V: Float> Sin for V
where
    V::Elem: SinKernel<V>,
{
    #[inline(always)]
    fn sin(self) -> Self {
        V::Elem::sin(self)
    }
}

impl<V: Widen> SinKernel<V> for f32 {
    #[inline(always)]
    fn sin(x: V) -> V {
        x.via_f64::<SinF32>()
    }
}

impl<V: Float<Elem = f64>> SinKernel<V> for f64 {
    #[inline(always)]
    fn sin(x: V) -> V {
        sin_f64(x)
    }
}

/// `2/π`, rounded.
const INV_PIO2: f64 = core::f64::consts::FRAC_2_PI;
/// `1.5 · 2^52`: adding and subtracting it rounds to an integer.
const TOINT: f64 = 6_755_399_441_055_744.0;

/// Per lane: `s` if `n mod 4 = 0`, `c` if 1, `−s` if 2, `−c` if 3.
#[inline(always)]
fn by_quadrant<V: Float>(n: V, s: V, c: V) -> V {
    let [quarter, one, two, four] = ks(n, [0.25, 1.0, 2.0, 4.0].map(V::Elem::from_f32));
    let q = n - four * (n * quarter).floor();
    V::select(
        q.eq(one),
        c,
        V::select(q.eq(two), -s, V::select(q.lt(one), s, -c)),
    )
}

/// `sin` of an `f32` input, in `f64`: the `f32` kernel.
struct SinF32;

impl WideFn for SinF32 {
    #[inline(always)]
    fn call<W: Float<Elem = f64>>(x: W) -> W {
        // musl e_rem_pio2f.c: π/2 = PIO2_1 (25 bits) + PIO2_1T.
        const PIO2_1: f64 = 1.570_796_310_901_641_845_70e0;
        const PIO2_1T: f64 = 1.589_325_477_352_819_669_16e-8;
        /// `2^28·π/2` as an `f32` (`0x4dc90fdb`): from here on, [`reduce`].
        const LIMIT: f64 = 421_657_440.0;

        let [inv_pio2, toint, pio2_1, pio2_1t, limit] =
            ks(x, [INV_PIO2, TOINT, PIO2_1, PIO2_1T, LIMIT]);
        let n = x.mul_add(inv_pio2, toint) - toint;
        let y = (-n).mul_add(pio2_1t, (-n).mul_add(pio2_1, x));
        let r = by_quadrant(n, sin_f32_kernel(y), cos_f32_kernel(y));

        let ax = x.abs();
        let r = finish(x, ax.ge(limit), r, sin_f32_large);
        // The polynomial turns `−0` into `+0`.
        let r = W::select(x.eq(k(x, 0.0)), x, r);
        nan_unless_finite(x, ax, r)
    }
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

/// `sin` of a lane beyond the `f32` path's reduction (an `f32` value).
fn sin_f32_large(x: f64) -> f64 {
    let (n, y0, y1) = reduce(x);
    let y = y0 + y1;
    by_quadrant(n, sin_f32_kernel(y), cos_f32_kernel(y))
}

/// The `f64` kernel.
#[inline(always)]
fn sin_f64<V: Float<Elem = f64>>(x: V) -> V {
    // musl e_rem_pio2.c: π/2 = PIO2_1 + PIO2_2 + PIO2_3 + PIO2_3T, each
    // PIO2_i with trailing zeros so that n·PIO2_i is exact.
    const PIO2_1: f64 = 1.570_796_326_734_125_614_17e0;
    const PIO2_2: f64 = 6.077_100_506_303_965_976_60e-11;
    const PIO2_2T: f64 = 2.022_266_248_795_950_631_54e-21;
    const PIO2_3: f64 = 2.022_266_248_711_166_455_80e-21;
    const PIO2_3T: f64 = 8.478_427_660_368_899_569_97e-32;
    /// `2^20·π/2` (`0x413921fb` high word): from here on, [`reduce`].
    const LIMIT: f64 = f64::from_bits(0x4139_21fb_0000_0000);

    let [inv_pio2, toint, p1, p2, p2t, p3, p3t, limit, two_m49] = ks(
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
    // cancellation: musl's condition, more than 49 bits lost.
    let w3 = n * p3;
    let r3 = r - w3;
    let w3 = n.mul_add(p3t, -((r - r3) - w3));
    let y0_3 = r3 - w3;
    let y1_3 = (r3 - y0_3) - w3;
    let deep = y0.abs().lt(x.abs() * two_m49);
    let (y0, y1) = (V::select(deep, y0_3, y0), V::select(deep, y1_3, y1));

    let r = by_quadrant(n, sin_f64_kernel(y0, y1), cos_f64_kernel(y0, y1));
    let ax = x.abs();
    let r = finish(x, ax.ge(limit), r, sin_f64_large);
    nan_unless_finite(x, ax, r)
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

/// `sin` of a lane beyond the `f64` path's reduction.
fn sin_f64_large(x: f64) -> f64 {
    let (n, y0, y1) = reduce(x);
    by_quadrant(n, sin_f64_kernel(y0, y1), cos_f64_kernel(y0, y1))
}

/// `2^k` for `|k| ≤ 2044`, in two normal factors (`core` has no `powi`).
fn pow2(k: i32) -> f64 {
    let p = |k: i32| f64::from_bits(((1023 + k) as u64) << 52);
    p(k / 2) * p(k - k / 2)
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

/// `x = n·π/2 + y0 + y1` exactly enough for any finite `x`: `n` (mod 4, as
/// a float) and `|y0 + y1| ≤ π/4` to about 106 bits (Payne–Hanek).
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

    // Bits b_k0 … b_(k0+191) of 2/π; earlier ones give multiples of 4.
    let k0 = (e - 1).max(1) as usize;
    let window = |i: usize| {
        let off = k0 - 1 + 64 * i;
        let (word, shift) = (off / 64, off % 64);
        if shift == 0 {
            TWO_OVER_PI[word]
        } else {
            (TWO_OVER_PI[word] << shift) | (TWO_OVER_PI[word + 1] >> (64 - shift))
        }
    };
    let w = [window(0), window(1), window(2)];

    // p = m · w, 256 bits, little-endian 64-bit limbs.
    let mut p = [0_u64; 4];
    let mut carry = 0_u128;
    for i in 0..3 {
        let t = u128::from(m) * u128::from(w[2 - i]) + carry;
        p[i] = t as u64;
        carry = t >> 64;
    }
    p[3] = carry as u64;

    // x·2/π = p · 2^−s, s = k0 + 191 − e.
    let s = (k0 as i32 + 191 - e) as u32;
    let bit = |i: u32| (p[(i / 64) as usize] >> (i % 64)) & 1;
    let mut n = (bit(s) + 2 * bit(s + 1)) as u32;

    // The fraction f = (p mod 2^s) / 2^s, moved to [−½, ½).
    let mut frac = p;
    for (i, limb) in frac.iter_mut().enumerate() {
        let lo = 64 * i as u32;
        if lo >= s {
            *limb = 0;
        } else if s - lo < 64 {
            *limb &= (1 << (s - lo)) - 1;
        }
    }
    let half = bit(s - 1) == 1;
    if half {
        // f − 1 = −(2^s − p mod 2^s): negate within s bits.
        n = (n + 1) & 3;
        let mut borrow = 0_u64;
        let mut neg = [0_u64; 4];
        for i in 0..4 {
            let (a, b1) = 0_u64.overflowing_sub(frac[i]);
            let (a, b2) = a.overflowing_sub(borrow);
            neg[i] = a;
            borrow = u64::from(b1 || b2);
        }
        frac = neg;
        for (i, limb) in frac.iter_mut().enumerate() {
            let lo = 64 * i as u32;
            if lo >= s {
                *limb = 0;
            } else if s - lo < 64 {
                *limb &= (1 << (s - lo)) - 1;
            }
        }
    }

    // |f| as hi + lo: its leading 106 bits.
    let top = (0..4).rev().find(|&i| frac[i] != 0);
    let (f_hi, f_lo) = match top {
        None => (0.0, 0.0),
        Some(i) => {
            let hb = 64 * i as i32 + 63 - frac[i].leading_zeros() as i32; // leading bit
            let get = |from: i32| -> u64 {
                // 53 bits starting at bit `from` going down (from = top bit).
                let mut v = 0_u64;
                for j in 0..53 {
                    let b = from - j;
                    let set = b >= 0 && (frac[(b / 64) as usize] >> (b % 64)) & 1 == 1;
                    v = (v << 1) | u64::from(set);
                }
                v
            };
            let hi = get(hb) as f64 * pow2(hb - 52 - s as i32);
            let lo = get(hb - 53) as f64 * pow2(hb - 105 - s as i32);
            (hi, lo)
        }
    };

    // y = f · π/2, in double-double.
    const PIO2_HI: f64 = f64::from_bits(0x3ff9_21fb_5444_2d18);
    const PIO2_LO: f64 = f64::from_bits(0x3c91_a626_3314_5c07);
    let ph = f_hi * PIO2_HI;
    let pe = -f_hi.residual(PIO2_HI, ph); // f_hi·PIO2_HI − ph, exactly
    let pl = pe + (f_hi * PIO2_LO + f_lo * PIO2_HI);
    let y0 = ph + pl;
    let y1 = pl - (y0 - ph);
    let (y0, y1) = if half { (-y0, -y1) } else { (y0, y1) };

    if x < 0.0 {
        (f64::from((4 - n) & 3), -y0, -y1)
    } else {
        (f64::from(n), y0, y1)
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{MathElement, SinOp};
    use crate::testing::dd::sin_dd;
    use crate::testing::{TestFloat, assert_exact, assert_ulps_dd, assert_ulps_every_f32, samples};
    use std::vec::Vec;

    /// `f64` inputs below the `f64` reduction limit within `2^−70` (relative)
    /// of a multiple of `π/2`, where round 2 cancels: without round 3 the
    /// last is off by 1.06 ULP.
    const NEAR_PIO2: [f64; 3] = [
        642_615.918_884_445_8,
        1_285_231.837_768_891_6,
        826_882.894_388_101_5,
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

    /// The exact reduction, through `sin_dd`, against 3000-bit values:
    /// huge inputs, the `f64` input closest to an odd multiple of `π/2`, and
    /// one where `sin` cancels to `−6.4e−11`.
    #[test]
    fn sin_reference_is_precise() {
        for (x, want) in [
            (1e22, -0.852_200_849_767_188_8),
            (1e300, -0.817_881_912_115_908_5),
            (f64::MAX, 0.004_961_954_789_184_062),
            (6_381_956_970_095_103.0 * 2f64.powi(797), 1.0),
            (1_647_099.329_165_285_5, -6.420_676_210_313_675e-11),
            (NEAR_PIO2[0], 8.859_201_669_192_259e-17),
            (NEAR_PIO2[1], -1.771_840_333_838_451_7e-16),
            (NEAR_PIO2[2], -4.551_307_990_356_309e-16),
            (1e6, -0.349_993_502_171_292_94),
            (3.0, 0.141_120_008_059_867_2),
            (0.75, 0.681_638_760_023_334_1),
        ] {
            let (hi, lo) = sin_dd(x);
            assert_eq!(hi + lo, want, "sin_dd({x:e})");
        }
    }

    /// Huge `f32` inputs, which take the exact reduction, on every backend.
    #[test]
    fn sin_huge_f32() {
        let cases = [
            (1e38_f32, 0.989_164_5_f32),
            (f32::MAX, -0.521_876_5),
            (1e10, -0.487_506_03),
            (421_657_440.0, -0.739_739_9),
            (12_345_678.0, -0.903_541_15),
        ];
        for (backend, got) in f32::every_backend::<SinOp>(&cases.map(|c| c.0)) {
            for (i, (x, want)) in cases.iter().enumerate() {
                assert_eq!(got[i], *want, "sin({x:e}) on {backend}");
            }
        }
    }

    #[test]
    fn sin_special_values() {
        let (inf, nan) = (f64::INFINITY, f64::NAN);
        let cases = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (inf, nan),
            (-inf, nan),
            (nan, nan),
        ];
        assert_exact::<f64, SinOp>("sin", &cases);
        let cases32 = cases.map(|(x, y)| (x as f32, y as f32));
        assert_exact::<f32, SinOp>("sin", &cases32);
    }

    /// Every `f32`, on every backend:
    /// `cargo test --release -- --ignored exhaustive` (against the
    /// platform's `f64` `sin`).
    #[test]
    #[ignore = "slow: 2^32 inputs per backend"]
    fn sin_f32_exhaustive() {
        assert_ulps_every_f32::<SinOp>("sin", f64::sin, 1.0);
    }
}
