//! Test-only helpers shared by every kernel's unit tests: IEEE-754 special
//! values and bit-exact comparison; ULP distance and input sweeps are planned.

use core::fmt::Debug;

pub(crate) trait TestFloat: Copy + PartialEq + Debug {
    const NAME: &str;
    /// Signed zeros, infinities, NaN, subnormals and one.
    const SPECIAL: [Self; 8];
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn is_nan(self) -> bool;
    fn bits(self) -> u64;
    /// The sign bit, in [`bits`](Self::bits).
    const SIGN: u64 = 1 << (core::mem::size_of::<Self>() * 8 - 1);
}

macro_rules! test_float {
    ($($T:ident: $subnormal:expr),*) => {
        $(
            impl TestFloat for $T {
                const NAME: &str = stringify!($T);
                const SPECIAL: [Self; 8] = [
                    0.0,
                    -0.0,
                    $T::INFINITY,
                    -$T::INFINITY,
                    $T::NAN,
                    $subnormal,
                    -$subnormal,
                    1.0,
                ];

                fn from_f64(x: f64) -> Self {
                    x as $T
                }

                fn to_f64(self) -> f64 {
                    self.into()
                }

                fn is_nan(self) -> bool {
                    $T::is_nan(self)
                }

                fn bits(self) -> u64 {
                    self.to_bits().into()
                }
            }
        )*
    };
}

test_float!(f32: 1e-40, f64: 1e-310);

/// Bit-exact equality, except that any two NaNs match.
pub(crate) fn assert_same<T: TestFloat>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let same = g.bits() == w.bits() || (g.is_nan() && w.is_nan());
        assert!(same, "{what}, index {i}: got {g:?}, want {w:?}");
    }
}
