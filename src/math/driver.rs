//! Slice drivers: apply a register-level operation to a whole slice, as full
//! registers plus one masked tail. Generic over [`SimdVector`], so every
//! function and backend shares them.
//!
//! Drivers are `#[inline(always)]`: inlined into a `#[target_feature]` kernel,
//! the register operations compile to that ISA's instructions.
//!
//! ```rust,ignore
//! #[target_feature(enable = "avx2,fma")]
//! pub(crate) fn exp(x: &[f32], out: &mut [f32]) {
//!     // SAFETY: this function requires AVX2 and FMA.
//!     unsafe { driver::unary::<F32x8>(x, out, exp_kernel) }
//! }
//! ```

#![allow(dead_code, reason = "used by the math and ops functions once they land")]

use crate::simd::SimdVector;

/// `out[i] = op(x[i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `x` and `out` have different lengths.
#[inline(always)]
pub(crate) unsafe fn unary<S: SimdVector>(
    x: &[S::Scalar],
    out: &mut [S::Scalar],
    op: impl Fn(S) -> S,
) {
    assert_eq!(x.len(), out.len(), "length mismatch");
    // SAFETY: the caller guarantees the ISA; both slices hold `x.len()`
    // elements.
    unsafe { drive(x.len(), [x.as_ptr()], out.as_mut_ptr(), |[a]| op(a)) }
}

/// `x[i] = op(x[i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
#[inline(always)]
pub(crate) unsafe fn unary_in_place<S: SimdVector>(x: &mut [S::Scalar], op: impl Fn(S) -> S) {
    let ptr = x.as_mut_ptr();
    // SAFETY: the caller guarantees the ISA; `ptr` is valid for `x.len()`
    // elements, and each chunk is read before it is overwritten.
    unsafe { drive(x.len(), [ptr.cast_const()], ptr, |[a]| op(a)) }
}

/// `out[i] = op(a[i], b[i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `a`, `b` and `out` do not all have the same length.
#[inline(always)]
pub(crate) unsafe fn binary<S: SimdVector>(
    a: &[S::Scalar],
    b: &[S::Scalar],
    out: &mut [S::Scalar],
    op: impl Fn(S, S) -> S,
) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    assert_eq!(a.len(), out.len(), "length mismatch");
    // SAFETY: the caller guarantees the ISA; all three slices hold `a.len()`
    // elements.
    unsafe {
        drive(a.len(), [a.as_ptr(), b.as_ptr()], out.as_mut_ptr(), |[a, b]| {
            op(a, b)
        })
    }
}

/// `a[i] = op(a[i], b[i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `a` and `b` have different lengths.
#[inline(always)]
pub(crate) unsafe fn binary_in_place<S: SimdVector>(
    a: &mut [S::Scalar],
    b: &[S::Scalar],
    op: impl Fn(S, S) -> S,
) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let ptr = a.as_mut_ptr();
    // SAFETY: the caller guarantees the ISA; both slices hold `a.len()`
    // elements, and each chunk of `a` is read before it is overwritten.
    unsafe { drive(a.len(), [ptr.cast_const(), b.as_ptr()], ptr, |[a, b]| op(a, b)) }
}

/// The loop behind every driver: `dst[i] = op(src[0][i], .., src[N-1][i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA, every `src` pointer must be valid for
/// `len` reads and `dst` for `len` writes. `dst` may equal a `src` pointer,
/// but must not otherwise overlap one.
#[inline(always)]
unsafe fn drive<S: SimdVector, const N: usize>(
    len: usize,
    src: [*const S::Scalar; N],
    dst: *mut S::Scalar,
    op: impl Fn([S; N]) -> S,
) {
    let full = len - len % S::LANES;
    let mut i = 0;
    while i < full {
        // SAFETY: the caller guarantees the ISA, and `i + LANES <= len`.
        unsafe { op(src.map(|p| S::load(p.add(i)))).store(dst.add(i)) };
        i += S::LANES;
    }

    let tail = len - full;
    if tail > 0 {
        // SAFETY: the caller guarantees the ISA; `tail < LANES` elements
        // remain from `full`.
        unsafe {
            op(src.map(|p| S::load_partial(p.add(full), tail))).store_partial(dst.add(full), tail)
        };
    }
}

#[cfg(test)]
mod tests {
    cfg_x86! {
        use super::*;
        use crate::detect::has_avx2_fma;
        use crate::simd::avx2::f32x8::F32x8;
        use std::panic::{AssertUnwindSafe, catch_unwind};
        use std::vec::Vec;

        const SENTINEL: f32 = 42.0;

        /// Lengths covering empty, tail-only, exact multiples and full + tail.
        fn lengths() -> impl Iterator<Item = usize> {
            0..=40
        }

        fn input(len: usize, seed: f32) -> Vec<f32> {
            (0..len).map(|i| seed + i as f32 * 0.75 - 9.0).collect()
        }

        /// A buffer of `len + 8` sentinels: writes past `len` stay visible.
        fn output(len: usize) -> Vec<f32> {
            std::vec![SENTINEL; len + 8]
        }

        fn assert_untouched_past(buf: &[f32], len: usize) {
            assert!(buf[len..].iter().all(|&v| v == SENTINEL), "wrote past {len}");
        }

        #[test]
        fn unary_matches_scalar() {
            if !has_avx2_fma() {
                return;
            }
            // SAFETY: AVX2 + FMA checked above.
            let (m, c) = unsafe { (F32x8::splat(1.5), F32x8::splat(0.25)) };
            for len in lengths() {
                let x = input(len, 1.0);
                let mut out = output(len);
                // SAFETY: AVX2 + FMA checked above.
                unsafe { unary::<F32x8>(&x, &mut out[..len], |v| v * m + c) };
                for i in 0..len {
                    assert_eq!(out[i], x[i] * 1.5 + 0.25, "len {len}, index {i}");
                }
                assert_untouched_past(&out, len);
            }
        }

        #[test]
        fn binary_matches_scalar() {
            if !has_avx2_fma() {
                return;
            }
            for len in lengths() {
                let (a, b) = (input(len, 1.0), input(len, -3.0));
                let mut out = output(len);
                // SAFETY: AVX2 + FMA checked above.
                unsafe { binary::<F32x8>(&a, &b, &mut out[..len], |a, b| a * b - a) };
                for i in 0..len {
                    assert_eq!(out[i], a[i] * b[i] - a[i], "len {len}, index {i}");
                }
                assert_untouched_past(&out, len);
            }
        }

        #[test]
        fn in_place_variants_match_out_of_place() {
            if !has_avx2_fma() {
                return;
            }
            for len in lengths() {
                let (a, b) = (input(len, 1.0), input(len, -3.0));
                let (mut want_u, mut want_b) = (std::vec![0.0; len], std::vec![0.0; len]);
                let mut got_u = output(len);
                got_u[..len].copy_from_slice(&a);
                let mut got_b = output(len);
                got_b[..len].copy_from_slice(&a);
                // SAFETY: AVX2 + FMA checked above.
                unsafe {
                    unary::<F32x8>(&a, &mut want_u, |v| -v / v);
                    unary_in_place::<F32x8>(&mut got_u[..len], |v| -v / v);
                    binary::<F32x8>(&a, &b, &mut want_b, |a, b| a / b);
                    binary_in_place::<F32x8>(&mut got_b[..len], &b, |a, b| a / b);
                }
                let same = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(same(&got_u[..len], &want_u), "unary_in_place, len {len}");
                assert!(same(&got_b[..len], &want_b), "binary_in_place, len {len}");
                assert_untouched_past(&got_u, len);
                assert_untouched_past(&got_b, len);
            }
        }

        #[test]
        fn length_mismatch_panics() {
            if !has_avx2_fma() {
                return;
            }
            let (x, mut out, mut a) = ([1.0; 9], [0.0; 8], [1.0; 9]);
            let panics = |f: &mut dyn FnMut()| catch_unwind(AssertUnwindSafe(f)).is_err();
            // AVX2 + FMA are checked above, and each call panics on the length
            // check before touching memory.
            assert!(panics(&mut || {
                // SAFETY: see above.
                unsafe { unary::<F32x8>(&x, &mut out, |v| v) }
            }));
            assert!(panics(&mut || {
                // SAFETY: see above.
                unsafe { binary::<F32x8>(&x, &x[..8], &mut out[..8], |a, _| a) }
            }));
            assert!(panics(&mut || {
                // SAFETY: see above.
                unsafe { binary_in_place::<F32x8>(&mut a, &x[..8], |a, _| a) }
            }));
        }
    }
}
