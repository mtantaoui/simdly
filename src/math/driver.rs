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
//!     unsafe { driver::unary::<F32x8, ExpOp>(x, out) }
//! }
//! ```
//!
//! The operation is a type ([`UnaryFn`], [`BinaryFn`], [`TernaryFn`]), not a closure or
//! `impl Fn`: either would go through a function compiled without the
//! kernel's target features, and unless LLVM inlines it, every intrinsic in
//! a large operation (`acos`, …) becomes a call.

#![allow(dead_code, reason = "used by the math and ops functions once they land")]

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use crate::simd::SimdVector;

/// A register operation `S -> S`, applied by the unary drivers.
pub(crate) trait UnaryFn<S> {
    fn call(x: S) -> S;
}

/// A register operation `(S, S) -> S`, applied by the binary drivers.
pub(crate) trait BinaryFn<S> {
    fn call(a: S, b: S) -> S;
}

/// A register operation `(S, S, S) -> S`, applied by the ternary drivers.
pub(crate) trait TernaryFn<S> {
    fn call(a: S, b: S, c: S) -> S;
}

/// `out[i] = F(x[i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `x` and `out` have different lengths.
#[inline(always)]
pub(crate) unsafe fn unary<S: SimdVector, F: UnaryFn<S>>(
    x: &[S::Scalar],
    out: &mut [S::Scalar],
) {
    assert_eq!(x.len(), out.len(), "length mismatch");
    // SAFETY: the caller guarantees the ISA; both slices hold `x.len()`
    // elements.
    unsafe { drive(x.len(), [x.as_ptr()], out.as_mut_ptr(), Unary::<F>(PhantomData)) }
}

/// [`unary`] into uninitialised memory: on return all of `out` is
/// initialised.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `x` and `out` have different lengths.
#[inline(always)]
pub(crate) unsafe fn unary_uninit<S: SimdVector, F: UnaryFn<S>>(
    x: &[S::Scalar],
    out: &mut [MaybeUninit<S::Scalar>],
) {
    assert_eq!(x.len(), out.len(), "length mismatch");
    // SAFETY: the caller guarantees the ISA; both slices hold `x.len()`
    // elements, and `MaybeUninit<T>` has the layout of `T`.
    unsafe { drive(x.len(), [x.as_ptr()], out.as_mut_ptr().cast(), Unary::<F>(PhantomData)) }
}

/// `x[i] = F(x[i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
#[inline(always)]
pub(crate) unsafe fn unary_in_place<S: SimdVector, F: UnaryFn<S>>(x: &mut [S::Scalar]) {
    let ptr = x.as_mut_ptr();
    // SAFETY: the caller guarantees the ISA; `ptr` is valid for `x.len()`
    // elements, and each chunk is read before it is overwritten.
    unsafe { drive(x.len(), [ptr.cast_const()], ptr, Unary::<F>(PhantomData)) }
}

/// `out[i] = F(a[i], b[i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `a`, `b` and `out` do not all have the same length.
#[inline(always)]
pub(crate) unsafe fn binary<S: SimdVector, F: BinaryFn<S>>(
    a: &[S::Scalar],
    b: &[S::Scalar],
    out: &mut [S::Scalar],
) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    assert_eq!(a.len(), out.len(), "length mismatch");
    // SAFETY: the caller guarantees the ISA; all three slices hold `a.len()`
    // elements.
    unsafe { drive(a.len(), [a.as_ptr(), b.as_ptr()], out.as_mut_ptr(), Binary::<F>(PhantomData)) }
}

/// [`binary`] into uninitialised memory: on return all of `out` is
/// initialised.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `a`, `b` and `out` do not all have the same length.
#[inline(always)]
pub(crate) unsafe fn binary_uninit<S: SimdVector, F: BinaryFn<S>>(
    a: &[S::Scalar],
    b: &[S::Scalar],
    out: &mut [MaybeUninit<S::Scalar>],
) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    assert_eq!(a.len(), out.len(), "length mismatch");
    // SAFETY: the caller guarantees the ISA; all three slices hold `a.len()`
    // elements, and `MaybeUninit<T>` has the layout of `T`.
    unsafe {
        drive(a.len(), [a.as_ptr(), b.as_ptr()], out.as_mut_ptr().cast(), Binary::<F>(PhantomData))
    }
}

/// `a[i] = F(a[i], b[i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `a` and `b` have different lengths.
#[inline(always)]
pub(crate) unsafe fn binary_in_place<S: SimdVector, F: BinaryFn<S>>(
    a: &mut [S::Scalar],
    b: &[S::Scalar],
) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let ptr = a.as_mut_ptr();
    // SAFETY: the caller guarantees the ISA; both slices hold `a.len()`
    // elements, and each chunk of `a` is read before it is overwritten.
    unsafe { drive(a.len(), [ptr.cast_const(), b.as_ptr()], ptr, Binary::<F>(PhantomData)) }
}

/// [`binary_uninit`] with three inputs: on return all of `out` is
/// initialised.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `a`, `b`, `c` and `out` do not all have the same length.
#[inline(always)]
pub(crate) unsafe fn ternary_uninit<S: SimdVector, F: TernaryFn<S>>(
    a: &[S::Scalar],
    b: &[S::Scalar],
    c: &[S::Scalar],
    out: &mut [MaybeUninit<S::Scalar>],
) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    assert_eq!(a.len(), c.len(), "length mismatch");
    assert_eq!(a.len(), out.len(), "length mismatch");
    let src = [a.as_ptr(), b.as_ptr(), c.as_ptr()];
    // SAFETY: the caller guarantees the ISA; all four slices hold `a.len()`
    // elements, and `MaybeUninit<T>` has the layout of `T`.
    unsafe { drive(a.len(), src, out.as_mut_ptr().cast(), Ternary::<F>(PhantomData)) }
}

/// `a[i] = F(a[i], b[i], c[i])`.
///
/// # Safety
///
/// The CPU must support `S`'s ISA.
///
/// # Panics
///
/// If `a`, `b` and `c` do not all have the same length.
#[inline(always)]
pub(crate) unsafe fn ternary_in_place<S: SimdVector, F: TernaryFn<S>>(
    a: &mut [S::Scalar],
    b: &[S::Scalar],
    c: &[S::Scalar],
) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    assert_eq!(a.len(), c.len(), "length mismatch");
    let ptr = a.as_mut_ptr();
    let src = [ptr.cast_const(), b.as_ptr(), c.as_ptr()];
    // SAFETY: the caller guarantees the ISA; all three slices hold `a.len()`
    // elements, and each chunk of `a` is read before it is overwritten.
    unsafe { drive(a.len(), src, ptr, Ternary::<F>(PhantomData)) }
}

/// An operation on `N` slices, one register from each.
trait Op<S: SimdVector, const N: usize> {
    /// `op` on the `LANES` elements at `i` of every slice.
    ///
    /// # Safety
    ///
    /// The CPU must support `S`'s ISA, and every pointer must be valid for
    /// `i + LANES` reads.
    unsafe fn full(&self, src: [*const S::Scalar; N], i: usize) -> S;

    /// `op` on the `n < LANES` elements at `i`; other lanes are zero.
    ///
    /// # Safety
    ///
    /// The CPU must support `S`'s ISA, and every pointer must be valid for
    /// `i + n` reads.
    unsafe fn partial(&self, src: [*const S::Scalar; N], i: usize, n: usize) -> S;
}

/// A [`UnaryFn`] as an [`Op`].
struct Unary<F>(PhantomData<F>);

impl<S: SimdVector, F: UnaryFn<S>> Op<S, 1> for Unary<F> {
    #[inline(always)]
    unsafe fn full(&self, [x]: [*const S::Scalar; 1], i: usize) -> S {
        // SAFETY: guaranteed by the caller.
        F::call(unsafe { S::load(x.add(i)) })
    }

    #[inline(always)]
    unsafe fn partial(&self, [x]: [*const S::Scalar; 1], i: usize, n: usize) -> S {
        // SAFETY: guaranteed by the caller.
        F::call(unsafe { S::load_partial(x.add(i), n) })
    }
}

/// A [`BinaryFn`] as an [`Op`].
struct Binary<F>(PhantomData<F>);

impl<S: SimdVector, F: BinaryFn<S>> Op<S, 2> for Binary<F> {
    #[inline(always)]
    unsafe fn full(&self, [a, b]: [*const S::Scalar; 2], i: usize) -> S {
        // SAFETY: guaranteed by the caller.
        unsafe { F::call(S::load(a.add(i)), S::load(b.add(i))) }
    }

    #[inline(always)]
    unsafe fn partial(&self, [a, b]: [*const S::Scalar; 2], i: usize, n: usize) -> S {
        // SAFETY: guaranteed by the caller.
        unsafe { F::call(S::load_partial(a.add(i), n), S::load_partial(b.add(i), n)) }
    }
}

/// A [`TernaryFn`] as an [`Op`].
struct Ternary<F>(PhantomData<F>);

impl<S: SimdVector, F: TernaryFn<S>> Op<S, 3> for Ternary<F> {
    #[inline(always)]
    unsafe fn full(&self, [a, b, c]: [*const S::Scalar; 3], i: usize) -> S {
        // SAFETY: guaranteed by the caller.
        unsafe { F::call(S::load(a.add(i)), S::load(b.add(i)), S::load(c.add(i))) }
    }

    #[inline(always)]
    unsafe fn partial(&self, [a, b, c]: [*const S::Scalar; 3], i: usize, n: usize) -> S {
        // SAFETY: guaranteed by the caller.
        unsafe {
            F::call(
                S::load_partial(a.add(i), n),
                S::load_partial(b.add(i), n),
                S::load_partial(c.add(i), n),
            )
        }
    }
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
    op: impl Op<S, N>,
) {
    let full = len - len % S::LANES;
    let mut i = 0;
    while i < full {
        // SAFETY: the caller guarantees the ISA, and `i + LANES <= len`.
        unsafe { op.full(src, i).store(dst.add(i)) };
        i += S::LANES;
    }

    let tail = len - full;
    if tail > 0 {
        // SAFETY: the caller guarantees the ISA; `tail < LANES` elements
        // remain from `full`.
        unsafe { op.partial(src, full, tail).store_partial(dst.add(full), tail) };
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

        /// Test operations, written once over `F32x8`'s operators.
        macro_rules! test_ops {
            ($($name:ident: |$($arg:ident),*| $body:expr;)*) => {
                $(
                    struct $name;
                    test_ops!(@impl $name: |$($arg),*| $body);
                )*
            };
            (@impl $name:ident: |$x:ident| $body:expr) => {
                impl UnaryFn<F32x8> for $name {
                    #[inline(always)]
                    fn call($x: F32x8) -> F32x8 {
                        $body
                    }
                }
            };
            (@impl $name:ident: |$a:ident, $b:ident| $body:expr) => {
                impl BinaryFn<F32x8> for $name {
                    #[inline(always)]
                    fn call($a: F32x8, $b: F32x8) -> F32x8 {
                        $body
                    }
                }
            };
        }

        test_ops! {
            // SAFETY: an F32x8 exists, so AVX2 + FMA are available.
            Affine: |v| v * unsafe { F32x8::splat(1.5) } + unsafe { F32x8::splat(0.25) };
            Identity: |v| v;
            NegDiv: |v| -v / v;
            MulSub: |a, b| a * b - a;
            First: |a, _b| a;
            DivOp: |a, b| a / b;
            SubOp: |a, b| a - b;
        }

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
            for len in lengths() {
                let x = input(len, 1.0);
                let mut out = output(len);
                // SAFETY: AVX2 + FMA checked above.
                unsafe { unary::<F32x8, Affine>(&x, &mut out[..len]) };
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
                unsafe { binary::<F32x8, MulSub>(&a, &b, &mut out[..len]) };
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
                    unary::<F32x8, NegDiv>(&a, &mut want_u);
                    unary_in_place::<F32x8, NegDiv>(&mut got_u[..len]);
                    binary::<F32x8, DivOp>(&a, &b, &mut want_b);
                    binary_in_place::<F32x8, DivOp>(&mut got_b[..len], &b);
                }
                let same = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits());
                assert!(same(&got_u[..len], &want_u), "unary_in_place, len {len}");
                assert!(same(&got_b[..len], &want_b), "binary_in_place, len {len}");
                assert_untouched_past(&got_u, len);
                assert_untouched_past(&got_b, len);
            }
        }

        #[test]
        fn binary_uninit_matches_binary() {
            if !has_avx2_fma() {
                return;
            }
            for len in lengths() {
                let (a, b) = (input(len, 1.0), input(len, -3.0));
                let mut want = std::vec![0.0; len];
                let mut got = Vec::with_capacity(len);
                // SAFETY: AVX2 + FMA checked above.
                unsafe {
                    binary::<F32x8, SubOp>(&a, &b, &mut want);
                    binary_uninit::<F32x8, SubOp>(&a, &b, &mut got.spare_capacity_mut()[..len]);
                }
                // SAFETY: `binary_uninit` initialised the first `len` elements.
                unsafe { got.set_len(len) };
                assert_eq!(got, want, "len {len}");
            }
        }

        struct Fma3;

        impl TernaryFn<F32x8> for Fma3 {
            #[inline(always)]
            fn call(a: F32x8, b: F32x8, c: F32x8) -> F32x8 {
                a * b - c
            }
        }

        #[test]
        fn ternary_matches_scalar() {
            if !has_avx2_fma() {
                return;
            }
            for len in lengths() {
                let (a, b, c) = (input(len, 1.0), input(len, -3.0), input(len, 0.5));
                let want: Vec<f32> = (0..len).map(|i| a[i] * b[i] - c[i]).collect();
                let mut got = Vec::with_capacity(len + 8);
                got.resize(len + 8, SENTINEL);
                let mut in_place = output(len);
                in_place[..len].copy_from_slice(&a);
                // SAFETY: AVX2 + FMA checked above; `MaybeUninit<f32>` has
                // the layout of `f32`.
                unsafe {
                    let out = core::slice::from_raw_parts_mut(got.as_mut_ptr().cast(), len);
                    ternary_uninit::<F32x8, Fma3>(&a, &b, &c, out);
                    ternary_in_place::<F32x8, Fma3>(&mut in_place[..len], &b, &c);
                }
                assert_eq!(got[..len], want, "ternary_uninit, len {len}");
                assert_eq!(in_place[..len], want, "ternary_in_place, len {len}");
                assert_untouched_past(&got, len);
                assert_untouched_past(&in_place, len);
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
                unsafe { unary::<F32x8, Identity>(&x, &mut out) }
            }));
            assert!(panics(&mut || {
                // SAFETY: see above.
                unsafe { binary::<F32x8, First>(&x, &x[..8], &mut out[..8]) }
            }));
            assert!(panics(&mut || {
                // SAFETY: see above.
                unsafe { binary_in_place::<F32x8, First>(&mut a, &x[..8]) }
            }));
        }
    }
}
