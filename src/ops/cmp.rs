//! Element-wise comparisons on `f32` and `f64` slices, producing `bool`s.
//!
//! Each comparison is a zero-sized type implementing [`CmpOp`], written once
//! for every lane type (`f32`, `f64` and the SIMD registers). Comparisons with
//! NaN are false, except `!=`, which is true, as with Rust's operators.
//!
//! AVX2 compares 32 elements per step (4 `f32` or 8 `f64` registers) and
//! narrows the lane masks to bytes with saturating packs, so all 32 `bool`s
//! are written with one store. AVX-512 compares into a bit mask, expands it to
//! one 0/1 lane per element and narrows that to bytes in a masked store, which
//! also handles the tail.

use core::mem::MaybeUninit;

/// A lane type that can be compared: a float, or a SIMD register of floats.
pub(crate) trait Compare: Copy {
    /// `bool` for a float; for a register, all-ones or all-zero lanes.
    type Mask: Copy;

    fn lt(self, rhs: Self) -> Self::Mask;
    fn le(self, rhs: Self) -> Self::Mask;
    fn gt(self, rhs: Self) -> Self::Mask;
    fn ge(self, rhs: Self) -> Self::Mask;
    fn eq(self, rhs: Self) -> Self::Mask;
    fn ne(self, rhs: Self) -> Self::Mask;
}

/// An element-wise comparison.
pub(crate) trait CmpOp {
    /// Applies the comparison lane-wise.
    fn apply<V: Compare>(a: V, b: V) -> V::Mask;
}

macro_rules! compare_float {
    ($($T:ty),*) => {
        $(
            impl Compare for $T {
                type Mask = bool;

                #[inline(always)]
                fn lt(self, rhs: Self) -> bool {
                    self < rhs
                }

                #[inline(always)]
                fn le(self, rhs: Self) -> bool {
                    self <= rhs
                }

                #[inline(always)]
                fn gt(self, rhs: Self) -> bool {
                    self > rhs
                }

                #[inline(always)]
                fn ge(self, rhs: Self) -> bool {
                    self >= rhs
                }

                #[inline(always)]
                fn eq(self, rhs: Self) -> bool {
                    self == rhs
                }

                #[inline(always)]
                fn ne(self, rhs: Self) -> bool {
                    self != rhs
                }
            }
        )*
    };
}

compare_float!(f32, f64);

macro_rules! cmp_op {
    ($($name:ident: $method:ident $op:tt),* $(,)?) => {
        $(
            #[doc = concat!("`a ", stringify!($op), " b`.")]
            pub(crate) struct $name;

            impl CmpOp for $name {
                #[inline(always)]
                fn apply<V: Compare>(a: V, b: V) -> V::Mask {
                    a.$method(b)
                }
            }
        )*
    };
}

cmp_op!(LtOp: lt <, LeOp: le <=, GtOp: gt >, GeOp: ge >=, EqOp: eq ==, NeOp: ne !=);

/// An element type with comparison kernels: `f32` or `f64`.
pub(crate) trait CmpElement: Compare<Mask = bool> {
    /// `out[i] = Op(a[i], b[i])`; on return all of `out` is initialised.
    ///
    /// # Panics
    ///
    /// If `a`, `b` and `out` do not all have the same length.
    fn cmp_uninit<Op: CmpOp>(a: &[Self], b: &[Self], out: &mut [MaybeUninit<bool>]);

    /// `out[i] = Op(a[i], b)`; on return all of `out` is initialised.
    ///
    /// # Panics
    ///
    /// If `a` and `out` have different lengths.
    fn cmp_scalar_uninit<Op: CmpOp>(a: &[Self], b: Self, out: &mut [MaybeUninit<bool>]);
}

macro_rules! cmp_element {
    ($($T:ty => $avx512:ident, $avx2:ident);* $(;)?) => {
        $(
            impl CmpElement for $T {
                #[inline]
                fn cmp_uninit<Op: CmpOp>(a: &[Self], b: &[Self], out: &mut [MaybeUninit<bool>]) {
                    assert_eq!(a.len(), b.len(), "length mismatch");
                    assert_eq!(a.len(), out.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_cmp::<x86::$avx512, Op>(a, b, out),
                        avx2 => x86::avx2_cmp::<x86::$avx2, Op>(a, b, out),
                        scalar => scalar_cmp::<Op, Self>(a, b, out),
                    }
                }

                #[inline]
                fn cmp_scalar_uninit<Op: CmpOp>(a: &[Self], b: Self, out: &mut [MaybeUninit<bool>]) {
                    assert_eq!(a.len(), out.len(), "length mismatch");
                    dispatch! {
                        avx512 => x86::avx512_cmp_scalar::<x86::$avx512, Op>(a, b, out),
                        avx2 => x86::avx2_cmp_scalar::<x86::$avx2, Op>(a, b, out),
                        scalar => scalar_cmp_scalar::<Op, Self>(a, b, out),
                    }
                }
            }
        )*
    };
}

cmp_element!(f32 => F32x16, F32x8; f64 => F64x8, F64x4);

fn scalar_cmp<Op: CmpOp, T: Compare<Mask = bool>>(a: &[T], b: &[T], out: &mut [MaybeUninit<bool>]) {
    for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
        o.write(Op::apply(x, y));
    }
}

fn scalar_cmp_scalar<Op: CmpOp, T: Compare<Mask = bool>>(
    a: &[T],
    b: T,
    out: &mut [MaybeUninit<bool>],
) {
    for (o, &x) in out.iter_mut().zip(a) {
        o.write(Op::apply(x, b));
    }
}

cfg_x86! {
    mod x86 {
        use core::arch::x86_64::{
            __m256i, __mmask8, __mmask16, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ,
            _CMP_LT_OQ, _CMP_NEQ_UQ, _mm512_cmp_pd_mask, _mm512_cmp_ps_mask,
            _mm512_mask_cvtepi32_storeu_epi8, _mm512_mask_cvtepi64_storeu_epi8,
            _mm512_maskz_set1_epi32, _mm512_maskz_set1_epi64,
            _mm256_and_si256, _mm256_castpd_si256, _mm256_castps_si256, _mm256_cmp_pd,
            _mm256_cmp_ps, _mm256_packs_epi16, _mm256_packs_epi32,
            _mm256_permutevar8x32_epi32, _mm256_set1_epi8, _mm256_setr_epi8,
            _mm256_setr_epi32, _mm256_shuffle_epi8, _mm256_storeu_si256,
        };
        use core::mem::MaybeUninit;

        use super::{CmpOp, Compare};
        use crate::simd::SimdVector;
        pub(super) use crate::simd::avx2::{f32x8::F32x8, f64x4::F64x4};
        pub(super) use crate::simd::avx512::{f32x16::F32x16, f64x8::F64x8};

        /// `impl Compare` for a register: AVX2 compares into a vector of
        /// all-ones / zero lanes (cast to integers), AVX-512 into a bit mask.
        macro_rules! compare {
            ($V:ty, $cmp:ident, $cast:ident; $($method:ident: $pred:ident),* $(,)?) => {
                impl Compare for $V {
                    type Mask = __m256i;
                    $(
                        #[inline(always)]
                        fn $method(self, rhs: Self) -> __m256i {
                            // SAFETY: a register of this type exists, so its ISA
                            // is available.
                            unsafe { $cast($cmp::<{ $pred }>(self.raw(), rhs.raw())) }
                        }
                    )*
                }
            };
            ($V:ty, $cmp:ident => $Mask:ty; $($method:ident: $pred:ident),* $(,)?) => {
                impl Compare for $V {
                    type Mask = $Mask;
                    $(
                        #[inline(always)]
                        fn $method(self, rhs: Self) -> $Mask {
                            // SAFETY: a register of this type exists, so its ISA
                            // is available.
                            unsafe { $cmp::<{ $pred }>(self.raw(), rhs.raw()) }
                        }
                    )*
                }
            };
        }

        // Ordered (`_OQ`): false if either side is NaN; `!=` is unordered
        // (`_UQ`): true if either side is NaN. Quiet: NaN raises no flag.
        macro_rules! compare_all {
            ($([$($register:tt)*])*) => {
                $(
                    compare!(
                        $($register)*;
                        lt: _CMP_LT_OQ,
                        le: _CMP_LE_OQ,
                        gt: _CMP_GT_OQ,
                        ge: _CMP_GE_OQ,
                        eq: _CMP_EQ_OQ,
                        ne: _CMP_NEQ_UQ,
                    );
                )*
            };
        }

        compare_all!(
            [F32x8, _mm256_cmp_ps, _mm256_castps_si256]
            [F64x4, _mm256_cmp_pd, _mm256_castpd_si256]
            [F32x16, _mm512_cmp_ps_mask => __mmask16]
            [F64x8, _mm512_cmp_pd_mask => __mmask8]
        );

        /// Elements compared per step: one 32-byte store of `bool`s.
        const STEP: usize = 32;

        /// An AVX2 register whose comparison masks narrow to `bool` bytes.
        pub(super) trait Pack: SimdVector + Compare<Mask = __m256i> {
            /// Narrows the masks of `STEP / LANES` registers, `mask(0)`,
            /// `mask(1)`, ..., to `STEP` `bool` bytes, in order.
            ///
            /// # Safety
            ///
            /// The CPU must support AVX2.
            unsafe fn pack(mask: impl FnMut(usize) -> __m256i) -> __m256i;
        }

        impl Pack for F32x8 {
            #[inline(always)]
            unsafe fn pack(mask: impl FnMut(usize) -> __m256i) -> __m256i {
                // SAFETY: the caller guarantees AVX2.
                unsafe { _mm256_and_si256(narrow(core::array::from_fn(mask)), _mm256_set1_epi8(1)) }
            }
        }

        impl Pack for F64x4 {
            #[inline(always)]
            unsafe fn pack(mut mask: impl FnMut(usize) -> __m256i) -> __m256i {
                // SAFETY: the caller guarantees AVX2.
                unsafe {
                    // A 64-bit lane mask is two equal 32-bit halves, so one
                    // pack narrows it to a 32-bit mask. Per 128-bit half, the
                    // result `j` holds elements 8j + [0 1 4 5] (low) and
                    // 8j + [2 3 6 7] (high).
                    let m = core::array::from_fn(|j| _mm256_packs_epi32(mask(2 * j), mask(2 * j + 1)));
                    // `narrow` keeps each register's lanes together, so byte
                    // group `j` is 8j + [0 1 4 5 2 3 6 7]: swap the middle pairs.
                    let order = _mm256_setr_epi8(
                        0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
                        0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
                    );
                    let p = _mm256_shuffle_epi8(narrow(m), order);
                    _mm256_and_si256(p, _mm256_set1_epi8(1))
                }
            }
        }

        /// Narrows four registers of 32-bit lane masks to 32 bytes (0xFF or
        /// 0), in order.
        ///
        /// # Safety
        ///
        /// The CPU must support AVX2.
        #[inline(always)]
        unsafe fn narrow(m: [__m256i; 4]) -> __m256i {
            // SAFETY: the caller guarantees AVX2.
            unsafe {
                // Signed saturation keeps -1 as -1: i32 -> i16 -> i8.
                let p = _mm256_packs_epi16(
                    _mm256_packs_epi32(m[0], m[1]),
                    _mm256_packs_epi32(m[2], m[3]),
                );
                // The packs work per 128-bit half, so 4-byte groups come out as
                // m0[0..4] m1[0..4] m2[0..4] m3[0..4] | m0[4..8] m1[4..8] ...
                _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7))
            }
        }

        /// The right-hand side: a slice, or one value broadcast to every lane.
        trait Rhs<V: SimdVector>: Copy {
            /// # Safety
            ///
            /// The CPU must support AVX2, and `i + LANES` elements must be
            /// readable.
            unsafe fn get(self, i: usize) -> V;

            /// # Safety
            ///
            /// The CPU must support AVX2, `n < LANES`, and `i + n` elements
            /// must be readable.
            unsafe fn get_partial(self, i: usize, n: usize) -> V;
        }

        impl<V: SimdVector> Rhs<V> for &[V::Scalar] {
            #[inline(always)]
            unsafe fn get(self, i: usize) -> V {
                // SAFETY: guaranteed by the caller.
                unsafe { V::load(self.as_ptr().add(i)) }
            }

            #[inline(always)]
            unsafe fn get_partial(self, i: usize, n: usize) -> V {
                // SAFETY: guaranteed by the caller.
                unsafe { V::load_partial(self.as_ptr().add(i), n) }
            }
        }

        /// One value in every lane.
        #[derive(Clone, Copy)]
        struct Splat<V>(V);

        impl<V: SimdVector> Rhs<V> for Splat<V> {
            #[inline(always)]
            unsafe fn get(self, _: usize) -> V {
                self.0
            }

            #[inline(always)]
            unsafe fn get_partial(self, _: usize, _: usize) -> V {
                self.0
            }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_cmp<V: Pack, Op: CmpOp>(
            a: &[V::Scalar],
            b: &[V::Scalar],
            out: &mut [MaybeUninit<bool>],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, `V` is an AVX2
            // register type, and the caller checked that all three slices have
            // the same length.
            unsafe { cmp::<V, Op, _>(a, b, out) }
        }

        #[target_feature(enable = "avx2,fma")]
        pub(super) fn avx2_cmp_scalar<V: Pack, Op: CmpOp>(
            a: &[V::Scalar],
            b: V::Scalar,
            out: &mut [MaybeUninit<bool>],
        ) {
            // SAFETY: AVX2 + FMA are enabled in this function, `V` is an AVX2
            // register type, and the caller checked that `a` and `out` have
            // the same length.
            unsafe { cmp::<V, Op, _>(a, Splat(V::splat(b)), out) }
        }

        /// `out[i] = Op(a[i], b[i])`, `STEP` elements per step.
        ///
        /// # Safety
        ///
        /// The CPU must support AVX2 + FMA, `out` must have `a.len()` elements
        /// and `b` must be readable for `a.len()` elements.
        #[inline(always)]
        unsafe fn cmp<V: Pack, Op: CmpOp, R: Rhs<V>>(
            a: &[V::Scalar],
            b: R,
            out: &mut [MaybeUninit<bool>],
        ) {
            let (len, x, dst) = (a.len(), a.as_ptr(), out.as_mut_ptr().cast::<u8>());
            let full = len - len % STEP;

            let mut i = 0;
            while i < full {
                // SAFETY: the caller guarantees AVX2 + FMA and the lengths, and
                // `i + STEP <= len`.
                unsafe {
                    let bools = V::pack(|k| {
                        let j = i + k * V::LANES;
                        Op::apply(V::load(x.add(j)), b.get(j))
                    });
                    _mm256_storeu_si256(dst.add(i).cast(), bools);
                }
                i += STEP;
            }

            let tail = len - full;
            if tail > 0 {
                let mut buf = [0_u8; STEP];
                // SAFETY: the caller guarantees AVX2 + FMA and the lengths. Each
                // register reads only the lanes left in `a` and `b`; registers
                // past the end are zero, and their results are not copied.
                // `pack` produces only 0 and 1, which are valid `bool`s.
                unsafe {
                    let bools = V::pack(|k| {
                        let j = full + k * V::LANES;
                        let (va, vb) = match tail.saturating_sub(k * V::LANES) {
                            0 => (V::zero(), V::zero()),
                            n if n >= V::LANES => (V::load(x.add(j)), b.get(j)),
                            n => (V::load_partial(x.add(j), n), b.get_partial(j, n)),
                        };
                        Op::apply(va, vb)
                    });
                    _mm256_storeu_si256(buf.as_mut_ptr().cast(), bools);
                    core::ptr::copy_nonoverlapping(buf.as_ptr(), dst.add(full), tail);
                }
            }
        }

        /// An AVX-512 register whose comparison bit mask can be written as
        /// `bool` bytes.
        pub(super) trait Bools: SimdVector + Compare<Mask = Self::Bits> {
            /// `__mmask16` or `__mmask8`: one bit per lane.
            type Bits: Copy;

            /// Writes lane `i`'s bit of `mask` as a `bool` to `dst[i]`, for the
            /// first `n <= LANES` lanes; later bytes are not touched.
            ///
            /// # Safety
            ///
            /// The CPU must support AVX-512F, and `dst` must be valid for `n`
            /// writes.
            unsafe fn store_bools(mask: Self::Bits, dst: *mut u8, n: usize);
        }

        macro_rules! bools {
            ($($V:ty: $Bits:ty, $set1:ident, $store:ident);*) => {
                $(
                    impl Bools for $V {
                        type Bits = $Bits;

                        #[inline(always)]
                        unsafe fn store_bools(mask: $Bits, dst: *mut u8, n: usize) {
                            debug_assert!(n <= Self::LANES);
                            // Lanes to write: all of them, or the first `n`.
                            let first: $Bits = if n == Self::LANES { !0 } else { (1 << n) - 1 };
                            // SAFETY: the caller guarantees AVX-512F, and that
                            // `dst` is valid for the `n` bytes written; the
                            // masked store writes no others. Lanes are 0 or 1,
                            // valid `bool`s.
                            unsafe { $store(dst.cast(), first, $set1(mask, 1)) }
                        }
                    }
                )*
            };
        }

        bools!(
            F32x16: __mmask16, _mm512_maskz_set1_epi32, _mm512_mask_cvtepi32_storeu_epi8;
            F64x8: __mmask8, _mm512_maskz_set1_epi64, _mm512_mask_cvtepi64_storeu_epi8
        );

        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_cmp<V: Bools, Op: CmpOp>(
            a: &[V::Scalar],
            b: &[V::Scalar],
            out: &mut [MaybeUninit<bool>],
        ) {
            // SAFETY: AVX-512F is enabled in this function, `V` is an AVX-512
            // register type, and the caller checked that all three slices have
            // the same length.
            unsafe { cmp512::<V, Op, _>(a, b, out) }
        }

        #[target_feature(enable = "avx512f")]
        pub(super) fn avx512_cmp_scalar<V: Bools, Op: CmpOp>(
            a: &[V::Scalar],
            b: V::Scalar,
            out: &mut [MaybeUninit<bool>],
        ) {
            // SAFETY: AVX-512F is enabled in this function, `V` is an AVX-512
            // register type, and the caller checked that `a` and `out` have
            // the same length.
            unsafe { cmp512::<V, Op, _>(a, Splat(V::splat(b)), out) }
        }

        /// `out[i] = Op(a[i], b[i])`, one register per step.
        ///
        /// # Safety
        ///
        /// The CPU must support AVX-512F, `out` must have `a.len()` elements
        /// and `b` must be readable for `a.len()` elements.
        #[inline(always)]
        unsafe fn cmp512<V: Bools, Op: CmpOp, R: Rhs<V>>(
            a: &[V::Scalar],
            b: R,
            out: &mut [MaybeUninit<bool>],
        ) {
            let (len, x, dst) = (a.len(), a.as_ptr(), out.as_mut_ptr().cast::<u8>());
            let full = len - len % V::LANES;

            let mut i = 0;
            while i < full {
                // SAFETY: the caller guarantees AVX-512F and the lengths, and
                // `i + LANES <= len`.
                unsafe {
                    let mask = Op::apply(V::load(x.add(i)), b.get(i));
                    V::store_bools(mask, dst.add(i), V::LANES);
                }
                i += V::LANES;
            }

            let tail = len - full;
            if tail > 0 {
                // SAFETY: the caller guarantees AVX-512F and the lengths; the
                // loads read and the store writes only the `tail` lanes left.
                unsafe {
                    let mask = Op::apply(V::load_partial(x.add(full), tail), b.get_partial(full, tail));
                    V::store_bools(mask, dst.add(full), tail);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestFloat;
    use std::vec::Vec;

    /// Covers empty, tail-only, one full 32-element step, and steps + tail.
    const MAX_LEN: usize = 100;

    /// Mixed values where `a[i] == b[i]` regularly, plus signed zeros,
    /// infinities, NaN and subnormals on either side.
    fn inputs<T: TestFloat>(len: usize) -> (Vec<T>, Vec<T>) {
        let a = (0..len).map(|i| match i % 9 {
            2 => T::SPECIAL[i % 8],
            _ => T::from_f64((i % 13) as f64 - 6.0),
        });
        let b = (0..len).map(|i| match i % 7 {
            3 => T::SPECIAL[(i / 7) % 8],
            _ => T::from_f64((i % 11) as f64 - 5.0),
        });
        (a.collect(), b.collect())
    }

    /// Runs `f` on an output of `len` elements inside a buffer of sentinels,
    /// checks every byte written is a valid `bool` and none past `len` was
    /// touched, then returns the result.
    fn run(len: usize, f: impl FnOnce(&mut [MaybeUninit<bool>])) -> Vec<bool> {
        const SENTINEL: u8 = 0xAA;
        let mut buf = std::vec![SENTINEL; len + 8];
        // SAFETY: `MaybeUninit<bool>` has the layout of `u8`, and `buf` holds
        // at least `len` bytes.
        let out = unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), len) };
        f(out);
        assert!(buf[..len].iter().all(|&b| b <= 1), "invalid bool byte");
        assert!(
            buf[len..].iter().all(|&b| b == SENTINEL),
            "wrote past {len}"
        );
        buf[..len].iter().map(|&b| b == 1).collect()
    }

    /// Runs the SIMD kernels directly, on every backend this CPU supports,
    /// whatever dispatch would pick: `(backend, result)` pairs.
    trait Simd: CmpElement {
        fn simd_cmp<Op: CmpOp>(a: &[Self], b: &[Self]) -> Vec<(&'static str, Vec<bool>)>;
        fn simd_cmp_scalar<Op: CmpOp>(a: &[Self], b: Self) -> Vec<(&'static str, Vec<bool>)>;
    }

    macro_rules! simd {
        ($($T:ty => $avx512:ident, $avx2:ident);*) => {
            $(
                impl Simd for $T {
                    fn simd_cmp<Op: CmpOp>(a: &[Self], b: &[Self]) -> Vec<(&'static str, Vec<bool>)> {
                        #[allow(unused_mut, reason = "empty on some targets")]
                        let mut results = Vec::new();
                        cfg_x86!(if {
                            if crate::detect::has_avx2_fma() {
                                // SAFETY: AVX2 + FMA checked just above.
                                let got = run(a.len(), |out| unsafe { x86::avx2_cmp::<x86::$avx2, Op>(a, b, out) });
                                results.push(("avx2", got));
                            }
                            if crate::detect::has_avx512f() {
                                // SAFETY: AVX-512F checked just above.
                                let got = run(a.len(), |out| unsafe { x86::avx512_cmp::<x86::$avx512, Op>(a, b, out) });
                                results.push(("avx512", got));
                            }
                        } else {
                            let _ = (a, b);
                        });
                        results
                    }

                    fn simd_cmp_scalar<Op: CmpOp>(a: &[Self], b: Self) -> Vec<(&'static str, Vec<bool>)> {
                        #[allow(unused_mut, reason = "empty on some targets")]
                        let mut results = Vec::new();
                        cfg_x86!(if {
                            if crate::detect::has_avx2_fma() {
                                // SAFETY: AVX2 + FMA checked just above.
                                let got = run(a.len(), |out| unsafe { x86::avx2_cmp_scalar::<x86::$avx2, Op>(a, b, out) });
                                results.push(("avx2", got));
                            }
                            if crate::detect::has_avx512f() {
                                // SAFETY: AVX-512F checked just above.
                                let got = run(a.len(), |out| unsafe { x86::avx512_cmp_scalar::<x86::$avx512, Op>(a, b, out) });
                                results.push(("avx512", got));
                            }
                        } else {
                            let _ = (a, b);
                        });
                        results
                    }
                }
            )*
        };
    }

    simd!(f32 => F32x16, F32x8; f64 => F64x8, F64x4);

    fn check<Op: CmpOp, T: TestFloat + Simd>(name: &str) {
        let name = std::format!("{name} {}", T::NAME);
        for len in 0..=MAX_LEN {
            let (a, b) = inputs::<T>(len);
            let want: Vec<bool> = a.iter().zip(&b).map(|(&x, &y)| Op::apply(x, y)).collect();

            let got = run(len, |out| T::cmp_uninit::<Op>(&a, &b, out));
            assert_eq!(got, want, "{name}, len {len}");
            let got = run(len, |out| scalar_cmp::<Op, T>(&a, &b, out));
            assert_eq!(got, want, "{name} scalar, len {len}");
            for (backend, got) in T::simd_cmp::<Op>(&a, &b) {
                assert_eq!(got, want, "{name} {backend}, len {len}");
            }

            for &s in &T::SPECIAL {
                let want: Vec<bool> = a.iter().map(|&x| Op::apply(x, s)).collect();
                let got = run(len, |out| T::cmp_scalar_uninit::<Op>(&a, s, out));
                assert_eq!(got, want, "{name} vs {s:?}, len {len}");
                let got = run(len, |out| scalar_cmp_scalar::<Op, T>(&a, s, out));
                assert_eq!(got, want, "{name} scalar vs {s:?}, len {len}");
                for (backend, got) in T::simd_cmp_scalar::<Op>(&a, s) {
                    assert_eq!(got, want, "{name} {backend} vs {s:?}, len {len}");
                }
            }
        }
    }

    fn check_all<Op: CmpOp>(name: &str) {
        check::<Op, f32>(name);
        check::<Op, f64>(name);
    }

    #[test]
    fn lt_matches_scalar() {
        check_all::<LtOp>("lt");
    }

    #[test]
    fn le_matches_scalar() {
        check_all::<LeOp>("le");
    }

    #[test]
    fn gt_matches_scalar() {
        check_all::<GtOp>("gt");
    }

    #[test]
    fn ge_matches_scalar() {
        check_all::<GeOp>("ge");
    }

    #[test]
    fn eq_matches_scalar() {
        check_all::<EqOp>("eq");
    }

    #[test]
    fn ne_matches_scalar() {
        check_all::<NeOp>("ne");
    }

    /// NaN and signed-zero semantics, spelled out.
    #[test]
    fn nan_and_signed_zero() {
        let nan = f32::NAN;
        assert!(!LtOp::apply(nan, 1.0) && !GeOp::apply(nan, 1.0) && !EqOp::apply(nan, nan));
        assert!(NeOp::apply(nan, nan));
        assert!(EqOp::apply(0.0_f32, -0.0) && !LtOp::apply(-0.0_f32, 0.0));
        let nan = f64::NAN;
        assert!(!LtOp::apply(nan, 1.0) && !GeOp::apply(nan, 1.0) && !EqOp::apply(nan, nan));
        assert!(NeOp::apply(nan, nan));
        assert!(EqOp::apply(0.0_f64, -0.0) && !LtOp::apply(-0.0_f64, 0.0));
    }
}
