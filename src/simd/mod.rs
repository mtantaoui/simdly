//! Register layer: one SIMD vector type per (ISA, element type) pair, all
//! implementing [`SimdVector`].

#![allow(dead_code, reason = "used once the slice drivers land")]

cfg_x86! {
    pub(crate) mod avx2;
    pub(crate) mod avx512;
}

cfg_neon! {
    pub(crate) mod neon;
}

/// A SIMD register holding [`LANES`](Self::LANES) values of type `Scalar`.
///
/// A value proves the CPU supports the type's ISA: constructors are `unsafe`
/// and require it, so methods taking `self` can use the ISA safely.
///
/// Loads and stores are unaligned; the `_partial` variants handle slice tails.
pub(crate) trait SimdVector: Copy {
    /// Element type.
    type Scalar: Copy;

    /// Number of lanes.
    const LANES: usize;

    /// Broadcasts `x` to every lane.
    ///
    /// # Safety
    ///
    /// The CPU must support the type's ISA.
    unsafe fn splat(x: Self::Scalar) -> Self;

    /// All lanes zero.
    ///
    /// # Safety
    ///
    /// The CPU must support the type's ISA.
    unsafe fn zero() -> Self;

    /// Loads `LANES` elements.
    ///
    /// # Safety
    ///
    /// The CPU must support the type's ISA, and `ptr` must be valid for
    /// `LANES` reads.
    unsafe fn load(ptr: *const Self::Scalar) -> Self;

    /// Loads the first `n` elements; the remaining lanes are zero.
    ///
    /// # Safety
    ///
    /// The CPU must support the type's ISA, `n < LANES`, and `ptr` must be
    /// valid for `n` reads.
    unsafe fn load_partial(ptr: *const Self::Scalar, n: usize) -> Self;

    /// Stores all `LANES` elements.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for `LANES` writes.
    unsafe fn store(self, ptr: *mut Self::Scalar);

    /// Stores the first `n` lanes; memory past them is not touched.
    ///
    /// # Safety
    ///
    /// `n < LANES`, and `ptr` must be valid for `n` writes.
    unsafe fn store_partial(self, ptr: *mut Self::Scalar, n: usize);
}
