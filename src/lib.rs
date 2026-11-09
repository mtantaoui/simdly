//! # simdly
//!
//! SIMD implementations of mathematical functions for `f32` and `f64`, with a
//! documented worst-case ULP bound for every function.
//!
//! ## Example
//!
//! ```
//! use simdly::SliceExt;
//!
//! let mut a = [1.0_f32, 2.0, 3.0];
//! a.add_assign(&[4.0, 5.0, 6.0]);
//! assert_eq!(a, [5.0, 7.0, 9.0]);
//! ```
//!
//! With the `alloc` feature (on by default), `add` and friends return a new
//! `Vec` instead; `add_scalar`, `sub_scalar`, … take one value as the right
//! operand of every element. Math functions are in [`SliceMath`]: `abs`, `acos`,
//! `asin`, `atan`, `atan2`, `cbrt`, `ceil`, `cos`, `exp`, `floor`, `hypot`, `hypot3`, `hypot4`, `ln`, `sin`, `sqrt`, `tan`, …, each with an `_in_place`
//! variant.
//!
//! ## Backends
//!
//! | Architecture | ISA             | `f32` lanes | `f64` lanes |
//! |--------------|-----------------|-------------|-------------|
//! | `x86_64`     | AVX2 + FMA      | 8           | 4           |
//! | `x86_64`     | AVX-512         | 16          | 8           |
//! | `aarch64`    | NEON            | 4           | 2           |
//! | any          | scalar fallback | 1           | 1           |
//!
//! On `x86_64` the fastest backend the CPU supports is detected at runtime, so
//! no flags are needed. On `aarch64` NEON is always used. [`backend()`]
//! reports the one in use. Without `std`, only backends enabled at compile
//! time (`-C target-feature=+avx2,+fma`) are used.
//!
//! ## Cargo features
//!
//! - `std` *(default)* — runtime CPU detection; implies `alloc`.
//! - `alloc` — `Vec`-returning slice APIs ([`SliceExt`]'s `add`, …).
//! - `rayon` — [`ParSliceMath`] and [`ParSliceExt`]: the math functions,
//!   arithmetic and comparisons on several threads (`par_sin`, `par_add`,
//!   …) for long slices; implies `std`.
//!
//! Without default features the crate is `no_std`.

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(any(feature = "std", test))]
extern crate std;

// Must come first: the backend-selection macros are textually scoped.
#[macro_use]
mod macros;

mod detect;
mod math;
mod ops;

pub use detect::{Backend, backend};
pub use math::SliceMath;
#[cfg(feature = "rayon")]
pub use math::par::ParSliceMath;
pub use ops::SliceExt;
#[cfg(feature = "rayon")]
pub use ops::par::ParSliceExt;

/// Length above which [`ParSliceMath`] splits a slice across threads (1 MiB
/// of `f32`). Up to it, the `par_` functions run on the calling thread.
#[cfg(feature = "rayon")]
pub const PARALLEL_SIMD_THRESHOLD: usize = 262_144;

/// Length above which [`ParSliceExt`] splits a slice: twice
/// [`PARALLEL_SIMD_THRESHOLD`], as arithmetic and comparisons do less work
/// per element than the math functions.
#[cfg(feature = "rayon")]
pub(crate) const PARALLEL_EXT_THRESHOLD: usize = 2 * PARALLEL_SIMD_THRESHOLD;

/// Elements per rayon task (64 KiB of `f32`): a multiple of every register
/// width, so only the last chunk has a masked tail.
#[cfg(feature = "rayon")]
pub(crate) const PARALLEL_CHUNK_SIZE: usize = 16_384;

cfg_simd! {
    mod simd;
}

#[cfg(test)]
mod testing;
