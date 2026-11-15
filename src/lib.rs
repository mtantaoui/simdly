//! # simdly
//!
//! Element-wise maths on `f32` and `f64` slices with SIMD instructions, and a
//! documented worst-case error, in ULPs, for every function.
//!
//! ## Example
//!
// The example needs `alloc` (`Vec` results); without it, it is not run.
#![cfg_attr(feature = "alloc", doc = "```")]
#![cfg_attr(not(feature = "alloc"), doc = "```ignore")]
//! use simdly::{SliceExt, SliceMath};
//!
//! let x = [0.0_f32, 0.5, 1.0, 2.0];
//!
//! let y = x.sin();                   // a new Vec<f32>
//! let z = x.add(&y);                 // element-wise x + sin(x)
//! let small = z.less_scalar(1.0);    // Vec<bool>
//! assert_eq!(small, [true, true, false, false]);
//!
//! let mut w = x;
//! w.exp_in_place();                  // in place: no allocation, no `alloc`
//! ```
//!
//! Arithmetic and comparisons are in [`SliceExt`], the math functions in
//! [`SliceMath`]. Each function returns a new `Vec` (with the `alloc`
//! feature, on by default) and has an in-place variant (`_assign` for
//! arithmetic, `_in_place` for math).
//!
//! ## Functions
//!
//! | Category                     | Functions                                                                 |
//! |------------------------------|---------------------------------------------------------------------------|
//! | Arithmetic                   | `add` `sub` `mul` `div`, and with one value: `add_scalar` …               |
//! | Comparisons                  | `less` `less_equal` `greater` `greater_equal` `equal` `not_equal`, `_scalar` |
//! | Rounding and sign            | `abs` `ceil` `floor` `trunc` `round` `round_ties_even` `neg` `sign` `clamp` |
//! | Powers and roots             | `sqrt` `cbrt` `recip` `powf` `powf_scalar` `powi`                         |
//! | Norms                        | `hypot` `hypot3` `hypot4`                                                 |
//! | Exponential and logarithms   | `exp` `ln` `log1p`                                                        |
//! | Trigonometric                | `sin` `cos` `tan` `asin` `acos` `atan` `atan2`                            |
//! | Hyperbolic                   | `sinh` `cosh` `tanh` `asinh` `acosh` `atanh`                              |
//! | Remainders                   | `fmod` `remainder`, and with one divisor: `fmod_scalar` `remainder_scalar` |
//! | Special and activation       | `erf` `gelu` `sigmoid` `log_sigmoid` `softplus`                           |
//!
//! Each function's page in [`SliceMath`] gives its error bound and special
//! values.
//!
//! ## Backends
//!
//! | Architecture | ISA             | `f32` lanes | `f64` lanes |
//! |--------------|-----------------|-------------|-------------|
//! | `x86_64`     | SSE4.1          | 4           | 2           |
//! | `x86_64`     | AVX2 + FMA      | 8           | 4           |
//! | `x86_64`     | AVX-512         | 16          | 8           |
//! | `aarch64`    | NEON            | 4           | 2           |
//! | any          | scalar fallback | 1           | 1           |
//!
//! On `x86_64` the fastest backend the CPU supports is detected at runtime, so
//! no flags are needed. On `aarch64` NEON is always used. [`backend()`]
//! reports the one in use. Without `std`, only backends enabled at compile
//! time (`-C target-feature=+avx2,+fma`, or `+sse4.1`) are used.
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
//!
// Without `rayon` the parallel traits are not compiled: their links above
// point to docs.rs, which documents every feature.
#![cfg_attr(
    not(feature = "rayon"),
    doc = "[`ParSliceMath`]: https://docs.rs/simdly/latest/simdly/trait.ParSliceMath.html",
    doc = "[`ParSliceExt`]: https://docs.rs/simdly/latest/simdly/trait.ParSliceExt.html"
)]
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
