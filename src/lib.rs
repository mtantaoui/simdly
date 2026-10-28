//! # simdly
//!
//! SIMD implementations of mathematical functions for `f32` and `f64`, with a
//! documented worst-case ULP bound for every function.
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
//! - `alloc` — `Vec`-returning slice APIs (planned).
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

cfg_simd! {
    mod simd;
}

#[cfg(test)]
mod testing;
