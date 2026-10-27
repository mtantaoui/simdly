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
//! The backend is selected at compile time from `target-feature` flags. On
//! `x86_64`, build with `RUSTFLAGS="-C target-feature=+avx2,+fma"` (or
//! `-C target-cpu=native`) to get a SIMD backend.
//!
//! ## Cargo features
//!
//! - `std` *(default)* — runtime CPU detection (planned); implies `alloc`.
//! - `alloc` — `Vec`-returning slice APIs.
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

mod math;
mod ops;
mod simd;

#[cfg(test)]
mod testing;
