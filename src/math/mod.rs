//! Element-wise mathematical functions.
//!
//! Each function lives in its own folder:
//!
//! ```text
//! math/exp/
//! ├── mod.rs      public docs + algorithm; routes to the active backend
//! ├── consts.rs   musl / fdlibm constants shared by every ISA
//! ├── avx2.rs     AVX2 kernel and its ULP tests
//! ├── avx512.rs
//! ├── neon.rs
//! └── scalar.rs
//! ```
//!
//! The public `SliceMath` / `VecMath` traits will be defined here, with
//! [`driver`] supplying the chunk + tail loop shared by every function.

mod driver;
