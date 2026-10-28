//! Element-wise arithmetic (`+ - * / %`, scalar and in-place variants) and
//! horizontal reductions (`sum`, `product`, `min`, `max`).
//!
//! The public `SliceExt` / `VecExt` traits will be defined here. Arithmetic
//! will be generic over the `SimdVector` trait in `crate::simd`; only
//! reductions will need per-ISA code.
