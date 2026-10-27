//! Element-wise arithmetic (`+ - * / %`, scalar and in-place variants) and
//! horizontal reductions (`sum`, `product`, `min`, `max`).
//!
//! The public `SliceExt` / `VecExt` traits will be defined here. Arithmetic is
//! generic over the register traits in [`crate::simd`]; only reductions need
//! per-ISA code.
