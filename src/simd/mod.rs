//! Register layer: one SIMD vector type per (ISA, element type) pair.
//!
//! This module will hold the internal traits every register type implements
//! (`Load`, `Store`, …). Each ISA submodule provides the concrete types and
//! their implementations; math kernels and the slice drivers are built on top.

cfg_avx2! {
    pub(crate) mod avx2;
}

cfg_avx512! {
    pub(crate) mod avx512;
}

cfg_neon! {
    pub(crate) mod neon;
}
