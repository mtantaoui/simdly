//! Register layer: one SIMD vector type per (ISA, element type) pair.
//!
//! This module will hold the register trait every type implements. Each ISA
//! submodule provides the concrete types; math kernels and the slice drivers
//! are built on top.

cfg_x86! {
    pub(crate) mod avx2;
    pub(crate) mod avx512;
}

cfg_neon! {
    pub(crate) mod neon;
}
