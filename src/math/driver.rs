//! Slice drivers: split a slice into full SIMD registers plus a masked tail
//! and apply a register-level operation to each chunk. Written once, generic
//! over the register type, and reused by every math function and backend.
