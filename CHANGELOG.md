# Changelog

All notable changes to `simdly` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Crate scaffolding: backend selection macros, module layout for the
  register layer (`simd`), math functions (`math`) and arithmetic (`ops`).
- `SimdVector` register trait and the AVX2 `F32x8` and `F64x4` register types.
- Runtime backend detection on `x86_64` with `std`: no `RUSTFLAGS` needed.
- `dispatch!`: calls the fastest available backend.
- Slice drivers (`unary`, `binary` and their in-place variants): full
  registers plus one masked tail, generic over `SimdVector`.
- `simdly::backend()` and `Backend`: report the backend in use.
- `SliceExt` for `[f32]` and `[f64]`: element-wise `add`, `sub`, `mul`, `div` (returning
  a `Vec`, with `alloc`) and their in-place `_assign` variants; AVX2 and
  scalar backends.
- `SliceExt` comparisons for `[f32]` and `[f64]`, like NumPy's: `less`, `less_equal`,
  `greater`, `greater_equal`, `equal`, `not_equal` and their `_scalar`
  variants, returning `Vec<bool>` (`alloc`); AVX2 and scalar backends.
