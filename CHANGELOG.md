# Changelog

All notable changes to `simdly` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Crate scaffolding: backend selection macros, module layout for the
  register layer (`simd`), math functions (`math`) and arithmetic (`ops`).
- `SimdVector` register trait, the AVX2 `F32x8` and `F64x4`, the AVX-512
  `F32x16` and `F64x8` and the NEON `F32x4` and `F64x2` register types.
- Runtime backend detection on `x86_64` with `std`: no `RUSTFLAGS` needed.
- `dispatch!`: calls the fastest available backend.
- Slice drivers (`unary`, `binary` and their in-place variants): full
  registers plus one masked tail, generic over `SimdVector`.
- `simdly::backend()` and `Backend`: report the backend in use.
- `SliceExt` for `[f32]` and `[f64]`: element-wise `add`, `sub`, `mul`, `div` (returning
  a `Vec`, with `alloc`) and their in-place `_assign` variants; AVX-512,
  AVX2, NEON and scalar backends.
- `SliceExt` comparisons for `[f32]` and `[f64]`, like NumPy's: `less`, `less_equal`,
  `greater`, `greater_equal`, `equal`, `not_equal` and their `_scalar`
  variants, returning `Vec<bool>` (`alloc`); AVX-512, AVX2, NEON and scalar
  backends.
- `SliceMath` for `[f32]` and `[f64]`, starting with `abs` (returning a
  `Vec`, with `alloc`) and `abs_in_place`; AVX-512, AVX2, NEON and scalar
  backends.
- `SliceMath::acos` and `acos_in_place`, ≤ 1 ULP (musl's algorithm), on
  every backend: written once over a `Float` lane trait.
- `SliceMath::asin` and `asin_in_place`, ≤ 1 ULP, on every backend; one
  formula for `|x| ≥ 0.5`, more accurate than musl's `f64` one.
- `SliceMath::atan` and `atan_in_place`, ≤ 1 ULP, on every backend; musl's
  five reductions as one, `(p·a − q)/(p + q·a)`, with one division.
- `SliceMath::atan2` and `atan2_in_place`, ≤ 1 ULP on every backend, with
  C99's signed zeros and infinities: no rounded `y/x` (exact remainder) and
  `π − atan` folded into the constants. `Float::residual` (exact `c − a·b`)
  and `Float::is_sign_negative`.
- `SliceMath::cbrt` and `cbrt_in_place`, ≤ 1 ULP (0.667) on every backend:
  exact reduction with `Float::frexp` / `Float::ldexp`, a polynomial guess
  and musl's final step.
- `SliceMath::exp` and `exp_in_place`, ≤ 1 ULP (0.91 `f32`, 0.84 `f64`) on
  every backend, subnormal results included: fdlibm's algorithm with `f32`
  kept in `f32`, and `2^k` applied in two normal factors.
- `SliceMath::hypot(&y)` and `hypot_in_place`, ≤ 1 ULP on every backend
  without overflow or underflow in between: exact squares (as musl's `hypot`)
  and one Newton step on the square root.
- `SliceMath::ln` and `ln_in_place`, ≤ 1 ULP (0.835 `f32`, 0.823 `f64`) on
  every backend, subnormal inputs included: fdlibm's algorithm, `f32` kept
  in `f32`.
- `SliceMath::cos` and `cos_in_place`, ≤ 1 ULP (0.501 `f32`, 0.730 `f64`)
  on every backend for every finite input: `sin`'s code with the quadrant
  shifted by one.
- `SliceMath::tan` and `tan_in_place`, ≤ 1 ULP (0.797 `f32`, 0.802 `f64`)
  on every backend for every finite input: musl's `k_tanf` / `k_tan`, one
  division per lane, only in registers that need it.
- `sin`, `cos`, `tan`: large arguments reduced in vector arithmetic,
  `f32` up to `f32::MAX` (`M·A mod 4`) and `f64` up to `2^33`
  (double-double); a faster scalar Payne–Hanek above. Faster than `std`
  in every argument range.
- `SliceMath::sin` and `sin_in_place`, ≤ 1 ULP (0.501 `f32`, 0.774 `f64`)
  on every backend for every finite input: musl's kernels and reduction,
  `f32` computed in `f64` (`Float::via_f64`), huge arguments reduced exactly
  (Payne–Hanek) only in registers that hold one (`Float::any`,
  `Float::map_lanes`).
- `SliceMath::floor` and `floor_in_place`, exact on every backend (one
  rounding instruction; float arithmetic on scalar); `Float` now requires
  `Floor`.
- `SliceMath::sqrt` and `sqrt_in_place`, correctly rounded on every backend;
  without `std`, the scalar fallback computes it exactly with integers.
