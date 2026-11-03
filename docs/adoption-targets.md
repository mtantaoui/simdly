# simdly: crates that could depend on it

The best targets apply `std` functions (`exp`, `ln`, `sin`, …) element by
element over contiguous `f32`/`f64` buffers. simdly can replace that loop
directly, single-threaded or with `rayon`.

## Before approaching them: a fix in simdly

linfa, smartcore and other ndarray-based crates are written once for a
generic float type `F` and never name `f32` or `f64`. simdly's traits are
implemented on `[f32]` and `[f64]` (the slices), not on a type `F`, so
generic code would need a `[F]: SliceMath<F>` bound on every function.
Rust doesn't propagate that bound from a trait definition, so it can't
simply go on their `Float` trait once.

A small trait implemented on `f32` and `f64` themselves (for example
`simdly::Element`, with methods like `sin_slice`) could be added to a
crate's `Float` trait in one place. Add it before approaching linfa.

## Confirmed in their source

### 1. linfa

`linfa-logistic` 0.8.1, [rust-ml/linfa](https://github.com/rust-ml/linfa),
the main Rust machine-learning toolkit.

- Logistic regression's sigmoid, `1 / (1 + (-x).exp())`, is applied with
  `probs.mapv_inplace(logistic)` and `yz.mapv_inplace(logistic)`.
- Its log-loss uses `exp` + `ln` the same way.
- Its softmax does `v.mapv_inplace(|n| (n - max).exp())`, and its
  log-sum-exp folds over `exp`.

**Fit:** good. These are contiguous ndarray buffers with per-element
`exp`/`ln`, inside training loops that run many times.

**Friction:**

- the generic `F` issue above;
- sigmoid and softmax would each need several simdly passes, because
  simdly has no `sigmoid` and no subtract-a-scalar;
- non-contiguous arrays would need to fall back to `mapv`.

Adding `sigmoid`, `log_sigmoid` and `sub_scalar` to simdly would make this
an easy PR.

### 2. Polars

Highest impact, but the hardest to get merged.

- Its trigonometry expressions (`polars-expr/src/dispatch/trigonometry.rs`,
  0.53) call `v.sin()`, `v.tan()` and so on per value through
  `apply_values`, generic over `num_traits::Float`.
- Chunks are Arrow buffers; null masks and the newer Float16 type are
  their concern.

### 3. ferray

A NumPy-style library, and probably the easiest to land.

- `sin`/`cos`/`tan` go through `libm` (about 1–2 ULP), and it already has
  a fast mode for `exp`.
- It's a young project with slice-shaped ufuncs, so it matches simdly's
  API closely.

### 4. burn-flex

A CPU backend for the Burn deep-learning framework.

- `exp` and `sin` are `unary_op(tensor, f32::exp, f64::exp)`, so they're
  plain `std` calls.
- The older burn-ndarray uses `macerator` SIMD for f32 element-wise ops,
  so it's the weaker target.

### 5. rapidgeo-distance

Haversine distances in batches, with rayon-parallel versions and `_into`
variants that write into a pre-allocated buffer.

- Haversine is built from `sin`, `cos`, `asin` and `sqrt`, all in
  simdly, so a batch could be computed slice by slice.
- It's small and maintained by one author, so it's an easy first adopter.

## Plausible, not confirmed

### 6. Candle

Its CPU backend maps unary ops with a closure (`unary_map`), with
MKL/Accelerate as optional features. simdly could be the default when
neither is enabled, but I haven't seen the code that does this.

### 7. scirs2

`scirs2-core`, a port of SciPy to Rust. It has its own `exp_simd`, for
which it claims a 2–4× speed-up on large arrays. It's a candidate if
simdly is faster or more accurate than that; benchmark it before
proposing anything.

### 8. DataFusion / arrow-rs

Their math functions wrap Arrow kernels, but the evidence is from an old
version (DataFusion 4.0). The change would belong in arrow-rs.

### 9. ndarray

`mapv(f64::exp)` is the usual path, but ndarray is slow to accept changes.
A small `ndarray-simdly` extension crate is more realistic than a PR.

## Not a fit

- **FunDSP**: it already processes audio in SIMD blocks of eight `f32`
  (the `wide` crate's `f32x8`), not slices.
- **statrs**: probability distributions evaluated one value at a time;
  there's no slice API to speed up.
- **smartcore** and **peroxide**: their element-wise math code wasn't
  found, so there's nothing concrete to propose yet.
- **geo** (georust): no batch distance API found.
- **sleef**, **simd_vector**, **rten**, **pulp**, **macerator**:
  alternatives or lower-level SIMD tools, not adopters.

## Suggested order

1. Add the scalar-level trait, plus `sigmoid`, `log_sigmoid` and
   `sub_scalar`. Together these remove most of the friction for linfa and
   other ndarray users.
2. Open PRs to **ferray** and **rapidgeo-distance**: easy wins that give
   you real users to cite.
3. Then **linfa** and **burn-flex**, each starting with an issue that has a
   benchmark of their current code.
4. Then **Polars** and **Candle**, once you have the full benchmark tables
   and adoption elsewhere to point to.

## How to approach each project

- **Open an issue before a PR**, with a benchmark of their current code
  and simdly's.
- **Offer simdly as an optional feature**, with a fallback to their
  current code.
- Include the **accuracy tables**, the **NaN / infinity / signed-zero
  behaviour**, and the **parallel speed-up over rayon with `std`**, since
  several of these projects already use rayon.

## Note on search results

One search result described simdly as working only in `f32` and claimed
4–13× speed-ups for `cos`. That's an outdated description that doesn't
match the current crate, which handles `f32` and `f64`. Search engines may
show it to people until they re-index docs.rs.

## Sources

- [linfa-logistic source (0.8.1)](https://docs.rs/crate/linfa-logistic/latest/source/src/lib.rs)
- [linfa on GitHub](https://github.com/rust-ml/linfa)
- [polars-expr trigonometry.rs (0.53.0)](https://docs.rs/crate/polars-expr/0.53.0/source/src/dispatch/trigonometry.rs)
- [ferray on GitHub](https://github.com/amberwhitehead/ferray)
- [ferray-ufunc README](https://docs.rs/crate/ferray-ufunc/0.3.7/source/README.md)
- [burn-flex unary.rs](https://docs.rs/crate/burn-flex/0.0.1/source/src/ops/unary.rs)
- [burn-flex README](https://docs.rs/crate/burn-flex/latest/source/README.md)
- [rapidgeo-distance README](https://docs.rs/crate/rapidgeo-distance/latest/source/README.md)
- [rapidgeo-distance batch.rs](https://docs.rs/crate/rapidgeo-distance/latest/source/src/batch.rs)
- [candle-core custom op tests](https://docs.rs/crate/candle-core/latest/source/tests/custom_op_tests.rs)
- [scirs2-core elementwise functions](https://docs.rs/scirs2-core/0.6.5/src/scirs2_core/ndarray_ext/elementwise/functions.rs.html)
- [DataFusion 4.0 math_expressions.rs](https://docs.rs/crate/datafusion/4.0.0/source/src/physical_plan/math_expressions.rs)
- [fundsp source](https://docs.rs/crate/fundsp/0.23.0/source/src/lib.rs)
- [smartcore](https://docs.rs/crate/smartcore/latest)
- [peroxide](https://docs.rs/crate/peroxide/latest)
- [sleef](https://docs.rs/sleef)
- [simd_vector](https://docs.rs/crate/simd_vector/0.4.1/source/README.md)
