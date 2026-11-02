# Parallelism

With the `rayon` feature, `ParSliceMath` runs every `SliceMath` function on
several threads (`par_sin`, `par_hypot`, `par_exp_in_place`, …), and
`ParSliceExt` every `SliceExt` operation (`par_add`, `par_mul_assign`,
`par_less`, `par_equal_scalar`, …).

```toml
[dependencies]
simdly = { version = "0.1", features = ["rayon"] }
```

```rust
use simdly::{ParSliceExt, ParSliceMath};

let x: Vec<f32> = (0..4_000_000).map(|i| i as f32 * 1e-3).collect();
let s = x.par_sin();
let m = x.par_less_scalar(1_000.0);
```

## How it splits a slice

A slice of at most `PARALLEL_SIMD_THRESHOLD` (262 144) elements runs on
the calling thread, exactly as `SliceMath`. A longer one is cut into chunks
of 16 384 elements (64 KiB of `f32`), and rayon runs the single-threaded
function on each, in its global thread pool.

16 384 is a multiple of every register width, so only the last chunk has
a masked tail. Each element is computed by the same instructions as on one
thread: the results are the same **bit for bit** (tested for every
function), and so are the panics on slices of different lengths.

## Why 262 144

Waking rayon's threads costs tens of microseconds, so a slice must be long
enough for the work saved to pay for it. Measured with the threshold
forced to 0 (AVX2, Intel Core Ultra 7 155H, 22 threads), time in µs:

| Elements  | `abs` `f32` 1 thread / `par` | `sqrt` `f64` 1 thread / `par` | `sin` `f32` 1 thread / `par` |
|-----------|------------------------------|-------------------------------|------------------------------|
| 65 536    | 5.1 / 24.0                   | 43.4 / 83.1                   | 93.3 / 111.8                 |
| 262 144   | 24.5 / 29.5                  | 174.0 / 110.1                 | 355.6 / 139.7                |
| 1 048 576 | 139.5 / 66.6                 | 697.7 / 209.0                 | 1452.8 / 365.3               |

At 65 536 elements, threads lose for every function. At 262 144, they
win for the expensive ones (`sin`, `exp`, `hypot`, `f64` `sqrt`) but not
yet for the cheapest `f32` ones, so the split starts above it. From a
million elements on, they win for every function.

## Arithmetic and comparisons: 524 288

`ParSliceExt` splits slices longer than twice the threshold, 524 288
elements. Its operations are a single instruction per register, cheaper
than `abs`, and lose to one thread longer. Same machine, threshold forced
to 0, one thread / `par` speed-up:

| Elements | `add_assign` `f32` | `less` `f32` | `less_scalar` `f32` | `add` `f64` |
|----------|--------------------|--------------|---------------------|-------------|
| 65 536   | 0.21×              | 0.27×        | 0.15×               | 0.36×       |
| 262 144  | 0.76×              | 0.97×        | 0.63×               | 1.66×       |
| 524 288  | 1.56×              | 2.32×        | 1.15×               | 2.71×       |

Against the rayon loop with `std`, `par` is faster at every length
measured.

## When it helps

- **Expensive functions** (`sin`, `cos`, `tan`, `exp`, `ln`, `hypot*`,
  `atan2`, …) scale with the number of cores: the work per element
  dominates.
- **Cheap functions** (`abs`, `ceil`, `floor`, `sqrt`) and all of
  arithmetic and comparisons are limited by memory bandwidth on large
  slices: several cores read and write faster than one, but far from
  proportionally.
- **Inside your own parallel code** (a `par_iter` over rows, say), call the
  `SliceMath` functions instead: the work is already split, and nesting
  only adds scheduling.

With `cargo bench --features rayon`, every benchmark group times the `par_`
functions next to simdly on one thread and the rayon loop a user would
write instead, `x.par_iter().map(|v| v.sin()).collect()`; the benchmark
pages report both speed-ups.
