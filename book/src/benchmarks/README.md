# Benchmarks

Each benchmark compares simdly with the plain scalar loop a user would write
instead, on the same input:

```rust
let simdly = a.acos();                                         // simdly
let scalar: Vec<f32> = a.iter().map(|x| x.acos()).collect();   // scalar
```

The scalar side uses `std`'s functions, and the compiler is free to
auto-vectorise it. The tables report the **speed-up**, scalar time divided by
simdly time (criterion's mean): `2.00×` means simdly takes half the time.

## Lengths

| Elements    | `f32` data | Fits in   |
|-------------|------------|-----------|
| 64          | 256 B      | L1        |
| 4 096       | 16 KiB     | L1        |
| 262 144     | 1 MiB      | L2 or L3  |
| 4 194 304   | 16 MiB     | memory    |

With `--features rayon`, every group also times the `par_` functions
(`par`) and the rayon loop a user would write instead (`rayon`,
`x.par_iter().map(|v| v.sin()).collect()`), at every length. Up to the
thresholds where `par_` starts threads, it runs on one thread, so the
small lengths show what rayon costs there and that `par_` does not pay
it.

In-place math benchmarks run each iteration on a fresh copy of the input
(the copy is not timed), so every implementation sees the same values.
In-place arithmetic reuses its slice: with `b` all ones, `a op= b` keeps
`a` normal, so the timing does not depend on the pass.

At 4 194 304 elements, cheap operations (`add`, `abs`, …) are limited by
memory bandwidth, so the speed-up shrinks towards 1× whatever the
instructions.

## Running them

```sh
cargo bench                          # all benchmarks, about 20 minutes
cargo bench --bench math -- acos     # one group
cargo bench --features rayon         # with the parallel results
python3 tools/bench_table.py         # the tables of these pages
```

A machine with AVX-512 runs the AVX-512 backend; to measure AVX2 on it, use
a machine without AVX-512. On `aarch64`, NEON is always used.

## Results

- [AVX2](./avx2.md)
- [AVX-512](./avx512.md)
- [NEON](./neon.md)
