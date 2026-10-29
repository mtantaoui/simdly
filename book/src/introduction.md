# Introduction

simdly computes element-wise maths on `f32` and `f64` slices with SIMD
instructions: AVX2 and AVX-512 on `x86_64`, NEON on `aarch64`, and a scalar
fallback everywhere else.

```rust
use simdly::{SliceExt, SliceMath};

let x = vec![-1.0_f32, -0.5, 0.0, 0.5, 1.0];
let angles = x.acos();      // [π, 2π/3, π/2, π/3, 0]
let sum = x.add(&angles);
```

Every function documents a worst-case error in
[ULPs](./accuracy.md), and the same bound holds on every backend.

This book covers:

- **Design:** how one implementation runs on every backend, and how
  accuracy is measured.
- **Functions:** the maths behind each algorithm: range reduction, the
  approximation, and where the rounding error goes.
- **Benchmarks:** speed-ups over a plain scalar loop on each backend.

The API reference is on [docs.rs](https://docs.rs/simdly).
