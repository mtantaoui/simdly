# wasm

Speed-up of simdly's WebAssembly SIMD backend (4 `f32` / 2 `f64` lanes,
built with `-C target-feature=+simd128`) over the plain loop a user would
write instead; see [Methodology](./README.md).

> **Pending:** full results. criterion does not build for WebAssembly, so
> these benchmarks do not run there yet.

A first measurement, under V8 (Node 20) on an Intel Core Ultra 7 155H, 4096
`f32` elements, best of 2000 runs. The plain loop calls Rust's functions,
which on `wasm32` are `libm`'s:

| Function  | Plain loop | simdly | Speed-up |
|-----------|-----------:|-------:|---------:|
| `exp`     | 19.56 µs   | 3.34 µs | 5.9×    |
| `tanh`    | 34.14 µs   | 18.06 µs | 1.9×   |
| `sigmoid` | 22.62 µs   | 10.89 µs | 2.1×   |

The plain loop has to be built without `simd128` to stay scalar, or with
it and calls that cannot be vectorised, as here: with `simd128`, LLVM
vectorises loops of branch-free arithmetic by itself, simdly's scalar
kernels included.
