# simdly

[![Linux](https://github.com/mtantaoui/simdly/actions/workflows/linux.yml/badge.svg)](https://github.com/mtantaoui/simdly/actions/workflows/linux.yml)
[![macOS](https://github.com/mtantaoui/simdly/actions/workflows/macos.yml/badge.svg)](https://github.com/mtantaoui/simdly/actions/workflows/macos.yml)
[![Windows](https://github.com/mtantaoui/simdly/actions/workflows/windows.yml/badge.svg)](https://github.com/mtantaoui/simdly/actions/workflows/windows.yml)

SIMD implementations of mathematical functions for Rust, with a documented
worst-case ULP bound for every function.

> **Status:** early development. Element-wise arithmetic, comparisons (like
> NumPy's), `abs`, `sqrt` and `acos` on `f32` and `f64` slices are implemented; see
> [Implementation status](#implementation-status).

```rust
use simdly::SliceExt;

let a = vec![1.0_f32, 2.0, 3.0];
let b = vec![4.0_f32, 5.0, 6.0];
let c = a.add(&b);   // [5.0, 7.0, 9.0]
let m = a.less(&b);  // [true, true, true]

use simdly::SliceMath;

let d = a.sub(&b).abs();  // [3.0, 3.0, 3.0]
```

## Implementation status

For `f32` and `f64`. ✅ implemented, 🚧 planned. An operation without a SIMD
backend still works everywhere: it runs on the next one available (AVX2, then
scalar).

| Operation                                      | AVX-512 | AVX2 | NEON | scalar |
|------------------------------------------------|---------|------|------|--------|
| Arithmetic: `add` `sub` `mul` `div`, `_assign` | ✅      | ✅   | ✅   | ✅     |
| Comparisons: `less` … `not_equal`, `_scalar`   | ✅      | ✅   | ✅   | ✅     |
| `abs`                                          | ✅      | ✅   | ✅   | ✅     |
| `acos`                                         | ✅      | ✅   | ✅   | ✅     |
| `asin`                                         | 🚧      | 🚧   | 🚧   | 🚧     |
| `atan`                                         | 🚧      | 🚧   | 🚧   | 🚧     |
| `atan2`                                        | 🚧      | 🚧   | 🚧   | 🚧     |
| `cbrt`                                         | 🚧      | 🚧   | 🚧   | 🚧     |
| `floor`                                        | 🚧      | 🚧   | 🚧   | 🚧     |
| `exp`                                          | 🚧      | 🚧   | 🚧   | 🚧     |
| `fmod`                                         | 🚧      | 🚧   | 🚧   | 🚧     |
| `ln`                                           | 🚧      | 🚧   | 🚧   | 🚧     |
| `hypot`                                        | 🚧      | 🚧   | 🚧   | 🚧     |
| `pow`                                          | 🚧      | 🚧   | 🚧   | 🚧     |
| `sin`                                          | 🚧      | 🚧   | 🚧   | 🚧     |
| `cos`                                          | 🚧      | 🚧   | 🚧   | 🚧     |
| `tan`                                          | 🚧      | 🚧   | 🚧   | 🚧     |
| `sqrt`                                         | ✅      | ✅   | ✅   | ✅     |
| `ceil`                                         | 🚧      | 🚧   | 🚧   | 🚧     |
| `hypot3`                                       | 🚧      | 🚧   | 🚧   | 🚧     |
| `hypot4`                                       | 🚧      | 🚧   | 🚧   | 🚧     |

The [book](book/) explains the maths behind each function and collects the
benchmarks (`mdbook serve book`).

## Supported targets

| Architecture | ISA             | `f32` lanes | `f64` lanes |
|--------------|-----------------|-------------|-------------|
| `x86_64`     | AVX2 + FMA      | 8           | 4           |
| `x86_64`     | AVX-512         | 16          | 8           |
| `aarch64`    | NEON            | 4           | 2           |
| any          | scalar fallback | 1           | 1           |

No flags are needed: on `x86_64` the fastest backend the CPU supports is
detected at runtime, and cross-compiling works as usual. On `aarch64` NEON is
always used. `simdly::backend()` reports the one in use.

Without the `std` feature, runtime detection is unavailable and only
backends enabled at compile time are used, e.g. on `x86_64`:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

## Cargo features

| Feature | Default | Enables                                         |
|---------|---------|-------------------------------------------------|
| `std`   | yes     | runtime CPU detection; implies `alloc`          |
| `alloc` | via `std` | `Vec`-returning slice APIs (`SliceExt::add`, …) |

With `default-features = false` the crate is `no_std`.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
