# simdly

[![Linux](https://github.com/mtantaoui/simdly/actions/workflows/linux.yml/badge.svg)](https://github.com/mtantaoui/simdly/actions/workflows/linux.yml)
[![macOS](https://github.com/mtantaoui/simdly/actions/workflows/macos.yml/badge.svg)](https://github.com/mtantaoui/simdly/actions/workflows/macos.yml)
[![Windows](https://github.com/mtantaoui/simdly/actions/workflows/windows.yml/badge.svg)](https://github.com/mtantaoui/simdly/actions/workflows/windows.yml)

SIMD implementations of mathematical functions for Rust, with a documented
worst-case ULP bound for every function.

> **Status:** early development — no functions are implemented yet.

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
| `alloc` | via `std` | `Vec`-returning slice APIs (planned)          |

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
