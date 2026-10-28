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

The backend is currently selected at compile time from `target-feature`
flags. On `x86_64`, enable AVX2 with:

```sh
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --release
```

## Cargo features

| Feature | Default | Enables                                         |
|---------|---------|-------------------------------------------------|
| `std`   | yes     | runtime CPU detection (planned); implies `alloc` |
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
