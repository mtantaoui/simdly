# Contributing to simdly

## Development setup

```sh
# x86_64 (AVX2 backend)
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test

# aarch64 (NEON is on by default)
cargo test

# scalar fallback / no_std
cargo test
cargo build --no-default-features
```

Before pushing:

```sh
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
```

## Layout

| Path                 | Contents                                                   |
|----------------------|------------------------------------------------------------|
| `src/macros.rs`      | `cfg_avx2!` / `cfg_avx512!` / `cfg_neon!` / `cfg_scalar!` — the only place backend conditions are written |
| `src/simd/<isa>/`    | Register types (`F32x8`, …) and their `Load` / `Store` impls |
| `src/math/<fn>/`     | One folder per function: `mod.rs` (docs), `consts.rs`, one file per ISA |
| `src/math/driver.rs` | Chunk + tail loops shared by every function                |
| `src/ops/`           | Element-wise arithmetic and reductions                     |
| `src/testing/`       | Test-only helpers (ULP distance, sweep generators)         |

## Conventions

- **Backend selection**: never write a `#[cfg(target_feature = ...)]`
  backend predicate by hand; use the macros in `src/macros.rs`.
- **Constants**: port verbatim from musl libc / fdlibm as
  `f64::from_bits(0x...)`, citing the source file in a comment.
- **Accuracy**: every function documents a worst-case ULP bound and has a
  sweep test that exercises it.
- **Unsafe**: every `unsafe` block gets a `// SAFETY:` comment.

## License

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed under MIT OR Apache-2.0, without any
additional terms or conditions.
