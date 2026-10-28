# Contributing to simdly

## Development setup

No flags are needed: with `std`, every backend your CPU supports is
detected at runtime and tested. Without `std`, only compile-time target
features count:

```sh
cargo test                                                          # runtime detection
cargo test --no-default-features                                    # no_std: scalar on x86_64
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo test --no-default-features  # no_std with AVX2
cargo build --target thumbv7em-none-eabihf --no-default-features    # cross-compile
```

`cargo run --example backend` prints the backend in use, and `cargo bench`
runs the benchmarks (criterion; HTML reports in
`target/criterion/report/index.html`).

Set `SIMDLY_REQUIRE_BACKENDS=avx2,avx512` to make the tests fail, instead of
skip, when a backend is missing (CI does this).

Before pushing:

```sh
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
```

## Layout

| Path                 | Contents                                                   |
|----------------------|------------------------------------------------------------|
| `src/macros.rs`      | `cfg_x86!` / `cfg_neon!` / `cfg_simd!` — architecture selection; `dispatch!` |
| `src/detect.rs`      | CPU feature detection (`has_avx2_fma`, …) and `backend()`  |
| `src/simd/<isa>/`    | Register types (`F32x8`, …) implementing `SimdVector`      |
| `src/math/<fn>/`     | One folder per function: `mod.rs` (docs), `consts.rs`, one file per ISA |
| `src/math/driver.rs` | Chunk + tail loops shared by every function                |
| `src/ops/`           | Element-wise arithmetic and reductions                     |
| `src/testing/`       | Test-only helpers (ULP distance, sweep generators)         |
| `examples/`          | One runnable example per public operation                  |
| `benches/`           | One benchmark per public operation, vs a plain scalar loop |

## Conventions

- **Backend selection**: never write an architecture or target-feature `cfg`
  by hand; use the macros in `src/macros.rs` and the checks in
  `src/detect.rs`.
- **Entry points**: each public function calls its backends through
  `dispatch!`; backend functions are safe `#[target_feature]` functions.
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
