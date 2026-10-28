//! Cost of runtime backend selection, which every dispatched call pays.
//!
//! ```sh
//! cargo bench --bench dispatch
//! ```

fn main() {
    divan::main();
}

/// Feature detection (cached by `std`) plus the fastest-first checks.
#[divan::bench(sample_size = 10_000)]
fn backend() -> simdly::Backend {
    simdly::backend()
}
