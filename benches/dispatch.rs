//! Cost of runtime backend selection, which every dispatched call pays.
//!
//! ```sh
//! cargo bench --bench dispatch
//! ```

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

/// Feature detection (cached by `std`) plus the fastest-first checks.
fn backend(c: &mut Criterion) {
    c.bench_function("backend", |b| b.iter(|| black_box(simdly::backend())));
}

criterion_group!(benches, backend);
criterion_main!(benches);
