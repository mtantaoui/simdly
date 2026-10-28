//! `SliceExt` comparisons against the plain scalar loop, from L1-resident to
//! larger-than-cache slices.
//!
//! ```sh
//! cargo bench --bench comparison
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use simdly::SliceExt;

const LENS: &[usize] = &[64, 4_096, 262_144, 4_194_304];

/// The value `_scalar` comparisons use; it falls inside the inputs' range.
const RHS: f32 = 8.0;

/// Values on both sides of `b`, so the results are not all equal.
fn inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..len).map(|i| (i % 17) as f32).collect();
    let b = (0..len).map(|i| (i % 13) as f32).collect();
    (a, b)
}

/// `simdly(a, b)` vs `a.iter().zip(b).map(op).collect()`, then the `_scalar`
/// form against one value.
fn compare(
    c: &mut Criterion,
    name: &str,
    simdly: impl Fn(&[f32], &[f32]) -> Vec<bool>,
    simdly_scalar: impl Fn(&[f32], f32) -> Vec<bool>,
    op: impl Fn(f32, f32) -> bool + Copy,
) {
    let mut group = c.benchmark_group(name);
    for &len in LENS {
        let (a, b) = inputs(len);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(BenchmarkId::new("simdly", len), |bench| {
            bench.iter(|| simdly(black_box(&a), black_box(&b)))
        });
        group.bench_function(BenchmarkId::new("scalar", len), |bench| {
            bench.iter(|| {
                black_box(&a)
                    .iter()
                    .zip(black_box(&b))
                    .map(|(&x, &y)| op(x, y))
                    .collect::<Vec<bool>>()
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group(format!("{name}_scalar"));
    for &len in LENS {
        let (a, _) = inputs(len);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(BenchmarkId::new("simdly", len), |bench| {
            bench.iter(|| simdly_scalar(black_box(&a), black_box(RHS)))
        });
        group.bench_function(BenchmarkId::new("scalar", len), |bench| {
            bench.iter(|| {
                let s = black_box(RHS);
                black_box(&a)
                    .iter()
                    .map(|&x| op(x, s))
                    .collect::<Vec<bool>>()
            })
        });
    }
    group.finish();
}

fn less(c: &mut Criterion) {
    compare(
        c,
        "less",
        |a, b| a.less(b),
        |a, s| a.less_scalar(s),
        |x, y| x < y,
    );
}

fn equal(c: &mut Criterion) {
    compare(
        c,
        "equal",
        |a, b| a.equal(b),
        |a, s| a.equal_scalar(s),
        |x, y| x == y,
    );
}

/// Shorter than criterion's defaults, which would take ~8 minutes here.
fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = config();
    targets = less, equal
}
criterion_main!(benches);
