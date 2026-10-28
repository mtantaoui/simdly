//! `SliceExt` comparisons against the plain scalar loop, from L1-resident to
//! larger-than-cache slices.
//!
//! ```sh
//! cargo bench --bench comparison
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use simdly::SliceExt;

const LENS: &[usize] = &[64, 4_096, 262_144, 4_194_304];

/// The value `_scalar` comparisons use; it falls inside the inputs' range.
const RHS: f64 = 8.0;

type Group<'a> = BenchmarkGroup<'a, WallTime>;

trait Float: Copy {
    const NAME: &str;
    fn from_f64(x: f64) -> Self;
}

impl Float for f32 {
    const NAME: &str = "f32";
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl Float for f64 {
    const NAME: &str = "f64";
    fn from_f64(x: f64) -> Self {
        x
    }
}

/// Values on both sides of `b`, so the results are not all equal.
fn inputs<T: Float>(len: usize) -> (Vec<T>, Vec<T>) {
    let a = (0..len).map(|i| T::from_f64((i % 17) as f64)).collect();
    let b = (0..len).map(|i| T::from_f64((i % 13) as f64)).collect();
    (a, b)
}

/// `simdly(a, b)` vs `a.iter().zip(b).map(op).collect()`.
fn slices<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&[T], &[T]) -> Vec<bool>,
    op: impl Fn(T, T) -> bool + Copy,
) {
    for &len in LENS {
        let (a, b) = inputs::<T>(len);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(
            BenchmarkId::new(format!("simdly/{}", T::NAME), len),
            |bench| bench.iter(|| simdly(black_box(&a), black_box(&b))),
        );
        group.bench_function(
            BenchmarkId::new(format!("scalar/{}", T::NAME), len),
            |bench| {
                bench.iter(|| {
                    black_box(&a)
                        .iter()
                        .zip(black_box(&b))
                        .map(|(&x, &y)| op(x, y))
                        .collect::<Vec<bool>>()
                })
            },
        );
    }
}

/// The `_scalar` form against [`RHS`].
fn scalar<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&[T], T) -> Vec<bool>,
    op: impl Fn(T, T) -> bool + Copy,
) {
    let rhs = T::from_f64(RHS);
    for &len in LENS {
        let (a, _) = inputs::<T>(len);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(
            BenchmarkId::new(format!("simdly/{}", T::NAME), len),
            |bench| bench.iter(|| simdly(black_box(&a), black_box(rhs))),
        );
        group.bench_function(
            BenchmarkId::new(format!("scalar/{}", T::NAME), len),
            |bench| {
                bench.iter(|| {
                    let s = black_box(rhs);
                    black_box(&a)
                        .iter()
                        .map(|&x| op(x, s))
                        .collect::<Vec<bool>>()
                })
            },
        );
    }
}

/// `less` for both element types; `equal` for `f32` only, to keep the run
/// short.
fn less(c: &mut Criterion) {
    let mut group = c.benchmark_group("less");
    slices::<f32>(&mut group, |a, b| a.less(b), |x, y| x < y);
    slices::<f64>(&mut group, |a, b| a.less(b), |x, y| x < y);
    group.finish();

    let mut group = c.benchmark_group("less_scalar");
    scalar::<f32>(&mut group, |a, s| a.less_scalar(s), |x, y| x < y);
    scalar::<f64>(&mut group, |a, s| a.less_scalar(s), |x, y| x < y);
    group.finish();
}

fn equal(c: &mut Criterion) {
    let mut group = c.benchmark_group("equal");
    slices::<f32>(&mut group, |a, b| a.equal(b), |x, y| x == y);
    group.finish();

    let mut group = c.benchmark_group("equal_scalar");
    scalar::<f32>(&mut group, |a, s| a.equal_scalar(s), |x, y| x == y);
    group.finish();
}

/// Shorter than criterion's defaults, which would take ~12 minutes here.
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
