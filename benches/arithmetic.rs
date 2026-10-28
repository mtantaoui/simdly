//! `SliceExt` arithmetic against the plain scalar loop, from L1-resident to
//! larger-than-cache slices.
//!
//! ```sh
//! cargo bench --bench arithmetic
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use simdly::SliceExt;

const LENS: &[usize] = &[64, 4_096, 262_144, 4_194_304];

type Group<'a> = BenchmarkGroup<'a, WallTime>;

trait Float: Copy {
    const NAME: &str;
    const ONE: Self;
    fn from_f64(x: f64) -> Self;
}

impl Float for f32 {
    const NAME: &str = "f32";
    const ONE: Self = 1.0;
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl Float for f64 {
    const NAME: &str = "f64";
    const ONE: Self = 1.0;
    fn from_f64(x: f64) -> Self {
        x
    }
}

fn inputs<T: Float>(len: usize) -> (Vec<T>, Vec<T>) {
    let a = (0..len)
        .map(|i| T::from_f64(i as f64 * 0.5 + 1.0))
        .collect();
    let b = (0..len)
        .map(|i| T::from_f64(3.0 - i as f64 * 0.25))
        .collect();
    (a, b)
}

/// `simdly(a, b)` vs `a.iter().zip(b).map(op).collect()`.
fn out_of_place<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&[T], &[T]) -> Vec<T>,
    op: impl Fn(T, T) -> T + Copy,
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
                        .collect::<Vec<T>>()
                })
            },
        );
    }
}

/// `simdly(a, b)` vs `for (x, y) in a.zip(b) { *x = op(*x, y) }`.
///
/// `b` is all ones so that repeating `a op= b` keeps `a` finite and normal:
/// drifting into subnormals would slow both sides down and skew the result.
fn in_place<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&mut [T], &[T]),
    op: impl Fn(T, T) -> T + Copy,
) {
    for &len in LENS {
        let (mut a, b) = (inputs::<T>(len).0, vec![T::ONE; len]);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(
            BenchmarkId::new(format!("simdly/{}", T::NAME), len),
            |bench| bench.iter(|| simdly(black_box(&mut a), black_box(&b))),
        );
        group.bench_function(
            BenchmarkId::new(format!("scalar/{}", T::NAME), len),
            |bench| {
                bench.iter(|| {
                    for (x, &y) in black_box(&mut a).iter_mut().zip(black_box(&b)) {
                        *x = op(*x, y);
                    }
                })
            },
        );
    }
}

/// `add` and `add_assign` for both element types; the other operations for
/// `f32` only, to keep the run short.
fn add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    out_of_place::<f32>(&mut group, |a, b| a.add(b), |x, y| x + y);
    out_of_place::<f64>(&mut group, |a, b| a.add(b), |x, y| x + y);
    group.finish();

    let mut group = c.benchmark_group("add_assign");
    in_place::<f32>(&mut group, |a, b| a.add_assign(b), |x, y| x + y);
    in_place::<f64>(&mut group, |a, b| a.add_assign(b), |x, y| x + y);
    group.finish();
}

fn sub(c: &mut Criterion) {
    let mut group = c.benchmark_group("sub");
    out_of_place::<f32>(&mut group, |a, b| a.sub(b), |x, y| x - y);
    group.finish();

    let mut group = c.benchmark_group("sub_assign");
    in_place::<f32>(&mut group, |a, b| a.sub_assign(b), |x, y| x - y);
    group.finish();
}

fn mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");
    out_of_place::<f32>(&mut group, |a, b| a.mul(b), |x, y| x * y);
    group.finish();

    let mut group = c.benchmark_group("mul_assign");
    in_place::<f32>(&mut group, |a, b| a.mul_assign(b), |x, y| x * y);
    group.finish();
}

fn div(c: &mut Criterion) {
    let mut group = c.benchmark_group("div");
    out_of_place::<f32>(&mut group, |a, b| a.div(b), |x, y| x / y);
    group.finish();

    let mut group = c.benchmark_group("div_assign");
    in_place::<f32>(&mut group, |a, b| a.div_assign(b), |x, y| x / y);
    group.finish();
}

/// Shorter than criterion's defaults, which would take ~20 minutes here.
fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = config();
    targets = add, sub, mul, div
}
criterion_main!(benches);
