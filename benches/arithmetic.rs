//! `SliceExt` arithmetic against the plain scalar loop, from L1-resident to
//! larger-than-cache slices.
//!
//! ```sh
//! cargo bench --bench arithmetic
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use simdly::SliceExt;

const LENS: &[usize] = &[64, 4_096, 262_144, 4_194_304];

fn inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..len).map(|i| i as f32 * 0.5 + 1.0).collect();
    let b = (0..len).map(|i| 3.0 - i as f32 * 0.25).collect();
    (a, b)
}

/// `simdly(a, b)` vs `a.iter().zip(b).map(op).collect()`.
fn out_of_place(
    c: &mut Criterion,
    name: &str,
    simdly: impl Fn(&[f32], &[f32]) -> Vec<f32>,
    op: impl Fn(f32, f32) -> f32 + Copy,
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
                    .collect::<Vec<f32>>()
            })
        });
    }
    group.finish();
}

/// `simdly(a, b)` vs `for (x, y) in a.zip(b) { *x = op(*x, y) }`.
///
/// `b` is all ones so that repeating `a op= b` keeps `a` finite and normal:
/// drifting into subnormals would slow both sides down and skew the result.
fn in_place(
    c: &mut Criterion,
    name: &str,
    simdly: impl Fn(&mut [f32], &[f32]),
    op: impl Fn(f32, f32) -> f32 + Copy,
) {
    let mut group = c.benchmark_group(name);
    for &len in LENS {
        let (mut a, b) = (inputs(len).0, vec![1.0_f32; len]);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(BenchmarkId::new("simdly", len), |bench| {
            bench.iter(|| simdly(black_box(&mut a), black_box(&b)))
        });
        group.bench_function(BenchmarkId::new("scalar", len), |bench| {
            bench.iter(|| {
                for (x, &y) in black_box(&mut a).iter_mut().zip(black_box(&b)) {
                    *x = op(*x, y);
                }
            })
        });
    }
    group.finish();
}

fn add(c: &mut Criterion) {
    out_of_place(c, "add", |a, b| a.add(b), |x, y| x + y);
    in_place(c, "add_assign", |a, b| a.add_assign(b), |x, y| x + y);
}

fn sub(c: &mut Criterion) {
    out_of_place(c, "sub", |a, b| a.sub(b), |x, y| x - y);
    in_place(c, "sub_assign", |a, b| a.sub_assign(b), |x, y| x - y);
}

fn mul(c: &mut Criterion) {
    out_of_place(c, "mul", |a, b| a.mul(b), |x, y| x * y);
    in_place(c, "mul_assign", |a, b| a.mul_assign(b), |x, y| x * y);
}

fn div(c: &mut Criterion) {
    out_of_place(c, "div", |a, b| a.div(b), |x, y| x / y);
    in_place(c, "div_assign", |a, b| a.div_assign(b), |x, y| x / y);
}

/// Shorter than criterion's defaults, which would take ~15 minutes here.
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
