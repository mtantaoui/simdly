//! `SliceMath` functions against the plain scalar loop, from L1-resident to
//! larger-than-cache slices.
//!
//! ```sh
//! cargo bench --bench math
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use simdly::SliceMath;

const LENS: &[usize] = &[64, 4_096, 262_144, 4_194_304];

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

/// Both signs, so `abs` changes half of them.
fn input<T: Float>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from_f64(i as f64 * 0.37 - 50.0))
        .collect()
}

/// `simdly(x)` vs `x.iter().map(f).collect()`.
fn out_of_place<T: Float>(group: &mut Group, simdly: impl Fn(&[T]) -> Vec<T>, f: impl Fn(T) -> T) {
    for &len in LENS {
        let x = input::<T>(len);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(
            BenchmarkId::new(format!("simdly/{}", T::NAME), len),
            |bench| bench.iter(|| simdly(black_box(&x))),
        );
        group.bench_function(
            BenchmarkId::new(format!("scalar/{}", T::NAME), len),
            |bench| bench.iter(|| black_box(&x).iter().map(|&v| f(v)).collect::<Vec<T>>()),
        );
    }
}

/// `simdly(x)` vs `for v in x { *v = f(*v) }`. Repeating it is fine for `abs`:
/// the values stay the same after the first pass.
fn in_place<T: Float>(group: &mut Group, simdly: impl Fn(&mut [T]), f: impl Fn(T) -> T) {
    for &len in LENS {
        let mut x = input::<T>(len);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(
            BenchmarkId::new(format!("simdly/{}", T::NAME), len),
            |bench| bench.iter(|| simdly(black_box(&mut x))),
        );
        group.bench_function(
            BenchmarkId::new(format!("scalar/{}", T::NAME), len),
            |bench| {
                bench.iter(|| {
                    for v in black_box(&mut x).iter_mut() {
                        *v = f(*v);
                    }
                })
            },
        );
    }
}

fn abs(c: &mut Criterion) {
    let mut group = c.benchmark_group("abs");
    out_of_place::<f32>(&mut group, |x| x.abs(), f32::abs);
    out_of_place::<f64>(&mut group, |x| x.abs(), f64::abs);
    group.finish();

    let mut group = c.benchmark_group("abs_in_place");
    in_place::<f32>(&mut group, |x| x.abs_in_place(), f32::abs);
    in_place::<f64>(&mut group, |x| x.abs_in_place(), f64::abs);
    group.finish();
}

/// Shorter than criterion's defaults.
fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = config();
    targets = abs
}
criterion_main!(benches);
