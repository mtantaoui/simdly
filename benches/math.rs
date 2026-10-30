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
    fn hypot(self, y: Self) -> Self;
}

impl Float for f32 {
    const NAME: &str = "f32";
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn hypot(self, y: Self) -> Self {
        f32::hypot(self, y)
    }
}

impl Float for f64 {
    const NAME: &str = "f64";
    fn from_f64(x: f64) -> Self {
        x
    }
    fn hypot(self, y: Self) -> Self {
        f64::hypot(self, y)
    }
}

/// Both signs, so `abs` changes half of them.
fn signed<T: Float>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from_f64(i as f64 * 0.37 - 50.0))
        .collect()
}

/// Positive, so `sqrt` and `ln` give no NaN.
fn positive<T: Float>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from_f64(i as f64 * 0.37 + 1.0))
        .collect()
}

/// In `[−1, 1]`, the domain of `acos` and `asin`.
fn unit<T: Float>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from_f64((i as f64 * 0.37).sin()))
        .collect()
}

/// In `[−80, 80]`: finite `exp` for `f32` and `f64`.
fn moderate<T: Float>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from_f64((i as f64 * 0.37).sin() * 80.0))
        .collect()
}

type Input<T> = fn(usize) -> Vec<T>;

/// `simdly(x)` vs `x.iter().map(f).collect()`.
fn out_of_place<T: Float>(
    group: &mut Group,
    input: Input<T>,
    simdly: impl Fn(&[T]) -> Vec<T>,
    f: impl Fn(T) -> T,
) {
    for &len in LENS {
        let x = input(len);
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

/// `simdly(x)` vs `for v in x { *v = f(*v) }`. Repeating it is fine: `abs`,
/// `ceil` and `floor` keep the values after the first pass, `sqrt` keeps them
/// positive, and `atan` and `cbrt` finite.
fn in_place<T: Float>(
    group: &mut Group,
    input: Input<T>,
    simdly: impl Fn(&mut [T]),
    f: impl Fn(T) -> T,
) {
    for &len in LENS {
        let mut x = input(len);
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
    out_of_place::<f32>(&mut group, signed, |x| x.abs(), f32::abs);
    out_of_place::<f64>(&mut group, signed, |x| x.abs(), f64::abs);
    group.finish();

    let mut group = c.benchmark_group("abs_in_place");
    in_place::<f32>(&mut group, signed, |x| x.abs_in_place(), f32::abs);
    in_place::<f64>(&mut group, signed, |x| x.abs_in_place(), f64::abs);
    group.finish();
}

/// Out of place only: in place, a second pass would see values above 1.
fn acos(c: &mut Criterion) {
    let mut group = c.benchmark_group("acos");
    out_of_place::<f32>(&mut group, unit, |x| x.acos(), f32::acos);
    out_of_place::<f64>(&mut group, unit, |x| x.acos(), f64::acos);
    group.finish();
}

/// Out of place only, as for `acos`.
fn asin(c: &mut Criterion) {
    let mut group = c.benchmark_group("asin");
    out_of_place::<f32>(&mut group, unit, |x| x.asin(), f32::asin);
    out_of_place::<f64>(&mut group, unit, |x| x.asin(), f64::asin);
    group.finish();
}

/// In place as well: `atan` of any value is finite.
fn atan(c: &mut Criterion) {
    let mut group = c.benchmark_group("atan");
    out_of_place::<f32>(&mut group, signed, |x| x.atan(), f32::atan);
    out_of_place::<f64>(&mut group, signed, |x| x.atan(), f64::atan);
    group.finish();

    let mut group = c.benchmark_group("atan_in_place");
    in_place::<f32>(&mut group, signed, |x| x.atan_in_place(), f32::atan);
    in_place::<f64>(&mut group, signed, |x| x.atan_in_place(), f64::atan);
    group.finish();
}

/// `simdly(a, b)` vs `a.iter().zip(b).map(|(a, b)| f(a, b))`, over all
/// four quadrants.
fn binary_vs_scalar<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&[T], &[T]) -> Vec<T>,
    f: impl Fn(T, T) -> T,
) {
    for &len in LENS {
        let (y, x) = (
            signed::<T>(len),
            (0..len)
                .map(|i| T::from_f64((i as f64 * 0.73).cos() * 30.0))
                .collect::<Vec<T>>(),
        );
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(
            BenchmarkId::new(format!("simdly/{}", T::NAME), len),
            |bench| bench.iter(|| simdly(black_box(&y), black_box(&x))),
        );
        group.bench_function(
            BenchmarkId::new(format!("scalar/{}", T::NAME), len),
            |bench| {
                bench.iter(|| {
                    black_box(&y)
                        .iter()
                        .zip(black_box(&x))
                        .map(|(&a, &b)| f(a, b))
                        .collect::<Vec<T>>()
                })
            },
        );
    }
}

/// `simdly(v)` vs `scalar(v)` on `N` slices, mixed signs and magnitudes.
fn nary_vs_scalar<T: Float, const N: usize>(
    group: &mut Group,
    simdly: impl Fn(&[Vec<T>; N]) -> Vec<T>,
    scalar: impl Fn(&[Vec<T>; N]) -> Vec<T>,
) {
    for &len in LENS {
        let v: [Vec<T>; N] = std::array::from_fn(|k| {
            (0..len)
                .map(|i| {
                    T::from_f64(
                        (i as f64 * (0.37 + 0.36 * k as f64)).sin() * (50.0 - 10.0 * k as f64),
                    )
                })
                .collect()
        });
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(
            BenchmarkId::new(format!("simdly/{}", T::NAME), len),
            |bench| bench.iter(|| simdly(black_box(&v))),
        );
        group.bench_function(
            BenchmarkId::new(format!("scalar/{}", T::NAME), len),
            |bench| bench.iter(|| scalar(black_box(&v))),
        );
    }
}

fn atan2(c: &mut Criterion) {
    let mut group = c.benchmark_group("atan2");
    binary_vs_scalar::<f32>(&mut group, |y, x| y.atan2(x), f32::atan2);
    binary_vs_scalar::<f64>(&mut group, |y, x| y.atan2(x), f64::atan2);
    group.finish();
}

fn cbrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("cbrt");
    out_of_place::<f32>(&mut group, signed, |x| x.cbrt(), f32::cbrt);
    out_of_place::<f64>(&mut group, signed, |x| x.cbrt(), f64::cbrt);
    group.finish();

    let mut group = c.benchmark_group("cbrt_in_place");
    in_place::<f32>(&mut group, signed, |x| x.cbrt_in_place(), f32::cbrt);
    in_place::<f64>(&mut group, signed, |x| x.cbrt_in_place(), f64::cbrt);
    group.finish();
}

/// Out of place only: in place, a second pass would overflow.
fn ceil(c: &mut Criterion) {
    let mut group = c.benchmark_group("ceil");
    out_of_place::<f32>(&mut group, signed, |x| x.ceil(), f32::ceil);
    out_of_place::<f64>(&mut group, signed, |x| x.ceil(), f64::ceil);
    group.finish();

    let mut group = c.benchmark_group("ceil_in_place");
    in_place::<f32>(&mut group, signed, |x| x.ceil_in_place(), f32::ceil);
    in_place::<f64>(&mut group, signed, |x| x.ceil_in_place(), f64::ceil);
    group.finish();
}

fn exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp");
    out_of_place::<f32>(&mut group, moderate, |x| x.exp(), f32::exp);
    out_of_place::<f64>(&mut group, moderate, |x| x.exp(), f64::exp);
    group.finish();
}

fn floor(c: &mut Criterion) {
    let mut group = c.benchmark_group("floor");
    out_of_place::<f32>(&mut group, signed, |x| x.floor(), f32::floor);
    out_of_place::<f64>(&mut group, signed, |x| x.floor(), f64::floor);
    group.finish();

    let mut group = c.benchmark_group("floor_in_place");
    in_place::<f32>(&mut group, signed, |x| x.floor_in_place(), f32::floor);
    in_place::<f64>(&mut group, signed, |x| x.floor_in_place(), f64::floor);
    group.finish();
}

/// Out of place only: in place, a second pass would see negative values.
fn ln(c: &mut Criterion) {
    let mut group = c.benchmark_group("ln");
    out_of_place::<f32>(&mut group, positive, |x| x.ln(), f32::ln);
    out_of_place::<f64>(&mut group, positive, |x| x.ln(), f64::ln);
    group.finish();
}

fn hypot(c: &mut Criterion) {
    let mut group = c.benchmark_group("hypot");
    binary_vs_scalar::<f32>(&mut group, |x, y| x.hypot(y), f32::hypot);
    binary_vs_scalar::<f64>(&mut group, |x, y| x.hypot(y), f64::hypot);
    group.finish();
}

/// Against `std`'s overflow-safe way: nested `hypot`s.
fn hypot3(c: &mut Criterion) {
    fn bench<T: Float>(group: &mut Group)
    where
        [T]: SliceMath<T>,
    {
        nary_vs_scalar::<T, 3>(
            group,
            |[x, y, z]| x.hypot3(y, z),
            |[x, y, z]| {
                x.iter()
                    .zip(y)
                    .zip(z)
                    .map(|((&a, &b), &c)| T::hypot(T::hypot(a, b), c))
                    .collect()
            },
        );
    }
    let mut group = c.benchmark_group("hypot3");
    bench::<f32>(&mut group);
    bench::<f64>(&mut group);
    group.finish();
}

fn hypot4(c: &mut Criterion) {
    fn bench<T: Float>(group: &mut Group)
    where
        [T]: SliceMath<T>,
    {
        nary_vs_scalar::<T, 4>(
            group,
            |[x, y, z, w]| x.hypot4(y, z, w),
            |[x, y, z, w]| {
                x.iter()
                    .zip(y)
                    .zip(z)
                    .zip(w)
                    .map(|(((&a, &b), &c), &d)| T::hypot(T::hypot(T::hypot(a, b), c), d))
                    .collect()
            },
        );
    }
    let mut group = c.benchmark_group("hypot4");
    bench::<f32>(&mut group);
    bench::<f64>(&mut group);
    group.finish();
}

fn cos(c: &mut Criterion) {
    let mut group = c.benchmark_group("cos");
    out_of_place::<f32>(&mut group, signed, |x| x.cos(), f32::cos);
    out_of_place::<f64>(&mut group, signed, |x| x.cos(), f64::cos);
    group.finish();

    let mut group = c.benchmark_group("cos_in_place");
    in_place::<f32>(&mut group, signed, |x| x.cos_in_place(), f32::cos);
    in_place::<f64>(&mut group, signed, |x| x.cos_in_place(), f64::cos);
    group.finish();
}

fn tan(c: &mut Criterion) {
    let mut group = c.benchmark_group("tan");
    out_of_place::<f32>(&mut group, signed, |x| x.tan(), f32::tan);
    out_of_place::<f64>(&mut group, signed, |x| x.tan(), f64::tan);
    group.finish();

    let mut group = c.benchmark_group("tan_in_place");
    in_place::<f32>(&mut group, signed, |x| x.tan_in_place(), f32::tan);
    in_place::<f64>(&mut group, signed, |x| x.tan_in_place(), f64::tan);
    group.finish();
}

fn sin(c: &mut Criterion) {
    let mut group = c.benchmark_group("sin");
    out_of_place::<f32>(&mut group, signed, |x| x.sin(), f32::sin);
    out_of_place::<f64>(&mut group, signed, |x| x.sin(), f64::sin);
    group.finish();

    let mut group = c.benchmark_group("sin_in_place");
    in_place::<f32>(&mut group, signed, |x| x.sin_in_place(), f32::sin);
    in_place::<f64>(&mut group, signed, |x| x.sin_in_place(), f64::sin);
    group.finish();
}

fn sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqrt");
    out_of_place::<f32>(&mut group, positive, |x| x.sqrt(), f32::sqrt);
    out_of_place::<f64>(&mut group, positive, |x| x.sqrt(), f64::sqrt);
    group.finish();

    let mut group = c.benchmark_group("sqrt_in_place");
    in_place::<f32>(&mut group, positive, |x| x.sqrt_in_place(), f32::sqrt);
    in_place::<f64>(&mut group, positive, |x| x.sqrt_in_place(), f64::sqrt);
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
    targets = abs, acos, asin, atan, atan2, cbrt, ceil, cos, exp, floor, hypot, hypot3, hypot4, ln, sin, sqrt, tan
}
criterion_main!(benches);
