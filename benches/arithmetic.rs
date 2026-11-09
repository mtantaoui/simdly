//! `SliceExt` arithmetic against the plain scalar loop, from L1-resident to
//! larger-than-cache slices. With `--features rayon`, the same groups also
//! time `ParSliceExt` (`par`) and the rayon loop (`rayon`), at every length
//! (`par_` runs on one thread up to twice `PARALLEL_SIMD_THRESHOLD`).
//!
//! ```sh
//! cargo bench --bench arithmetic
//! cargo bench --bench arithmetic --features rayon
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use simdly::ParSliceExt;
use simdly::SliceExt;

const LENS: &[usize] = &[64, 4_096, 262_144, 4_194_304];

type Group<'a> = BenchmarkGroup<'a, WallTime>;

/// `Some(f)` with the `rayon` feature, `None` (and `f` not compiled)
/// without.
#[cfg(feature = "rayon")]
macro_rules! par {
    ($f:expr) => {
        Some($f)
    };
}
#[cfg(not(feature = "rayon"))]
macro_rules! par {
    ($f:expr) => {
        None
    };
}

trait Float: Copy + Send + Sync {
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

type Binary<T> = fn(&[T], &[T]) -> Vec<T>;
type Assign<T> = fn(&mut [T], &[T]);
type BinaryScalar<T> = fn(&[T], T) -> Vec<T>;
type AssignScalar<T> = fn(&mut [T], T);

/// `simdly(a, b)` vs `a.iter().zip(b).map(op).collect()`; with `rayon`,
/// `par(a, b)` vs the same with `par_iter`.
fn out_of_place<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&[T], &[T]) -> Vec<T>,
    par: Option<Binary<T>>,
    op: impl Fn(T, T) -> T + Copy + Sync,
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
        #[cfg(feature = "rayon")]
        if let Some(par) = par {
            group.bench_function(BenchmarkId::new(format!("par/{}", T::NAME), len), |bench| {
                bench.iter(|| par(black_box(&a), black_box(&b)))
            });
            group.bench_function(
                BenchmarkId::new(format!("rayon/{}", T::NAME), len),
                |bench| {
                    bench.iter(|| {
                        black_box(&a)
                            .par_iter()
                            .zip(black_box(&b))
                            .map(|(&x, &y)| op(x, y))
                            .collect::<Vec<T>>()
                    })
                },
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = par;
}

/// `simdly(a, b)` vs `for (x, y) in a.zip(b) { *x = op(*x, y) }`; with
/// `rayon`, `par(a, b)` vs the same with `par_iter_mut`.
///
/// `b` is all ones so that repeating `a op= b` keeps `a` finite and normal:
/// drifting into subnormals would slow both sides down and skew the result.
fn in_place<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&mut [T], &[T]),
    par: Option<Assign<T>>,
    op: impl Fn(T, T) -> T + Copy + Sync,
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
        #[cfg(feature = "rayon")]
        if let Some(par) = par {
            group.bench_function(BenchmarkId::new(format!("par/{}", T::NAME), len), |bench| {
                bench.iter(|| par(black_box(&mut a), black_box(&b)))
            });
            group.bench_function(
                BenchmarkId::new(format!("rayon/{}", T::NAME), len),
                |bench| {
                    bench.iter(|| {
                        black_box(&mut a)
                            .par_iter_mut()
                            .zip(black_box(&b))
                            .for_each(|(x, &y)| *x = op(*x, y))
                    })
                },
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = par;
}

/// The `_scalar` variants: `simdly(a, 1)` vs `a.iter().map(op).collect()`
/// (out of place) or `for x in a { *x = op(*x, 1) }` (in place); with
/// `rayon`, `par` vs the same with `par_iter` / `par_iter_mut`. A right
/// operand of one keeps repeated in-place runs finite and normal.
fn with_scalar<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&[T], T) -> Vec<T>,
    par: Option<BinaryScalar<T>>,
    op: impl Fn(T, T) -> T + Copy + Sync,
) {
    for &len in LENS {
        let (a, k) = (inputs::<T>(len).0, T::ONE);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(
            BenchmarkId::new(format!("simdly/{}", T::NAME), len),
            |bench| bench.iter(|| simdly(black_box(&a), black_box(k))),
        );
        group.bench_function(
            BenchmarkId::new(format!("scalar/{}", T::NAME), len),
            |bench| {
                bench.iter(|| {
                    let k = black_box(k);
                    black_box(&a).iter().map(|&x| op(x, k)).collect::<Vec<T>>()
                })
            },
        );
        #[cfg(feature = "rayon")]
        if let Some(par) = par {
            group.bench_function(BenchmarkId::new(format!("par/{}", T::NAME), len), |bench| {
                bench.iter(|| par(black_box(&a), black_box(k)))
            });
            group.bench_function(
                BenchmarkId::new(format!("rayon/{}", T::NAME), len),
                |bench| {
                    bench.iter(|| {
                        let k = black_box(k);
                        black_box(&a)
                            .par_iter()
                            .map(|&x| op(x, k))
                            .collect::<Vec<T>>()
                    })
                },
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = par;
}

/// [`with_scalar`], in place.
fn with_scalar_in_place<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&mut [T], T),
    par: Option<AssignScalar<T>>,
    op: impl Fn(T, T) -> T + Copy + Sync,
) {
    for &len in LENS {
        let (mut a, k) = (inputs::<T>(len).0, T::ONE);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(
            BenchmarkId::new(format!("simdly/{}", T::NAME), len),
            |bench| bench.iter(|| simdly(black_box(&mut a), black_box(k))),
        );
        group.bench_function(
            BenchmarkId::new(format!("scalar/{}", T::NAME), len),
            |bench| {
                bench.iter(|| {
                    let k = black_box(k);
                    for x in black_box(&mut a).iter_mut() {
                        *x = op(*x, k);
                    }
                })
            },
        );
        #[cfg(feature = "rayon")]
        if let Some(par) = par {
            group.bench_function(BenchmarkId::new(format!("par/{}", T::NAME), len), |bench| {
                bench.iter(|| par(black_box(&mut a), black_box(k)))
            });
            group.bench_function(
                BenchmarkId::new(format!("rayon/{}", T::NAME), len),
                |bench| {
                    bench.iter(|| {
                        let k = black_box(k);
                        black_box(&mut a)
                            .par_iter_mut()
                            .for_each(|x| *x = op(*x, k))
                    })
                },
            );
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = par;
}

/// `add` and `add_assign` for both element types; the other operations for
/// `f32` only, to keep the run short.
fn add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    out_of_place::<f32>(
        &mut group,
        |a, b| a.add(b),
        par!(|a, b| a.par_add(b)),
        |x, y| x + y,
    );
    out_of_place::<f64>(
        &mut group,
        |a, b| a.add(b),
        par!(|a, b| a.par_add(b)),
        |x, y| x + y,
    );
    group.finish();

    let mut group = c.benchmark_group("add_assign");
    in_place::<f32>(
        &mut group,
        |a, b| a.add_assign(b),
        par!(|a, b| a.par_add_assign(b)),
        |x, y| x + y,
    );
    in_place::<f64>(
        &mut group,
        |a, b| a.add_assign(b),
        par!(|a, b| a.par_add_assign(b)),
        |x, y| x + y,
    );
    group.finish();
}

fn sub(c: &mut Criterion) {
    let mut group = c.benchmark_group("sub");
    out_of_place::<f32>(
        &mut group,
        |a, b| a.sub(b),
        par!(|a, b| a.par_sub(b)),
        |x, y| x - y,
    );
    group.finish();

    let mut group = c.benchmark_group("sub_assign");
    in_place::<f32>(
        &mut group,
        |a, b| a.sub_assign(b),
        par!(|a, b| a.par_sub_assign(b)),
        |x, y| x - y,
    );
    group.finish();
}

fn mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");
    out_of_place::<f32>(
        &mut group,
        |a, b| a.mul(b),
        par!(|a, b| a.par_mul(b)),
        |x, y| x * y,
    );
    group.finish();

    let mut group = c.benchmark_group("mul_assign");
    in_place::<f32>(
        &mut group,
        |a, b| a.mul_assign(b),
        par!(|a, b| a.par_mul_assign(b)),
        |x, y| x * y,
    );
    group.finish();
}

fn div(c: &mut Criterion) {
    let mut group = c.benchmark_group("div");
    out_of_place::<f32>(
        &mut group,
        |a, b| a.div(b),
        par!(|a, b| a.par_div(b)),
        |x, y| x / y,
    );
    group.finish();

    let mut group = c.benchmark_group("div_assign");
    in_place::<f32>(
        &mut group,
        |a, b| a.div_assign(b),
        par!(|a, b| a.par_div_assign(b)),
        |x, y| x / y,
    );
    group.finish();
}

/// `sub_scalar` and `sub_scalar_assign` for both element types, and
/// `div_scalar` for `f32`: `add_scalar` and `mul_scalar` run the same loop as
/// `sub_scalar`.
fn scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("sub_scalar");
    with_scalar::<f32>(
        &mut group,
        |a, k| a.sub_scalar(k),
        par!(|a, k| a.par_sub_scalar(k)),
        |x, y| x - y,
    );
    with_scalar::<f64>(
        &mut group,
        |a, k| a.sub_scalar(k),
        par!(|a, k| a.par_sub_scalar(k)),
        |x, y| x - y,
    );
    group.finish();

    let mut group = c.benchmark_group("sub_scalar_assign");
    with_scalar_in_place::<f32>(
        &mut group,
        |a, k| a.sub_scalar_assign(k),
        par!(|a, k| a.par_sub_scalar_assign(k)),
        |x, y| x - y,
    );
    with_scalar_in_place::<f64>(
        &mut group,
        |a, k| a.sub_scalar_assign(k),
        par!(|a, k| a.par_sub_scalar_assign(k)),
        |x, y| x - y,
    );
    group.finish();

    let mut group = c.benchmark_group("div_scalar");
    with_scalar::<f32>(
        &mut group,
        |a, k| a.div_scalar(k),
        par!(|a, k| a.par_div_scalar(k)),
        |x, y| x / y,
    );
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
    targets = add, sub, mul, div, scalar
}
criterion_main!(benches);
