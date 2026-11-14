//! `SliceMath` functions against the plain scalar loop, from L1-resident to
//! larger-than-cache slices. With `--features rayon`, the same groups also
//! time `ParSliceMath` (`par`) and the rayon loop with `std` (`rayon`), at
//! every length (`par_` runs on one thread up to `PARALLEL_SIMD_THRESHOLD`).
//!
//! ```sh
//! cargo bench --bench math
//! cargo bench --bench math --features rayon
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "rayon")]
use simdly::ParSliceMath;
use simdly::SliceMath;

const LENS: &[usize] = &[64, 4_096, 262_144, 4_194_304];

type Group<'a> = BenchmarkGroup<'a, WallTime>;

trait Float: Copy + Send + Sync {
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

fn id<T: Float>(name: &str, len: usize) -> BenchmarkId {
    BenchmarkId::new(format!("{name}/{}", T::NAME), len)
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

/// `[−5, 5]`: where `tanh` has not yet rounded to `±1`, as activations
/// mostly are.
fn activation<T: Float>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from_f64((i as f64 * 0.37).sin() * 5.0))
        .collect()
}

/// `[1, 81]`: `acosh`'s domain.
fn above_one<T: Float>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from_f64(1.0 + (i as f64 * 0.37).sin().abs() * 80.0))
        .collect()
}

type Input<T> = fn(usize) -> Vec<T>;
type Unary<T> = fn(&[T]) -> Vec<T>;
type Binary<T> = fn(&[T], &[T]) -> Vec<T>;

/// The slice traits `hypot3` and `hypot4` need: `SliceMath`, and
/// `ParSliceMath` with `rayon`.
#[cfg(feature = "rayon")]
trait Slice<T>: SliceMath<T> + ParSliceMath<T> {}
#[cfg(feature = "rayon")]
impl<T, S: SliceMath<T> + ParSliceMath<T> + ?Sized> Slice<T> for S {}
#[cfg(not(feature = "rayon"))]
trait Slice<T>: SliceMath<T> {}
#[cfg(not(feature = "rayon"))]
impl<T, S: SliceMath<T> + ?Sized> Slice<T> for S {}

/// `simdly(x)` vs `x.iter().map(f).collect()`; with `rayon`, `par(x)` vs
/// `x.par_iter().map(f).collect()` too.
fn out_of_place<T: Float>(
    group: &mut Group,
    input: Input<T>,
    simdly: impl Fn(&[T]) -> Vec<T>,
    par: Option<Unary<T>>,
    f: impl Fn(T) -> T + Sync,
) {
    for &len in LENS {
        let x = input(len);
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(id::<T>("simdly", len), |bench| {
            bench.iter(|| simdly(black_box(&x)))
        });
        group.bench_function(id::<T>("scalar", len), |bench| {
            bench.iter(|| black_box(&x).iter().map(|&v| f(v)).collect::<Vec<T>>())
        });
        #[cfg(feature = "rayon")]
        if let Some(par) = par {
            group.bench_function(id::<T>("par", len), |bench| {
                bench.iter(|| par(black_box(&x)))
            });
            group.bench_function(id::<T>("rayon", len), |bench| {
                bench.iter(|| black_box(&x).par_iter().map(|&v| f(v)).collect::<Vec<T>>())
            });
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = par;
}

/// Times `run` on a fresh copy of `x` each iteration (the copy is not
/// timed): a pass changes the values, and every implementation must see the
/// same ones.
fn bench_in_place<T: Float>(group: &mut Group, id: BenchmarkId, x: &[T], run: impl Fn(&mut [T])) {
    let size = if x.len() <= 4_096 {
        BatchSize::SmallInput
    } else {
        BatchSize::LargeInput
    };
    group.bench_function(id, |bench| {
        bench.iter_batched_ref(|| x.to_vec(), |x| run(black_box(x)), size)
    });
}

/// `simdly(x)` vs `for v in x { *v = f(*v) }`; with `rayon`, `par(x)` vs
/// `x.par_iter_mut().for_each(..)` too.
fn in_place<T: Float>(
    group: &mut Group,
    input: Input<T>,
    simdly: impl Fn(&mut [T]),
    par: Option<fn(&mut [T])>,
    f: impl Fn(T) -> T + Sync,
) {
    for &len in LENS {
        let x = input(len);
        group.throughput(Throughput::Elements(len as u64));
        bench_in_place(group, id::<T>("simdly", len), &x, &simdly);
        bench_in_place(group, id::<T>("scalar", len), &x, |x| {
            for v in x.iter_mut() {
                *v = f(*v);
            }
        });
        #[cfg(feature = "rayon")]
        if let Some(par) = par {
            bench_in_place(group, id::<T>("par", len), &x, par);
            bench_in_place(group, id::<T>("rayon", len), &x, |x| {
                x.par_iter_mut().for_each(|v| *v = f(*v))
            });
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = par;
}

fn abs(c: &mut Criterion) {
    let mut group = c.benchmark_group("abs");
    out_of_place::<f32>(
        &mut group,
        signed,
        |x| x.abs(),
        par!(|x| x.par_abs()),
        f32::abs,
    );
    out_of_place::<f64>(
        &mut group,
        signed,
        |x| x.abs(),
        par!(|x| x.par_abs()),
        f64::abs,
    );
    group.finish();

    let mut group = c.benchmark_group("abs_in_place");
    in_place::<f32>(
        &mut group,
        signed,
        |x| x.abs_in_place(),
        par!(|x| x.par_abs_in_place()),
        f32::abs,
    );
    in_place::<f64>(
        &mut group,
        signed,
        |x| x.abs_in_place(),
        par!(|x| x.par_abs_in_place()),
        f64::abs,
    );
    group.finish();
}

fn acos(c: &mut Criterion) {
    let mut group = c.benchmark_group("acos");
    out_of_place::<f32>(
        &mut group,
        unit,
        |x| x.acos(),
        par!(|x| x.par_acos()),
        f32::acos,
    );
    out_of_place::<f64>(
        &mut group,
        unit,
        |x| x.acos(),
        par!(|x| x.par_acos()),
        f64::acos,
    );
    group.finish();
}

fn asin(c: &mut Criterion) {
    let mut group = c.benchmark_group("asin");
    out_of_place::<f32>(
        &mut group,
        unit,
        |x| x.asin(),
        par!(|x| x.par_asin()),
        f32::asin,
    );
    out_of_place::<f64>(
        &mut group,
        unit,
        |x| x.asin(),
        par!(|x| x.par_asin()),
        f64::asin,
    );
    group.finish();
}

fn atan(c: &mut Criterion) {
    let mut group = c.benchmark_group("atan");
    out_of_place::<f32>(
        &mut group,
        signed,
        |x| x.atan(),
        par!(|x| x.par_atan()),
        f32::atan,
    );
    out_of_place::<f64>(
        &mut group,
        signed,
        |x| x.atan(),
        par!(|x| x.par_atan()),
        f64::atan,
    );
    group.finish();

    let mut group = c.benchmark_group("atan_in_place");
    in_place::<f32>(
        &mut group,
        signed,
        |x| x.atan_in_place(),
        par!(|x| x.par_atan_in_place()),
        f32::atan,
    );
    in_place::<f64>(
        &mut group,
        signed,
        |x| x.atan_in_place(),
        par!(|x| x.par_atan_in_place()),
        f64::atan,
    );
    group.finish();
}

/// `simdly(a, b)` vs `a.iter().zip(b).map(|(a, b)| f(a, b))`, over all
/// four quadrants; with `rayon`, `par(a, b)` vs the same with `par_iter`.
fn binary_vs_scalar<T: Float>(
    group: &mut Group,
    simdly: impl Fn(&[T], &[T]) -> Vec<T>,
    par: Option<Binary<T>>,
    f: impl Fn(T, T) -> T + Sync,
) {
    binary_inputs_vs_scalar(group, signed, wave, simdly, par, f);
}

/// `cos(0.73·i)·30`: the second operand of the binary benchmarks.
fn wave<T: Float>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from_f64((i as f64 * 0.73).cos() * 30.0))
        .collect()
}

/// `cos(0.73·i)·6`: exponents for `powf`.
fn exponents<T: Float>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from_f64((i as f64 * 0.73).cos() * 6.0))
        .collect()
}

/// [`binary_vs_scalar`] on inputs from `first` and `second`.
fn binary_inputs_vs_scalar<T: Float>(
    group: &mut Group,
    first: Input<T>,
    second: Input<T>,
    simdly: impl Fn(&[T], &[T]) -> Vec<T>,
    par: Option<Binary<T>>,
    f: impl Fn(T, T) -> T + Sync,
) {
    for &len in LENS {
        let (y, x) = (first(len), second(len));
        group.throughput(Throughput::Elements(len as u64));
        group.bench_function(id::<T>("simdly", len), |bench| {
            bench.iter(|| simdly(black_box(&y), black_box(&x)))
        });
        group.bench_function(id::<T>("scalar", len), |bench| {
            bench.iter(|| {
                black_box(&y)
                    .iter()
                    .zip(black_box(&x))
                    .map(|(&a, &b)| f(a, b))
                    .collect::<Vec<T>>()
            })
        });
        #[cfg(feature = "rayon")]
        if let Some(par) = par {
            group.bench_function(id::<T>("par", len), |bench| {
                bench.iter(|| par(black_box(&y), black_box(&x)))
            });
            group.bench_function(id::<T>("rayon", len), |bench| {
                bench.iter(|| {
                    black_box(&y)
                        .par_iter()
                        .zip(black_box(&x))
                        .map(|(&a, &b)| f(a, b))
                        .collect::<Vec<T>>()
                })
            });
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = par;
}

type Nary<T, const N: usize> = fn(&[Vec<T>; N]) -> Vec<T>;

/// `simdly(v)` vs `scalar(v)` on `N` slices, mixed signs and magnitudes;
/// with `rayon`, `par[0](v)` vs `par[1](v)` (the rayon loop) too.
fn nary_vs_scalar<T: Float, const N: usize>(
    group: &mut Group,
    simdly: impl Fn(&[Vec<T>; N]) -> Vec<T>,
    scalar: impl Fn(&[Vec<T>; N]) -> Vec<T>,
    par: Option<[Nary<T, N>; 2]>,
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
        group.bench_function(id::<T>("simdly", len), |bench| {
            bench.iter(|| simdly(black_box(&v)))
        });
        group.bench_function(id::<T>("scalar", len), |bench| {
            bench.iter(|| scalar(black_box(&v)))
        });
        #[cfg(feature = "rayon")]
        if let Some([par, rayon]) = par {
            group.bench_function(id::<T>("par", len), |bench| {
                bench.iter(|| par(black_box(&v)))
            });
            group.bench_function(id::<T>("rayon", len), |bench| {
                bench.iter(|| rayon(black_box(&v)))
            });
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = par;
}

fn atan2(c: &mut Criterion) {
    let mut group = c.benchmark_group("atan2");
    binary_vs_scalar::<f32>(
        &mut group,
        |y, x| y.atan2(x),
        par!(|y, x| y.par_atan2(x)),
        f32::atan2,
    );
    binary_vs_scalar::<f64>(
        &mut group,
        |y, x| y.atan2(x),
        par!(|y, x| y.par_atan2(x)),
        f64::atan2,
    );
    group.finish();
}

fn cbrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("cbrt");
    out_of_place::<f32>(
        &mut group,
        signed,
        |x| x.cbrt(),
        par!(|x| x.par_cbrt()),
        f32::cbrt,
    );
    out_of_place::<f64>(
        &mut group,
        signed,
        |x| x.cbrt(),
        par!(|x| x.par_cbrt()),
        f64::cbrt,
    );
    group.finish();

    let mut group = c.benchmark_group("cbrt_in_place");
    in_place::<f32>(
        &mut group,
        signed,
        |x| x.cbrt_in_place(),
        par!(|x| x.par_cbrt_in_place()),
        f32::cbrt,
    );
    in_place::<f64>(
        &mut group,
        signed,
        |x| x.cbrt_in_place(),
        par!(|x| x.par_cbrt_in_place()),
        f64::cbrt,
    );
    group.finish();
}

fn ceil(c: &mut Criterion) {
    let mut group = c.benchmark_group("ceil");
    out_of_place::<f32>(
        &mut group,
        signed,
        |x| x.ceil(),
        par!(|x| x.par_ceil()),
        f32::ceil,
    );
    out_of_place::<f64>(
        &mut group,
        signed,
        |x| x.ceil(),
        par!(|x| x.par_ceil()),
        f64::ceil,
    );
    group.finish();

    let mut group = c.benchmark_group("ceil_in_place");
    in_place::<f32>(
        &mut group,
        signed,
        |x| x.ceil_in_place(),
        par!(|x| x.par_ceil_in_place()),
        f32::ceil,
    );
    in_place::<f64>(
        &mut group,
        signed,
        |x| x.ceil_in_place(),
        par!(|x| x.par_ceil_in_place()),
        f64::ceil,
    );
    group.finish();
}

fn exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.exp(),
        par!(|x| x.par_exp()),
        f32::exp,
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.exp(),
        par!(|x| x.par_exp()),
        f64::exp,
    );
    group.finish();
}

/// Against `std`'s, on `[−1, 1]`: both branches of the reduction.
fn log1p(c: &mut Criterion) {
    let mut group = c.benchmark_group("log1p");
    out_of_place::<f32>(
        &mut group,
        unit,
        |x| x.log1p(),
        par!(|x| x.par_log1p()),
        f32::ln_1p,
    );
    out_of_place::<f64>(
        &mut group,
        unit,
        |x| x.log1p(),
        par!(|x| x.par_log1p()),
        f64::ln_1p,
    );
    group.finish();
}

/// Against the formula as a user would write it, `1/(1 + e⁻ˣ)`.
fn sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.sigmoid(),
        par!(|x| x.par_sigmoid()),
        |x| 1.0 / (1.0 + (-x).exp()),
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.sigmoid(),
        par!(|x| x.par_sigmoid()),
        |x| 1.0 / (1.0 + (-x).exp()),
    );
    group.finish();
}

fn log_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_sigmoid");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.log_sigmoid(),
        par!(|x| x.par_log_sigmoid()),
        |x| x.min(0.0) - (-x.abs()).exp().ln_1p(),
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.log_sigmoid(),
        par!(|x| x.par_log_sigmoid()),
        |x| x.min(0.0) - (-x.abs()).exp().ln_1p(),
    );
    group.finish();
}

fn softplus(c: &mut Criterion) {
    let mut group = c.benchmark_group("softplus");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.softplus(),
        par!(|x| x.par_softplus()),
        |x| x.max(0.0) + (-x.abs()).exp().ln_1p(),
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.softplus(),
        par!(|x| x.par_softplus()),
        |x| x.max(0.0) + (-x.abs()).exp().ln_1p(),
    );
    group.finish();
}

fn sinh(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinh");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.sinh(),
        par!(|x| x.par_sinh()),
        f32::sinh,
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.sinh(),
        par!(|x| x.par_sinh()),
        f64::sinh,
    );
    group.finish();
}

fn cosh(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosh");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.cosh(),
        par!(|x| x.par_cosh()),
        f32::cosh,
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.cosh(),
        par!(|x| x.par_cosh()),
        f64::cosh,
    );
    group.finish();
}

fn atanh(c: &mut Criterion) {
    let mut group = c.benchmark_group("atanh");
    out_of_place::<f32>(
        &mut group,
        unit,
        |x| x.atanh(),
        par!(|x| x.par_atanh()),
        f32::atanh,
    );
    out_of_place::<f64>(
        &mut group,
        unit,
        |x| x.atanh(),
        par!(|x| x.par_atanh()),
        f64::atanh,
    );
    group.finish();
}

fn asinh(c: &mut Criterion) {
    let mut group = c.benchmark_group("asinh");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.asinh(),
        par!(|x| x.par_asinh()),
        f32::asinh,
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.asinh(),
        par!(|x| x.par_asinh()),
        f64::asinh,
    );
    group.finish();
}

fn acosh(c: &mut Criterion) {
    let mut group = c.benchmark_group("acosh");
    out_of_place::<f32>(
        &mut group,
        above_one,
        |x| x.acosh(),
        par!(|x| x.par_acosh()),
        f32::acosh,
    );
    out_of_place::<f64>(
        &mut group,
        above_one,
        |x| x.acosh(),
        par!(|x| x.par_acosh()),
        f64::acosh,
    );
    group.finish();
}

fn tanh(c: &mut Criterion) {
    let mut group = c.benchmark_group("tanh");
    out_of_place::<f32>(
        &mut group,
        activation,
        |x| x.tanh(),
        par!(|x| x.par_tanh()),
        f32::tanh,
    );
    out_of_place::<f64>(
        &mut group,
        activation,
        |x| x.tanh(),
        par!(|x| x.par_tanh()),
        f64::tanh,
    );
    group.finish();
}

fn erf(c: &mut Criterion) {
    let mut group = c.benchmark_group("erf");
    out_of_place::<f32>(
        &mut group,
        activation,
        |x| x.erf(),
        par!(|x| x.par_erf()),
        libm::erff,
    );
    out_of_place::<f64>(
        &mut group,
        activation,
        |x| x.erf(),
        par!(|x| x.par_erf()),
        libm::erf,
    );
    group.finish();
}

fn powf(c: &mut Criterion) {
    let mut group = c.benchmark_group("powf");
    binary_inputs_vs_scalar::<f32>(
        &mut group,
        above_one,
        exponents,
        |x, y| x.powf(y),
        par!(|x, y| x.par_powf(y)),
        f32::powf,
    );
    binary_inputs_vs_scalar::<f64>(
        &mut group,
        above_one,
        exponents,
        |x, y| x.powf(y),
        par!(|x, y| x.par_powf(y)),
        f64::powf,
    );
    group.finish();
}

fn powf_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("powf_scalar");
    out_of_place::<f32>(
        &mut group,
        above_one,
        |x| x.powf_scalar(2.5),
        par!(|x| x.par_powf_scalar(2.5)),
        |v| v.powf(2.5),
    );
    out_of_place::<f64>(
        &mut group,
        above_one,
        |x| x.powf_scalar(2.5),
        par!(|x| x.par_powf_scalar(2.5)),
        |v| v.powf(2.5),
    );
    group.finish();
}

fn powi(c: &mut Criterion) {
    let mut group = c.benchmark_group("powi");
    out_of_place::<f32>(
        &mut group,
        activation,
        |x| x.powi(3),
        par!(|x| x.par_powi(3)),
        |v| v.powi(3),
    );
    out_of_place::<f64>(
        &mut group,
        activation,
        |x| x.powi(3),
        par!(|x| x.par_powi(3)),
        |v| v.powi(3),
    );
    group.finish();
}

fn trunc(c: &mut Criterion) {
    let mut group = c.benchmark_group("trunc");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.trunc(),
        par!(|x| x.par_trunc()),
        f32::trunc,
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.trunc(),
        par!(|x| x.par_trunc()),
        f64::trunc,
    );
    group.finish();
}

fn round(c: &mut Criterion) {
    let mut group = c.benchmark_group("round");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.round(),
        par!(|x| x.par_round()),
        f32::round,
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.round(),
        par!(|x| x.par_round()),
        f64::round,
    );
    group.finish();
}

fn round_ties_even(c: &mut Criterion) {
    let mut group = c.benchmark_group("round_ties_even");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.round_ties_even(),
        par!(|x| x.par_round_ties_even()),
        f32::round_ties_even,
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.round_ties_even(),
        par!(|x| x.par_round_ties_even()),
        f64::round_ties_even,
    );
    group.finish();
}

fn neg(c: &mut Criterion) {
    let mut group = c.benchmark_group("neg");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.neg(),
        par!(|x| x.par_neg()),
        |v| -v,
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.neg(),
        par!(|x| x.par_neg()),
        |v| -v,
    );
    group.finish();
}

fn recip(c: &mut Criterion) {
    let mut group = c.benchmark_group("recip");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.recip(),
        par!(|x| x.par_recip()),
        |v| 1.0 / v,
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.recip(),
        par!(|x| x.par_recip()),
        |v| 1.0 / v,
    );
    group.finish();
}

fn sign(c: &mut Criterion) {
    let mut group = c.benchmark_group("sign");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.sign(),
        par!(|x| x.par_sign()),
        |v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.sign(),
        par!(|x| x.par_sign()),
        |v| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        },
    );
    group.finish();
}

fn clamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("clamp");
    out_of_place::<f32>(
        &mut group,
        moderate,
        |x| x.clamp(-1.0, 1.0),
        par!(|x| x.par_clamp(-1.0, 1.0)),
        |v| v.clamp(-1.0, 1.0),
    );
    out_of_place::<f64>(
        &mut group,
        moderate,
        |x| x.clamp(-1.0, 1.0),
        par!(|x| x.par_clamp(-1.0, 1.0)),
        |v| v.clamp(-1.0, 1.0),
    );
    group.finish();
}

fn floor(c: &mut Criterion) {
    let mut group = c.benchmark_group("floor");
    out_of_place::<f32>(
        &mut group,
        signed,
        |x| x.floor(),
        par!(|x| x.par_floor()),
        f32::floor,
    );
    out_of_place::<f64>(
        &mut group,
        signed,
        |x| x.floor(),
        par!(|x| x.par_floor()),
        f64::floor,
    );
    group.finish();

    let mut group = c.benchmark_group("floor_in_place");
    in_place::<f32>(
        &mut group,
        signed,
        |x| x.floor_in_place(),
        par!(|x| x.par_floor_in_place()),
        f32::floor,
    );
    in_place::<f64>(
        &mut group,
        signed,
        |x| x.floor_in_place(),
        par!(|x| x.par_floor_in_place()),
        f64::floor,
    );
    group.finish();
}

fn ln(c: &mut Criterion) {
    let mut group = c.benchmark_group("ln");
    out_of_place::<f32>(
        &mut group,
        positive,
        |x| x.ln(),
        par!(|x| x.par_ln()),
        f32::ln,
    );
    out_of_place::<f64>(
        &mut group,
        positive,
        |x| x.ln(),
        par!(|x| x.par_ln()),
        f64::ln,
    );
    group.finish();
}

fn hypot(c: &mut Criterion) {
    let mut group = c.benchmark_group("hypot");
    binary_vs_scalar::<f32>(
        &mut group,
        |x, y| x.hypot(y),
        par!(|x, y| x.par_hypot(y)),
        f32::hypot,
    );
    binary_vs_scalar::<f64>(
        &mut group,
        |x, y| x.hypot(y),
        par!(|x, y| x.par_hypot(y)),
        f64::hypot,
    );
    group.finish();
}

/// Against `std`'s overflow-safe way: nested `hypot`s.
fn hypot3(c: &mut Criterion) {
    fn bench<T: Float>(group: &mut Group)
    where
        [T]: Slice<T>,
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
            par!([
                |[x, y, z]| x.par_hypot3(y, z),
                |[x, y, z]| {
                    (x, y, z)
                        .into_par_iter()
                        .map(|(&a, &b, &c)| T::hypot(T::hypot(a, b), c))
                        .collect()
                },
            ]),
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
        [T]: Slice<T>,
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
            par!([
                |[x, y, z, w]| x.par_hypot4(y, z, w),
                |[x, y, z, w]| {
                    (x, y, z, w)
                        .into_par_iter()
                        .map(|(&a, &b, &c, &d)| T::hypot(T::hypot(T::hypot(a, b), c), d))
                        .collect()
                },
            ]),
        );
    }
    let mut group = c.benchmark_group("hypot4");
    bench::<f32>(&mut group);
    bench::<f64>(&mut group);
    group.finish();
}

fn cos(c: &mut Criterion) {
    let mut group = c.benchmark_group("cos");
    out_of_place::<f32>(
        &mut group,
        signed,
        |x| x.cos(),
        par!(|x| x.par_cos()),
        f32::cos,
    );
    out_of_place::<f64>(
        &mut group,
        signed,
        |x| x.cos(),
        par!(|x| x.par_cos()),
        f64::cos,
    );
    group.finish();

    let mut group = c.benchmark_group("cos_in_place");
    in_place::<f32>(
        &mut group,
        signed,
        |x| x.cos_in_place(),
        par!(|x| x.par_cos_in_place()),
        f32::cos,
    );
    in_place::<f64>(
        &mut group,
        signed,
        |x| x.cos_in_place(),
        par!(|x| x.par_cos_in_place()),
        f64::cos,
    );
    group.finish();
}

fn tan(c: &mut Criterion) {
    let mut group = c.benchmark_group("tan");
    out_of_place::<f32>(
        &mut group,
        signed,
        |x| x.tan(),
        par!(|x| x.par_tan()),
        f32::tan,
    );
    out_of_place::<f64>(
        &mut group,
        signed,
        |x| x.tan(),
        par!(|x| x.par_tan()),
        f64::tan,
    );
    group.finish();

    let mut group = c.benchmark_group("tan_in_place");
    in_place::<f32>(
        &mut group,
        signed,
        |x| x.tan_in_place(),
        par!(|x| x.par_tan_in_place()),
        f32::tan,
    );
    in_place::<f64>(
        &mut group,
        signed,
        |x| x.tan_in_place(),
        par!(|x| x.par_tan_in_place()),
        f64::tan,
    );
    group.finish();
}

fn sin(c: &mut Criterion) {
    let mut group = c.benchmark_group("sin");
    out_of_place::<f32>(
        &mut group,
        signed,
        |x| x.sin(),
        par!(|x| x.par_sin()),
        f32::sin,
    );
    out_of_place::<f64>(
        &mut group,
        signed,
        |x| x.sin(),
        par!(|x| x.par_sin()),
        f64::sin,
    );
    group.finish();

    let mut group = c.benchmark_group("sin_in_place");
    in_place::<f32>(
        &mut group,
        signed,
        |x| x.sin_in_place(),
        par!(|x| x.par_sin_in_place()),
        f32::sin,
    );
    in_place::<f64>(
        &mut group,
        signed,
        |x| x.sin_in_place(),
        par!(|x| x.par_sin_in_place()),
        f64::sin,
    );
    group.finish();
}

fn sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqrt");
    out_of_place::<f32>(
        &mut group,
        positive,
        |x| x.sqrt(),
        par!(|x| x.par_sqrt()),
        f32::sqrt,
    );
    out_of_place::<f64>(
        &mut group,
        positive,
        |x| x.sqrt(),
        par!(|x| x.par_sqrt()),
        f64::sqrt,
    );
    group.finish();

    let mut group = c.benchmark_group("sqrt_in_place");
    in_place::<f32>(
        &mut group,
        positive,
        |x| x.sqrt_in_place(),
        par!(|x| x.par_sqrt_in_place()),
        f32::sqrt,
    );
    in_place::<f64>(
        &mut group,
        positive,
        |x| x.sqrt_in_place(),
        par!(|x| x.par_sqrt_in_place()),
        f64::sqrt,
    );
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
    targets = abs, acos, acosh, asin, asinh, atan, atan2, atanh, cbrt, ceil, clamp, cos, cosh, erf, exp, floor, neg, recip, round, round_ties_even, sign, trunc, hypot, hypot3, hypot4, ln, log1p, log_sigmoid, powf, powf_scalar, powi, sigmoid, sin, sinh, softplus, sqrt, tan, tanh
}
criterion_main!(benches);
