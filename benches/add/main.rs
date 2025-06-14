use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array1;
use simdly::simd::traits::{SimdAdd, SimdCos};

// --- Configuration ---
const VECTOR_LENGTHS: &[usize] = &[
    512,
    1024,
    1_048_576,     // Huge: Stress test for SIMD and parallelism
    1_073_741_824, // Mega: Tests the limits of SIMD and parallelism
];

fn generate_data(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = vec![1.0; len];
    let b: Vec<f32> = vec![1.0; len];

    (a, b)
}

fn bench_vector_addition(c: &mut Criterion) {
    // --- Setup that applies to all benchmarks ---
    for &vector_len in VECTOR_LENGTHS.iter() {
        // The group name already contains the vector length, providing context.
        let mut group = c.benchmark_group(format!("VectorAddition/{vector_len}"));

        // group.sample_size(100);

        // Set throughput so results are reported in terms of elements/sec.
        // group.throughput(Throughput::Elements(vector_len as u64));

        // Generate the data once per group.
        let (a_vec, b_vec) = generate_data(vector_len);

        // --- Benchmark 1: Scalar Addition ---
        group.bench_function("scalar add", |bencher| {
            bencher.iter(|| {
                // Black-box the output to prevent the entire operation
                // from being optimized away.
                black_box(a_vec.scalar_add(black_box(&b_vec)))
            });
        });

        // --- Benchmark 2: SIMD Addition (AVX2) ---
        group.bench_function("simd add (simdly)", |bencher| {
            bencher.iter(|| black_box(a_vec.simd_add(black_box(&b_vec))));
        });

        // --- Benchmark 3: Parallel SIMD Addition (AVX2) ---
        group.bench_function("parallel simd add (simdly)", |bencher| {
            bencher.iter(|| black_box(a_vec.par_simd_add(black_box(&b_vec))));
        });

        // --- Benchmark 4: ndarray ---
        // ndarray often uses SIMD internally if available, so it's a great baseline.
        // Setup for this specific benchmark can be done here.
        // Cloning is necessary because `from_vec` takes ownership.
        let a_arr = Array1::from_vec(a_vec.clone());
        let b_arr = Array1::from_vec(b_vec.clone());

        group.bench_function("ndarray", |bencher| {
            bencher.iter(|| black_box(&a_arr + &b_arr));
        });

        group.finish();
    }
}

fn bench_vector_cos(c: &mut Criterion) {
    // --- Setup that applies to all benchmarks ---
    for &vector_len in VECTOR_LENGTHS.iter() {
        // The group name already contains the vector length, providing context.
        let mut group = c.benchmark_group(format!("VectorAddition/{vector_len}"));

        // group.sample_size(100);

        // Set throughput so results are reported in terms of elements/sec.
        // group.throughput(Throughput::Elements(vector_len as u64));

        // Generate the data once per group.
        let (a_vec, _) = generate_data(vector_len);

        // --- Benchmark 1: Scalar Addition ---
        group.bench_function("scalar cos", |bencher| {
            bencher.iter(|| {
                // Black-box the output to prevent the entire operation
                // from being optimized away.
                black_box(a_vec.scalar_cos())
            });
        });

        // --- Benchmark 2: SIMD Addition (AVX2) ---
        group.bench_function("simd cos (simdly)", |bencher| {
            bencher.iter(|| black_box(a_vec.simd_cos()));
        });

        // --- Benchmark 3: Parallel SIMD Addition (AVX2) ---
        group.bench_function("parallel simd cos (simdly)", |bencher| {
            bencher.iter(|| black_box(a_vec.par_simd_cos()));
        });

        // --- Benchmark 4: ndarray ---
        // ndarray often uses SIMD internally if available, so it's a great baseline.
        // Setup for this specific benchmark can be done here.
        // Cloning is necessary because `from_vec` takes ownership.
        let a_arr = Array1::from_vec(a_vec.clone());

        group.bench_function("ndarray", |bencher| {
            bencher.iter(|| black_box(a_arr.cos()));
        });

        group.finish();
    }
}

criterion_group!(
    benches,
    // bench_vector_addition,
    bench_vector_cos
);

criterion_main!(benches);
