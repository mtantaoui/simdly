//! Benchmarks for AVX2 SIMD outer product computation.
//!
//! This benchmark suite compares the performance of the SIMD-optimized outer product
//! implementation against scalar alternatives, ndarray, and faer across various input sizes and patterns.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use faer::Mat;
use ndarray::Array1;
use simdly::simd::avx2::outer::outer;

/// Scalar reference implementation for performance comparison.
fn scalar_outer_product(vec_a: &[f32], vec_b: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(vec_a.len() * vec_b.len());
    for &a_val in vec_a {
        for &b_val in vec_b {
            result.push(a_val * b_val);
        }
    }
    result
}

/// ndarray outer product implementation.
fn ndarray_outer_product(vec_a: &[f32], vec_b: &[f32]) -> Vec<f32> {
    let a = Array1::from_vec(vec_a.to_vec());
    let b = Array1::from_vec(vec_b.to_vec());

    // Create outer product using broadcasting
    let a_reshaped = a.view().insert_axis(ndarray::Axis(1));
    let b_reshaped = b.view().insert_axis(ndarray::Axis(0));
    let result = &a_reshaped * &b_reshaped;

    result.into_raw_vec()
}

/// faer outer product implementation.
fn faer_outer_product(vec_a: &[f32], vec_b: &[f32]) -> Vec<f32> {
    let a_mat = Mat::from_fn(vec_a.len(), 1, |i, _| vec_a[i]);
    let b_mat = Mat::from_fn(1, vec_b.len(), |_, j| vec_b[j]);

    let result = &a_mat * &b_mat;

    // Convert result to Vec<f32>
    let mut output = Vec::with_capacity(vec_a.len() * vec_b.len());
    for i in 0..result.nrows() {
        for j in 0..result.ncols() {
            output.push(result[(i, j)]);
        }
    }
    output
}

/// Benchmark scaling behavior: how performance scales with vector size.
fn bench_scaling_behavior(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_behavior");

    // Square matrices of increasing size
    let sizes = [8, 16, 32, 64, 128, 256, 512, 1024];

    for size in sizes {
        let a: Vec<f32> = (1..=size).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (1..=size).map(|i| i as f32 * 0.02).collect();

        let elements = size * size;
        group.throughput(Throughput::Elements(elements as u64));

        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result = outer(black_box(a), black_box(b));
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result = scalar_outer_product(black_box(a), black_box(b));
                    black_box(result)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray", size),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result = ndarray_outer_product(black_box(a), black_box(b));
                    black_box(result)
                })
            },
        );

        // group.bench_with_input(
        //     BenchmarkId::new("faer", size),
        //     &(&a, &b),
        //     |bench, (a, b)| {
        //         bench.iter(|| {
        //             let result = faer_outer_product(black_box(a), black_box(b));
        //             black_box(result)
        //         })
        //     },
        // );
    }

    group.finish();
}

criterion_group!(benches, bench_scaling_behavior,);

criterion_main!(benches);
