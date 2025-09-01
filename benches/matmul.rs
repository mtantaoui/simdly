//! Matrix Multiplication Benchmark Comparison
//!
//! Compares BLIS implementation vs ndarray vs faer across different matrix sizes.
//!
//! # Usage:
//! ```bash
//! # Run all matrix multiplication benchmarks
//! cargo bench --bench matmul
//!
//! # Run specific benchmark group
//! cargo bench --bench matmul -- matmul_blis
//! cargo bench --bench matmul -- matmul_comparison
//!
//! # Run with release optimizations (recommended)
//! cargo bench --release --bench matmul
//!
//! # Save results to file
//! cargo bench --bench matmul > matmul_results.txt
//! ```

use std::cmp::min;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use faer::Mat;
use ndarray::Array2;
use rand::prelude::*;
use simdly::{simd::avx2::dot::matmul, utils::alloc_zeroed_vec};

// Import our implementations

/// Create a test matrix in column-major format for our BLIS implementation
fn create_blis_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut matrix = alloc_zeroed_vec(rows * cols, 32);
    for col in 0..cols {
        for row in 0..rows {
            let idx = col * rows + row; // column-major indexing
            matrix[idx] = rng.random_range(-1.0..1.0);
        }
    }
    matrix
}

/// Create test matrix for ndarray (row-major by default)
fn create_ndarray_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |_| rng.random_range(-1.0..1.0))
}

/// Create test matrix for faer
fn create_faer_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Mat<f32> {
    Mat::from_fn(rows, cols, |_, _| rng.random_range(-1.0..1.0))
}

/// Benchmark all implementations across different matrix sizes for comparison
fn bench_matmul_by_size(c: &mut Criterion) {
    let sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        // (1000, 1000, 1000),
        (3000, 3000, 3000),
        (8000, 8000, 8000),
        // (100, 150, 200), // Non-square
        // (512, 256, 128), // Different aspect ratios
    ];

    // let mc = 64;
    // let nc = 448;

    let mc = 192;
    let nc = 896;
    for (m, k, n) in sizes {
        // Cache blocking parameters optimized for AVX2 performance
        // mc: M-dimension blocking for L2 cache (typically 64-128)
        // nc: N-dimension blocking for L3 cache (typically 256-512)
        // let mc = min(m, 192); // L2 cache blocking - rows of A and C
        // let nc = min(n, 896); // L3 cache blocking - columns of B and C

        let group_name = format!("matmul_{}x{}x{}", m, k, n);
        let mut group = c.benchmark_group(&group_name);
        group.sample_size(20); // Reduce sample size for large matrices

        let mut rng = StdRng::seed_from_u64(42);

        // Prepare data for each implementation
        let a_blis = create_blis_matrix(m, k, &mut rng);
        let b_blis = create_blis_matrix(k, n, &mut rng);
        let mut c_blis = alloc_zeroed_vec(m * n, 32);

        rng = StdRng::seed_from_u64(42); // Reset RNG for consistency
        let a_ndarray = create_ndarray_matrix(m, k, &mut rng);
        let b_ndarray = create_ndarray_matrix(k, n, &mut rng);

        rng = StdRng::seed_from_u64(42); // Reset RNG for consistency
        let a_faer = create_faer_matrix(m, k, &mut rng);
        let b_faer = create_faer_matrix(k, n, &mut rng);
        let mut c_faer = Mat::zeros(m, n);

        c_blis.fill(0.0);

        // Benchmark Original Simdly
        group.bench_function("Simdly_Packing", |bench| {
            bench.iter(|| {
                matmul(
                    black_box(&a_blis),
                    black_box(&b_blis),
                    black_box(&mut c_blis),
                    black_box(m),
                    black_box(n),
                    black_box(k),
                    black_box(mc),
                    black_box(nc),
                );
                black_box(&c_blis);
            });
        });

        // Benchmark ndarray
        group.bench_function("ndarray", |bench| {
            bench.iter(|| {
                let result = black_box(&a_ndarray).dot(black_box(&b_ndarray));
                black_box(result);
            });
        });

        // Benchmark faer
        c_faer.fill_zero();
        group.bench_function("faer", |bench| {
            bench.iter(|| {
                faer::linalg::matmul::matmul(
                    c_faer.as_mut(),
                    black_box(a_faer.as_ref()),
                    black_box(b_faer.as_ref()),
                    None,
                    1.0,
                    faer::Parallelism::Rayon(0),
                );
                black_box(&c_faer);
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_matmul_by_size);
criterion_main!(benches);
