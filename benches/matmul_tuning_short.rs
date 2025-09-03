//! Simple BLIS Parameter Tuning Benchmark
//!
//! A focused benchmark for finding optimal MC, KC, NC cache blocking parameters.
//! Tests strategic parameter combinations rather than exhaustive search.
//!
//! # Usage:
//! ```bash
//! cargo bench --bench matmul_tuning_simple
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use simdly::simd::avx2::dot::matmul;

/// Create a random test matrix in column-major format
fn create_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows * cols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}

/// Benchmark strategic MC, KC, NC parameter combinations
fn bench_matmul_params(c: &mut Criterion) {
    // Test matrix size
    let (m, n, k) = (1024, 1024, 1024);

    // Create test matrices
    let a = create_matrix(m, k, 42);
    let b = create_matrix(k, n, 43);

    let mut group = c.benchmark_group("matmul_params");
    group.sample_size(10); // Fewer samples for faster benchmarking

    // Strategic parameter combinations based on cache hierarchy
    let params = [
        // Format: (MC, KC, NC, description)
        (64, 64, 128, "Small blocks"),
        (64, 64, 192, "Optimal from analysis"),
        (64, 64, 256, "Medium NC"),
        (64, 96, 192, "Larger KC"),
        (96, 64, 192, "Larger MC"),
        (128, 64, 256, "Balanced large"),
        (64, 128, 192, "Large KC"),
        (128, 128, 256, "All large"),
    ];

    for &(mc, kc, nc, desc) in &params {
        let mut c_result = vec![0.0; m * n];

        group.bench_with_input(
            BenchmarkId::new(
                "params",
                format!("MC{}_KC{}_NC{}_{}", mc, kc, nc, desc.replace(' ', "_")),
            ),
            &(mc, kc, nc),
            |bench, &(mc, kc, nc)| {
                bench.iter(|| {
                    c_result.fill(0.0); // Clear result matrix
                    matmul(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c_result),
                        black_box(m),
                        black_box(n),
                        black_box(k),
                        black_box(mc),
                        black_box(kc),
                        black_box(nc),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Quick performance comparison benchmark
fn bench_matmul_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_sizes");
    group.sample_size(10);

    // Test different matrix sizes with optimal parameters (MC=64, KC=64, NC=192)
    let sizes = [256, 512, 1024];

    for &size in &sizes {
        let a = create_matrix(size, size, 42);
        let b = create_matrix(size, size, 43);
        let mut c_result = vec![0.0; size * size];

        group.bench_with_input(
            BenchmarkId::new("size", format!("{}x{}", size, size)),
            &size,
            |bench, &size| {
                bench.iter(|| {
                    c_result.fill(0.0);
                    matmul(
                        black_box(&a),
                        black_box(&b),
                        black_box(&mut c_result),
                        black_box(size),
                        black_box(size),
                        black_box(size),
                        black_box(128), // MC
                        black_box(128), // KC
                        black_box(256), // NC
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_matmul_params, bench_matmul_sizes);
criterion_main!(benches);
