//! BLIS Parameter Tuning Benchmark Suite
//!
//! Criterion-based benchmarks for finding optimal BLIS cache blocking parameters.
//! Organizes benchmarks by matrix size to compare MC×NC parameter combinations
//! within each size group, similar to examples/matmul.rs organization.
//!
//! # Usage:
//! ```bash
//! # Run all parameter tuning benchmarks (grouped by matrix size)
//! cargo bench --bench matmul_tuning
//!
//! # Run specific matrix size groups
//! cargo bench --bench matmul_tuning -- "64x64x64"     # Small matrices
//! cargo bench --bench matmul_tuning -- "128x128x128"  # Medium matrices  
//! cargo bench --bench matmul_tuning -- "256x256x256"  # Large matrices
//! cargo bench --bench matmul_tuning -- "512x512x512"  # Very large matrices
//!
//! # Run with release optimizations (required for accurate results)
//! cargo bench --release --bench matmul_tuning
//!
//! # Generate HTML report with performance comparison graphs
//! cargo bench --bench matmul_tuning && open target/criterion/report/index.html
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use simdly::simd::avx2::dot::matmul;

/// Configuration for parameter tuning benchmarks
#[derive(Debug, Clone)]
struct TuningConfig {
    mc_values: Vec<usize>,
    nc_values: Vec<usize>,
}

impl TuningConfig {
    fn comprehensive() -> Self {
        Self {
            // Match examples/matmul.rs comprehensive configuration
            mc_values: (64..=1024).step_by(64).collect(), // [64, 128, 192, ..., 1024]
            nc_values: (64..=1024).step_by(64).collect(), // [64, 128, 192, ..., 1024]
        }
    }
}

/// Create a test matrix in column-major format
fn create_test_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut matrix = vec![0.0; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            let idx = col * rows + row; // column-major indexing
            matrix[idx] = rng.random_range(-1.0..1.0);
        }
    }
    matrix
}

/// Benchmark 512x512x512 matrices with different MC×NC combinations (Large matrices)  
fn bench_matrix(c: &mut Criterion) {
    let config = TuningConfig::comprehensive();
    let (m, n, k) = (2048, 2048, 2048);

    let mut group = c.benchmark_group(format!("{}x{}x{}", m, n, k));
    // Let criterion handle sample size automatically

    // Create test matrices once
    let mut rng = StdRng::seed_from_u64(42);
    let a = create_test_matrix(m, k, &mut rng);
    let b = create_test_matrix(k, n, &mut rng);
    let mut c = vec![0.0; m * n];

    // Test comprehensive MC×NC combinations for large matrices
    for mc in &config.mc_values {
        for nc in &config.nc_values {
            let param_id = format!("MC{}_NC{}", mc, nc);

            c.fill(0.0);

            group.bench_with_input(
                BenchmarkId::new("params", &param_id),
                &(mc, nc),
                |bench, &(mc, nc)| {
                    bench.iter(|| {
                        matmul(
                            black_box(&a),
                            black_box(&b),
                            black_box(&mut c),
                            black_box(m),
                            black_box(n),
                            black_box(k),
                            black_box(*mc),
                            black_box(*nc),
                        );
                        black_box(&c);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_matrix);
criterion_main!(benches);
