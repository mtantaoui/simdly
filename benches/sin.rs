//! Comprehensive Sine Benchmarks with SIMD vs Scalar Comparison
//!
//! This benchmark suite evaluates the performance of different sine implementations
//! across various vector sizes. It focuses on comparing scalar and SIMD
//! implementations across different CPU cache hierarchies to understand their
//! performance characteristics.
//!
//! # Benchmark Categories
//!
//! ## 1. **Scalar vs SIMD Comparison**
//! - Pure scalar implementation using standard library sine (baseline)
//! - ARM NEON SIMD vectorized implementation using custom math functions
//!
//! ## 2. **Memory Hierarchy Analysis**
//! - L1 Cache: 4 KiB vectors (raw compute performance)
//! - L2 Cache: 64 KiB vectors (L1→L2 transition)
//! - L3 Cache: 1-16 MiB vectors (L2→L3 transition)
//! - Main Memory: 64+ MiB vectors (memory bandwidth bound)

use std::hint::black_box;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// No fast_sin function available

// Import the sine implementations
use simdly::simd::SimdMath;

// ================================================================================================
// BENCHMARK CONFIGURATION
// ================================================================================================

/// Vector sizes designed to test performance across CPU cache hierarchies.
const VECTOR_SIZES: &[usize] = &[
    1_024,     // 4 KiB - L1 cache
    16_384,    // 64 KiB - L1→L2 transition
    262_144,   // 1 MiB - L2 cache, parallel SIMD threshold
    1_048_576, // 4 MiB - L3 cache
    4_194_304, // 16 MiB - L3→RAM transition
];

// ================================================================================================
// TEST DATA GENERATION
// ================================================================================================

/// Generates reproducible pseudo-random test data for benchmarking.
///
/// Uses a fixed seed to ensure consistent data across benchmark runs,
/// enabling meaningful performance comparisons over time.
///
/// # Arguments
///
/// * `len` - Number of f32 elements to generate in the vector
///
/// # Returns
///
/// A vector containing random f32 values in range [0.0, 2π) for meaningful sine input
fn generate_test_data(len: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

    (0..len)
        .map(|_| rng.random::<f32>() * 2.0 * std::f32::consts::PI) // Range [0, 2π)
        .collect()
}

// ================================================================================================
// BENCHMARK IMPLEMENTATIONS
// ================================================================================================

/// Benchmarks all sine implementations across different vector sizes.
fn benchmark_sin_implementations(c: &mut Criterion) {
    for &size in VECTOR_SIZES {
        let mut group = c.benchmark_group(format!("Sine {}", format_size(size)));

        // Configure throughput measurement for bandwidth analysis
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>()) as u64, // Reading input vector once
        ));

        // Generate test data once per size for consistency
        let input_vec = generate_test_data(size);
        let input_slice = input_vec.as_slice();

        // Benchmark 1: Scalar Sine (Baseline)
        group.bench_with_input(BenchmarkId::new("Scalar", size), input_slice, |b, input| {
            b.iter(|| black_box(input.iter().map(|x| x.sin()).collect::<Vec<f32>>()))
        });

        // Benchmark 2: SIMD Sine
        group.bench_with_input(BenchmarkId::new("SIMD", size), input_slice, |b, input| {
            b.iter(|| black_box(input.sin()))
        });

        // Benchmark 3: Parallel SIMD Sine
        group.bench_with_input(
            BenchmarkId::new("Parallel SIMD", size),
            input_slice,
            |b, input| b.iter(|| black_box(black_box(input).par_sin())),
        );

        // Note: No fast_sin function available in the library

        group.finish();
    }
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

/// Formats vector sizes into human-readable strings.
fn format_size(elements: usize) -> String {
    let bytes = elements * std::mem::size_of::<f32>();

    if bytes >= 1_073_741_824 {
        // 1 GiB
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        // 1 MiB
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        // 1 KiB
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Main benchmark orchestrator function.
fn all_benchmarks(c: &mut Criterion) {
    println!("🚀 Starting Comprehensive Sine Benchmarks");
    println!(
        "   Testing {} vector sizes from {} to {}",
        VECTOR_SIZES.len(),
        format_size(VECTOR_SIZES[0]),
        format_size(*VECTOR_SIZES.last().unwrap())
    );

    let start_time = Instant::now();

    // Core benchmark suite - comprehensive size analysis
    benchmark_sin_implementations(c);

    let elapsed = start_time.elapsed();
    println!(
        "✅ Benchmark suite completed in {:.2} seconds",
        elapsed.as_secs_f64()
    );
}

// ================================================================================================
// CRITERION INTEGRATION
// ================================================================================================

// Register all benchmark groups with Criterion
criterion_group!(benches, all_benchmarks);
criterion_main!(benches);
