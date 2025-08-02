//! Comprehensive Cosine Benchmarks with SIMD vs Scalar Comparison
//!
//! This benchmark suite evaluates the performance of different cosine implementations
//! across various vector sizes. It focuses on comparing scalar and SIMD
//! implementations across different CPU cache hierarchies to understand their
//! performance characteristics.
//!
//! # Benchmark Categories
//!
//! ## 1. **Scalar vs SIMD Comparison**
//! - Pure scalar implementation using standard library cosine (baseline)
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

use simdly::simd::avx2::slice::{fast_cos, parallel_simd_cos};
#[cfg(target_arch = "aarch64")]
use simdly::simd::neon::slice::scalar_cos;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use simdly::simd::avx2::slice::scalar_cos;

// Import the cosine implementations
use simdly::simd::SimdMath;

// ================================================================================================
// BENCHMARK CONFIGURATION
// ================================================================================================

/// Vector sizes designed to test performance across CPU cache hierarchies.
///
/// Each size targets a specific level of the memory hierarchy:
///
/// - **4 KiB**: Fits in L1 data cache (32-64 KiB) - Tests raw computational throughput
/// - **16 KiB**: Still fits in L1 but approaches limits - Tests L1 efficiency  
/// - **64 KiB**: Exceeds most L1 caches, uses L2 (256 KiB-1 MiB) - Tests L1→L2 transition
/// - **256 KiB**: Fits comfortably in L2 cache - Tests L2 cache efficiency
/// - **1 MiB**: Approaches L2 limits, may use L3 (8-64 MiB) - Tests L2→L3 transition
/// - **4 MiB**: Fits in L3 cache - Tests L3 cache efficiency
/// - **16 MiB**: May exceed smaller L3 caches - Tests L3→RAM transition
/// - **64 MiB**: Exceeds most L3 caches - Tests main memory bandwidth
/// - **128 MiB**: Definitely memory-bound - Tests sustained memory throughput
///
/// Note: f32 = 4 bytes, so 1M elements = 4 MiB
const VECTOR_SIZES: &[usize] = &[
    1_024,      // 4 KiB - L1 cache
    4_096,      // 16 KiB - L1 cache
    16_384,     // 64 KiB - L1→L2 transition
    65_536,     // 256 KiB - L2 cache
    262_144,    // 1 MiB - L2 cache
    1_048_576,  // 4 MiB - L2→L3 transition
    4_194_304,  // 16 MiB - L3 cache
    16_777_216, // 64 MiB - L3→RAM transition
    33_554_432, // 128 MiB - Main memory
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
/// A vector containing random f32 values in range [0.0, 2π) for meaningful cosine input
///
/// # Data Characteristics
///
/// - **Reproducible**: Fixed seed ensures identical data across runs  
/// - **Realistic**: Random values in [0, 2π) exercise full cosine range
/// - **Numerically stable**: Values avoid edge cases that could skew performance
/// - **Cache-friendly**: Sequential generation optimizes initial cache population
fn generate_test_data(len: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

    (0..len)
        .map(|_| rng.random::<f32>() * 2.0 * std::f32::consts::PI) // Range [0, 2π)
        .collect()
}

// ================================================================================================
// BENCHMARK IMPLEMENTATIONS
// ================================================================================================

/// Benchmarks all cosine implementations across different vector sizes.
///
/// This is the main benchmark function that creates comprehensive performance
/// measurements for scalar and SIMD cosine implementations.
fn benchmark_cosine_implementations(c: &mut Criterion) {
    for &size in VECTOR_SIZES {
        let mut group = c.benchmark_group(format!("Cosine_{}", format_size(size)));

        // Configure throughput measurement for bandwidth analysis
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>()) as u64, // Reading input vector once
        ));

        // Generate test data once per size for consistency
        let input_vec = generate_test_data(size);
        let input_slice = input_vec.as_slice();

        // Benchmark 1: Scalar Cosine (Baseline)
        group.bench_with_input(BenchmarkId::new("scalar", size), input_slice, |b, input| {
            b.iter(|| black_box(scalar_cos(black_box(input))))
        });

        // Benchmark 2: SIMD Cosine
        group.bench_with_input(BenchmarkId::new("simd", size), input_slice, |b, input| {
            b.iter(|| black_box(input.cos()))
        });

        // Benchmark 3:
        group.bench_with_input(
            BenchmarkId::new("parallel SIMD", size),
            input_slice,
            |b, input| b.iter(|| black_box(parallel_simd_cos(black_box(input)))),
        );

        // Benchmark 4:
        group.bench_with_input(BenchmarkId::new("Fast", size), input_slice, |b, input| {
            b.iter(|| black_box(fast_cos(black_box(input))))
        });

        group.finish();
    }
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

/// Formats vector sizes into human-readable strings.
///
/// Converts byte counts into appropriate units (KiB, MiB, GiB) for display.
fn format_size(elements: usize) -> String {
    let bytes = elements * std::mem::size_of::<f32>();

    if bytes >= 1_073_741_824 {
        // 1 GiB
        format!("{:.1}_GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        // 1 MiB
        format!("{:.1}_MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        // 1 KiB
        format!("{:.1}_KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}_B")
    }
}

/// Main benchmark orchestrator function.
///
/// Coordinates all benchmark suites and provides a comprehensive performance analysis
/// of cosine implementations across different scenarios.
fn all_benchmarks(c: &mut Criterion) {
    println!("🚀 Starting Comprehensive Cosine Benchmarks");
    println!(
        "   Testing {} vector sizes from {} to {}",
        VECTOR_SIZES.len(),
        format_size(VECTOR_SIZES[0]),
        format_size(*VECTOR_SIZES.last().unwrap())
    );

    let start_time = Instant::now();

    // Core benchmark suite - comprehensive size analysis
    benchmark_cosine_implementations(c);

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
