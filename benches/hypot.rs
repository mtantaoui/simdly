//! Comprehensive 2D Hypotenuse Benchmarks with SIMD vs Scalar Comparison
//!
//! This benchmark suite evaluates the performance of different 2D hypotenuse implementations
//! across various vector sizes. It focuses on comparing scalar and SIMD
//! implementations across different CPU cache hierarchies to understand their
//! performance characteristics.
//!
//! # Benchmark Categories
//!
//! ## 1. **Scalar vs SIMD Comparison**
//! - Pure scalar implementation using standard library hypot (baseline)
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

// Import the hypot implementations
use simdly::simd::SimdMath;

// ================================================================================================
// BENCHMARK CONFIGURATION
// ================================================================================================

/// Vector sizes designed to test performance across CPU cache hierarchies.
///
/// Each size targets a specific level of the memory hierarchy:
///
/// - **4 KiB**: Fits in L1 data cache (32-64 KiB) - Tests raw computational throughput
/// - **64 KiB**: Exceeds most L1 caches, uses L2 (256 KiB-1 MiB) - Tests L1→L2 transition
/// - **1 MiB**: Approaches L2 limits, may use L3 (8-64 MiB) - Tests L2→L3 transition, parallel SIMD threshold
/// - **4 MiB**: Fits in L3 cache - Tests L3 cache efficiency
/// - **16 MiB**: May exceed smaller L3 caches - Tests L3→RAM transition
///
/// Note: f32 = 4 bytes, so 1M elements = 4 MiB
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
/// A tuple containing (x_vector, y_vector) with appropriate ranges for 2D hypotenuse
///
/// # Data Characteristics
///
/// - **Reproducible**: Fixed seed ensures identical data across runs  
/// - **Realistic**: Values in [-100.0, 100.0] for meaningful 2D distance calculation
/// - **Numerically stable**: Values avoid edge cases that could skew performance
/// - **Cache-friendly**: Sequential generation optimizes initial cache population
fn generate_test_data(len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

    let x_values: Vec<f32> = (0..len).map(|_| rng.random_range(-100.0..=100.0)).collect();

    let y_values: Vec<f32> = (0..len).map(|_| rng.random_range(-100.0..=100.0)).collect();

    (x_values, y_values)
}

/// Scalar hypot implementation for benchmarking baseline.
fn scalar_hypot(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(x, y)| x.hypot(*y)).collect()
}

// ================================================================================================
// BENCHMARK IMPLEMENTATIONS
// ================================================================================================

/// Benchmarks all hypot implementations across different vector sizes.
///
/// This is the main benchmark function that creates comprehensive performance
/// measurements for scalar and SIMD hypot implementations.
fn benchmark_hypot_implementations(c: &mut Criterion) {
    for &size in VECTOR_SIZES {
        let mut group = c.benchmark_group(format!("Hypot {}", format_size(size)));

        // Configure throughput measurement for bandwidth analysis
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>() * 2) as u64, // Reading two input vectors
        ));

        // Generate test data once per size for consistency
        let (x_vec, y_vec) = generate_test_data(size);
        let x_slice = x_vec.as_slice();
        let y_slice = y_vec.as_slice();

        // Benchmark 1: Scalar Hypot (Baseline)
        group.bench_with_input(
            BenchmarkId::new("Scalar", size),
            &(x_slice, y_slice),
            |b, (x, y)| b.iter(|| black_box(scalar_hypot(black_box(x), black_box(y)))),
        );

        // Benchmark 2: SIMD Hypot
        group.bench_with_input(
            BenchmarkId::new("SIMD", size),
            &(x_slice, y_slice),
            |b, (x, y)| b.iter(|| black_box(x.hypot(black_box(y)))),
        );

        // Benchmark 3: Parallel SIMD Hypot
        group.bench_with_input(
            BenchmarkId::new("Parallel SIMD", size),
            &(x_slice, y_slice),
            |b, (x, y)| b.iter(|| black_box(x.par_hypot(black_box(y)))),
        );

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
///
/// Coordinates all benchmark suites and provides a comprehensive performance analysis
/// of hypot implementations across different scenarios.
fn all_benchmarks(c: &mut Criterion) {
    println!("🚀 Starting Comprehensive 2D Hypotenuse Benchmarks");
    println!(
        "   Testing {} vector sizes from {} to {}",
        VECTOR_SIZES.len(),
        format_size(VECTOR_SIZES[0]),
        format_size(*VECTOR_SIZES.last().unwrap())
    );

    let start_time = Instant::now();

    // Core benchmark suite - comprehensive size analysis
    benchmark_hypot_implementations(c);

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
