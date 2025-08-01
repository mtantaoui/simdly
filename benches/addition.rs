//! Comprehensive Addition Benchmarks with Rayon Parallelization
//!
//! This benchmark suite evaluates the performance of different addition implementations
//! across various vector sizes. It focuses on comparing scalar, SIMD, and parallel
//! implementations across different CPU cache hierarchies to understand their
//! performance characteristics.
//!
//! # Benchmark Categories
//!
//! ## 1. **Scalar vs SIMD vs Parallel Comparison**
//! - Pure scalar implementation (baseline)
//! - AVX2 SIMD vectorized implementation
//! - Rayon-based parallel SIMD implementation

use std::hint::black_box;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Import the addition implementations
use simdly::SimdAdd;

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
    1_024,       // 4 KiB - L1 cache
    4_096,       // 16 KiB - L1 cache
    16_384,      // 64 KiB - L1→L2 transition
    65_536,      // 256 KiB - L2 cache
    262_144,     // 1 MiB - L2 cache
    2 * 262_144, // 2 MiB - L2 cache
    1_048_576,   // 4 MiB - L2→L3 transition
    4_194_304,   // 16 MiB - L3 cache
    16_777_216,  // 64 MiB - L3→RAM transition
    33_554_432,  // 128 MiB - Main memory
];

/// Threshold for enabling parallel benchmarks.
///
/// Below this size, parallel overhead typically exceeds benefits.
/// Based on empirical testing in slice.rs implementation.
const PARALLEL_SIZE_THRESHOLD: usize = 10_000;

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
/// * `len` - Number of f32 elements to generate in each vector
///
/// # Returns
///
/// A tuple of two vectors (a, b) containing random f32 values in range [0.0, 1.0)
///
/// # Data Characteristics
///
/// - **Reproducible**: Fixed seed ensures identical data across runs  
/// - **Realistic**: Random values exercise full SIMD pipeline
/// - **Numerically stable**: Values in [0,1) avoid overflow/underflow issues
/// - **Cache-friendly**: Sequential generation optimizes initial cache population
fn generate_test_data(len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

    let a: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();

    let b: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();

    (a, b)
}

// ================================================================================================
// BENCHMARK IMPLEMENTATIONS
// ================================================================================================

/// Benchmarks all addition implementations across different vector sizes.
///
/// This is the main benchmark function that creates comprehensive performance
/// measurements for scalar, SIMD, and parallel addition implementations.
fn benchmark_addition_implementations(c: &mut Criterion) {
    for &size in VECTOR_SIZES {
        let mut group = c.benchmark_group(format!("Addition_{}", format_size(size)));

        // Configure throughput measurement for bandwidth analysis
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>() * 2) as u64, // *2 for reading both input vectors
        ));

        // Generate test data once per size for consistency
        let (a_vec, b_vec) = generate_test_data(size);
        let a_slice = a_vec.as_slice();
        let b_slice = b_vec.as_slice();

        // Benchmark 1: SIMD Addition
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(a_slice, b_slice),
            |b, (a, b_data)| b.iter(|| black_box(a.simd_add(black_box(*b_data)))),
        );

        // Benchmark 2: Scalar Addition (Baseline)
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &(a_slice, b_slice),
            |b, (a, b_data)| b.iter(|| black_box(a.scalar_add(black_box(*b_data)))),
        );

        // Benchmark 3: Parallel SIMD Addition (only for larger sizes)
        if size >= PARALLEL_SIZE_THRESHOLD {
            group.bench_with_input(
                BenchmarkId::new("parallel_simd", size),
                &(a_slice, b_slice),
                |b, (a, b_data)| b.iter(|| black_box(a.par_simd_add(black_box(*b_data)))),
            );
        }

        // Benchmark 4: ndarray Reference Implementation
        let a_ndarray = Array1::from_vec(a_vec.clone());
        let b_ndarray = Array1::from_vec(b_vec.clone());
        group.bench_with_input(
            BenchmarkId::new("ndarray", size),
            &(&a_ndarray, &b_ndarray),
            |b, (a, b_data)| b.iter(|| black_box(*a + *b_data)),
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
        format!("{:.1}_GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        // 1 MiB
        format!("{:.1}_MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        // 1 KiB
        format!("{:.1}_KiB", bytes as f64 / 1024.0)
    } else {
        format!("{}_B", bytes)
    }
}

/// Main benchmark orchestrator function.
///
/// Coordinates all benchmark suites and provides a comprehensive performance analysis
/// of addition implementations across different scenarios.
fn all_benchmarks(c: &mut Criterion) {
    println!("🚀 Starting Comprehensive Addition Benchmarks");
    println!(
        "   Testing {} vector sizes from {} to {}",
        VECTOR_SIZES.len(),
        format_size(VECTOR_SIZES[0]),
        format_size(*VECTOR_SIZES.last().unwrap())
    );

    let start_time = Instant::now();

    // Core benchmark suite - comprehensive size analysis
    benchmark_addition_implementations(c);

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
