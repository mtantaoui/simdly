use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use simdly::simd::traits::{SimdAdd, SimdCos};

// ====================================================================================
// --- Configuration: A good strategy covers different memory/cache hierarchies ---
// ====================================================================================

/// Vector sizes chosen to test performance across different CPU cache levels.
///
/// *   4 KiB: Fits comfortably in L1 data cache (typically 32-64 KiB). Tests raw compute.
/// *   64 KiB: Often pushes the limits of L1, starts involving L2 cache.
/// *   1 MiB: Fits in L2 cache (typically 256 KiB - 4 MiB), but not L1.
/// *   16 MiB: Exceeds most L2 caches, fits in L3 cache (typically 8-64 MiB).
/// *   64 MiB: Exceeds most L3 caches. This becomes a memory-bound benchmark,
///     testing the speed of data transfer from RAM.
///
/// An f32 is 4 bytes. `(1024 * 1024 * 4) / 4 = 1_048_576` elements is 4 MiB.
const VECTOR_SIZES: &[usize] = &[
    1024,             // 4 KiB
    16 * 1024,        // 64 KiB
    256 * 1024,       // 1 MiB
    4 * 1024 * 1024,  // 16 MiB
    16 * 1024 * 1024, // 64 MiB
];

/// Generates pseudo-random f32 vectors. Using a fixed seed ensures that the "random"
/// data is the same for every benchmark run, making results comparable over time.
fn generate_random_data(len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    let a: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();
    let b: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();
    (a, b)
}

// ====================================================================================
// --- Main Benchmark Definitions ---
// ====================================================================================

/// The main function that defines all our benchmark groups.
fn all_benchmarks(c: &mut Criterion) {
    // --- Benchmark Suite 1: Vector Addition ---
    for &size in VECTOR_SIZES {
        let mut group = c.benchmark_group("Addition".to_string());
        // Set throughput so the report shows elements/sec or GiB/s
        group.throughput(Throughput::Bytes(
            size as u64 * std::mem::size_of::<f32>() as u64,
        ));

        // Generate data once per size
        let (a_vec, b_vec) = generate_random_data(size);
        let a_arr = Array1::from_vec(a_vec.clone());
        let b_arr = Array1::from_vec(b_vec.clone());

        // For each implementation, we create a benchmark that will be plotted
        // against the input size (`size`).
        group.bench_with_input(BenchmarkId::new("scalar", size), &a_vec, |b, v| {
            b.iter(|| black_box(v.scalar_add(black_box(&b_vec))))
        });

        group.bench_with_input(BenchmarkId::new("simd (simdly)", size), &a_vec, |b, v| {
            b.iter(|| black_box(v.simd_add(black_box(&b_vec))))
        });

        group.bench_with_input(
            BenchmarkId::new("parallel simd (simdly)", size),
            &a_vec,
            |b, v| b.iter(|| black_box(v.par_simd_add(black_box(&b_vec)))),
        );

        // Corrected ndarray benchmark for addition
        group.bench_with_input(
            BenchmarkId::new("ndarray", size),
            &a_arr, // Pass only the first array as input
            |b, a| {
                // `a` is the &Array1 from the line above
                // The `iter` closure captures `b_arr` from the surrounding `for` loop scope.
                b.iter(|| black_box(a + &b_arr));
            },
        );
        group.finish();
    }

    // --- Benchmark Suite 2: Vector Cosine ---
    for &size in VECTOR_SIZES {
        let mut group = c.benchmark_group("Cosine".to_string());
        group.throughput(Throughput::Bytes(
            size as u64 * std::mem::size_of::<f32>() as u64,
        ));

        let (a_vec, _) = generate_random_data(size);
        let a_arr = Array1::from_vec(a_vec.clone());

        group.bench_with_input(BenchmarkId::new("scalar", size), &a_vec, |b, v| {
            b.iter(|| black_box(v.scalar_cos()))
        });

        group.bench_with_input(BenchmarkId::new("simd (simdly)", size), &a_vec, |b, v| {
            b.iter(|| black_box(v.simd_cos()))
        });

        group.bench_with_input(
            BenchmarkId::new("parallel simd (simdly)", size),
            &a_vec,
            |b, v| b.iter(|| black_box(v.par_simd_cos())),
        );

        group.bench_with_input(BenchmarkId::new("ndarray", size), &a_arr, |b, v| {
            b.iter(|| black_box(v.cos()))
        });
        group.finish();
    }
}

// Register the benchmark suite with Criterion
criterion_group!(benches, all_benchmarks);
criterion_main!(benches);
