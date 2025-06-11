use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use ndarray::Array1;
use simdly::simd::{neon::add::parallel_scalar_add, traits::SimdAdd};
use std::hint::black_box;

// --- Configuration ---
const VECTOR_LENGTHS: &[usize] = &[
    30_000,
    // 30_000,  // first threshold scalar -> Simd
    // 150_000, // simd to par_simd
    // // 500_000, // simd to par_simd
    // 128, // Small: Test overhead
    // // 512, 1024, 4096, // Medium: Likely L1/L2 cache-resident
    // // // 16_384,
    // // 65_536,  // Large: Likely L3 cache or memory-bound, good for parallelism
    // 262_144,   // Very Large: Definitely memory-bound
    // 1_048_576, // Huge: Stress test for SIMD and parallelism
    // // 4_194_304,  // Massive: Pushes the limits of SIMD and parallelism
    // 16_777_216, // Gigantic: Extreme case for SIMD and parallelism
    // // 67_108_864, // Extreme: Tests the limits of SIMD and parallelism
    // // 268_435_456, // Ultra: Tests the limits of SIMD and parallelism
    // 1_073_741_824, // Mega: Tests the limits of SIMD and parallelism
    //                // 4_294_967_296, // Giga: Tests the limits of SIMD and parallelism
];

fn generate_data(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = vec![1.0; len];
    let b: Vec<f32> = vec![1.0; len];

    (a, b)
}

fn bench_vector_addition(c: &mut Criterion) {
    let lengths = VECTOR_LENGTHS.iter()
    // .rev()
    ;

    // --- Setup that applies to all benchmarks ---

    for &vector_len in lengths {
        // The group name already contains the vector length, providing context.
        let mut group = c.benchmark_group(format!("VectorAddition/{vector_len}"));

        // group.sample_size(100);

        // Set throughput so results are reported in terms of elements/sec.
        group.throughput(Throughput::Elements(vector_len as u64));

        // Generate the data once per group.
        let (a_vec, b_vec) = generate_data(vector_len);

        // --- Benchmark 1: Scalar Addition ---
        group.bench_function("scalar_add", |bencher| {
            bencher.iter(|| {
                // Black-box the output to prevent the entire operation
                // from being optimized away.
                black_box(a_vec.scalar_add(black_box(&b_vec)))
            });
        });

        // --- Benchmark 2: SIMD Addition (AVX2) ---
        group.bench_function("parallel_scalar_add", |bencher| {
            bencher.iter(|| parallel_scalar_add(black_box(&a_vec), black_box(&b_vec)));
        });

        // // --- Benchmark 2: SIMD Addition (AVX2) ---
        // group.bench_function("simd_add (store)", |bencher| {
        //     bencher
        //         .iter(|| unsafe { simd_add_optimized_store(black_box(&a_vec), black_box(&b_vec)) });
        // });

        // // --- Benchmark 2: SIMD Addition (AVX2) ---
        // group.bench_function("simd_add (stream)", |bencher| {
        //     bencher.iter(|| unsafe {
        //         simd_add_optimized_stream(black_box(&a_vec), black_box(&b_vec))
        //     });
        // });

        // --- Benchmark 2: SIMD Addition (AVX2) ---
        group.bench_function("simd_add (simdly)", |bencher| {
            bencher.iter(|| black_box(a_vec.simd_add(black_box(&b_vec))));
        });

        // --- Benchmark 3: Parallel SIMD Addition (AVX2) ---
        group.bench_function("par_simd_add (avx2)", |bencher| {
            bencher.iter(|| black_box(a_vec.par_simd_add(black_box(&b_vec))));
        });

        // --- Benchmark 4: ndarray ---
        // ndarray often uses SIMD internally if available, so it's a great baseline.
        group.bench_function("ndarray", |bencher| {
            // Setup for this specific benchmark can be done here.
            // Cloning is necessary because `from_vec` takes ownership.
            let a_arr = Array1::from_vec(a_vec.clone());
            let b_arr = Array1::from_vec(b_vec.clone());
            bencher.iter(|| black_box(&a_arr + &b_arr));
        });

        group.finish();
    }
}

fn criterion_config() -> Criterion {
    Criterion::default().with_plots() // This enables plotting
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_vector_addition
}

// criterion_group!(benches, bench_vector_addition);

criterion_main!(benches);
