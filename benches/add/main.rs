mod add_avx2;

use add_avx2::bench_vector_addition;

use criterion::{criterion_group, criterion_main};

// --- Configuration ---
const VECTOR_LENGTHS: &[usize] = &[
    30_000,  // first threshold scalar -> Simd
    150_000, // simd to par_simd
    500_000, // simd to par_simd
             // 128,     // Small: Test overhead// Medium: Likely L1/L2 cache-resident
             // 512, 1024, 4096, // Medium: Likely L1/L2 cache-resident
             // // 16_384,
             // 65_536,  // Large: Likely L3 cache or memory-bound, good for parallelism
             // 262_144, // Very Large: Definitely memory-bound
             // 1_048_576, // Huge: Stress test for SIMD and parallelism
             // 4_194_304,  // Massive: Pushes the limits of SIMD and parallelism
             // 16_777_216, // Gigantic: Extreme case for SIMD and parallelism
             // 67_108_864, // Extreme: Tests the limits of SIMD and parallelism
             // 268_435_456, // Ultra: Tests the limits of SIMD and parallelism
             // 1_073_741_824, // Mega: Tests the limits of SIMD and parallelism
             // 4_294_967_296, // Giga: Tests the limits of SIMD and parallelism
];

fn generate_data(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = vec![1.0; len];
    let b: Vec<f32> = vec![1.0; len];

    (a, b)
}

criterion_group!(benches, bench_vector_addition);

criterion_main!(benches);
