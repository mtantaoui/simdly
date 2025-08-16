//! Comprehensive Arctangent Benchmarks with SIMD vs Scalar Comparison

use std::hint::black_box;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// No fast_atan function available
use simdly::simd::SimdMath;

const VECTOR_SIZES: &[usize] = &[
    1_024,      // 4 KiB - L1 cache
    16_384,     // 64 KiB - L1→L2 transition
    262_144,    // 1 MiB - L2 cache, parallel SIMD threshold
    1_048_576,  // 4 MiB - L3 cache
    4_194_304,  // 16 MiB - L3→RAM transition
];

fn generate_test_data(len: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..len)
        .map(|_| (rng.random::<f32>() - 0.5) * 20.0)
        .collect() // Range [-10, 10] for atan
}

fn benchmark_atan_implementations(c: &mut Criterion) {
    for &size in VECTOR_SIZES {
        let mut group = c.benchmark_group(format!("Arctangent {}", format_size(size)));
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>()) as u64,
        ));

        let input_vec = generate_test_data(size);
        let input_slice = input_vec.as_slice();

        group.bench_with_input(BenchmarkId::new("Scalar", size), input_slice, |b, input| {
            b.iter(|| black_box(input.iter().map(|x| x.atan()).collect::<Vec<f32>>()))
        });

        group.bench_with_input(BenchmarkId::new("SIMD", size), input_slice, |b, input| {
            b.iter(|| black_box(input.atan()))
        });

        group.bench_with_input(
            BenchmarkId::new("Parallel SIMD", size),
            input_slice,
            |b, input| b.iter(|| black_box(black_box(input).par_atan())),
        );

        // Note: No fast_atan function available in the library

        group.finish();
    }
}

fn format_size(elements: usize) -> String {
    let bytes = elements * std::mem::size_of::<f32>();
    if bytes >= 1_073_741_824 {
        format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MiB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn all_benchmarks(c: &mut Criterion) {
    println!("🚀 Starting Comprehensive Arctangent Benchmarks");
    let start_time = Instant::now();
    benchmark_atan_implementations(c);
    let elapsed = start_time.elapsed();
    println!(
        "✅ Benchmark suite completed in {:.2} seconds",
        elapsed.as_secs_f64()
    );
}

criterion_group!(benches, all_benchmarks);
criterion_main!(benches);
