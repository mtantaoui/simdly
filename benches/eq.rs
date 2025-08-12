use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simdly::simd::SimdCmp;
use std::f32::consts::PI;

/// Generate test data for comparison benchmarks
fn generate_test_data(size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut data_a = Vec::with_capacity(size);
    let mut data_b = Vec::with_capacity(size);

    for i in 0..size {
        let x = (i as f32) * 0.01;
        let y = (i as f32) * 0.01 + 0.001; // Slight offset for variety

        data_a.push(x.sin() * PI);
        data_b.push(y.cos() * PI);
    }

    (data_a, data_b)
}

/// Scalar implementation for element-wise equality comparison
fn scalar_eq(a: &[f32], b: &[f32]) -> Vec<bool> {
    a.iter().zip(b.iter()).map(|(x, y)| x == y).collect()
}

/// Benchmark equality comparison operations across different vector sizes
fn bench_eq_operations(c: &mut Criterion) {
    let sizes = [64, 256, 1024, 4096, 16384];

    for size in sizes {
        let (data_a, data_b) = generate_test_data(size);

        let mut group = c.benchmark_group("eq_comparison");
        group.throughput(Throughput::Elements(size as u64));

        // Benchmark scalar equality
        group.bench_with_input(BenchmarkId::new("scalar_eq", size), &size, |b, _| {
            b.iter(|| {
                let result = scalar_eq(black_box(&data_a), black_box(&data_b));
                black_box(result)
            });
        });

        // Benchmark SIMD equality
        group.bench_with_input(BenchmarkId::new("simd_eq", size), &size, |b, _| {
            b.iter(|| {
                let result = data_a.elementwise_eq(black_box(&data_b));
                black_box(result)
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_eq_operations);
criterion_main!(benches);
