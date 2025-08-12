//! Comprehensive Two-Argument Arctangent Benchmarks with SIMD vs Scalar Comparison

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use simdly::simd::SimdMath;

const VECTOR_SIZES: &[usize] = &[
    1_024,
    4_096,
    16_384,
    65_536,
    2 * 65_536,
    3 * 65_536,
    262_144,
    1_048_576,
    4_194_304,
    16_777_216,
    33_554_432,
];

fn generate_test_data(len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(42);
    let y_data: Vec<f32> = (0..len)
        .map(|_| {
            let val = (rng.random::<f32>() - 0.5) * 200.0; // Range [-100, 100] for y
                                                           // Avoid exact zero to prevent (0,0) case
            if val.abs() < 1e-6 {
                if val >= 0.0 {
                    1e-6
                } else {
                    -1e-6
                }
            } else {
                val
            }
        })
        .collect();

    let x_data: Vec<f32> = (0..len)
        .map(|_| {
            let val = (rng.random::<f32>() - 0.5) * 200.0; // Range [-100, 100] for x
                                                           // Avoid exact zero to prevent (0,0) case
            if val.abs() < 1e-6 {
                if val >= 0.0 {
                    1e-6
                } else {
                    -1e-6
                }
            } else {
                val
            }
        })
        .collect();

    (y_data, x_data)
}

fn benchmark_atan2_implementations(c: &mut Criterion) {
    for &size in VECTOR_SIZES {
        let mut group = c.benchmark_group(format!("Two-Argument Arctangent {}", format_size(size)));
        group.throughput(Throughput::Bytes(
            (size * std::mem::size_of::<f32>() * 2) as u64,
        )); // 2x for both y and x inputs

        let (y_vec, x_vec) = generate_test_data(size);
        let y_slice = y_vec.as_slice();
        let x_slice = x_vec.as_slice();

        group.bench_with_input(
            BenchmarkId::new("Scalar", size),
            &(y_slice, x_slice),
            |b, (y_input, x_input)| {
                b.iter(|| {
                    black_box(
                        y_input
                            .iter()
                            .zip(x_input.iter())
                            .map(|(y, x)| y.atan2(*x))
                            .collect::<Vec<f32>>(),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("SIMD", size),
            &(y_slice, x_slice),
            |b, (y_input, x_input)| b.iter(|| black_box(y_input.atan2(x_input))),
        );

        group.bench_with_input(
            BenchmarkId::new("Parallel SIMD", size),
            &(y_slice, x_slice),
            |b, (y_input, x_input)| {
                b.iter(|| black_box(black_box(y_input).par_atan2(black_box(x_input))))
            },
        );

        group.finish();
    }
}

/// Formats vector sizes for human-readable benchmark output.
///
/// Converts byte counts into readable units with appropriate suffixes.
/// This helps in understanding benchmark results at a glance.
fn format_size(size: usize) -> String {
    const UNITS: &[&str] = &["", "K", "M", "G"];
    let mut value = size as f64;
    let mut unit_index = 0;

    while value >= 1024.0 && unit_index < UNITS.len() - 1 {
        value /= 1024.0;
        unit_index += 1;
    }

    if value.fract() == 0.0 {
        format!("{:.0}{}", value, UNITS[unit_index])
    } else {
        format!("{:.1}{}", value, UNITS[unit_index])
    }
}

criterion_group!(benches, benchmark_atan2_implementations);
criterion_main!(benches);
