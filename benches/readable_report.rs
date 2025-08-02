use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use simdly::{simd::SimdMath, SimdAdd};
use std::time::Duration;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use simdly::simd::avx2::slice::scalar_cos;

#[cfg(target_arch = "aarch64")]
use simdly::simd::neon::slice::scalar_cos;

fn configure_criterion() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(100)
    // Removed .with_plots() to avoid system dependencies
}

/// Addition Operations Comparison - Small/Medium Sizes
fn addition_comparison_small_medium(c: &mut Criterion) {
    let mut group = c.benchmark_group("🔢 Addition Operations (Small/Medium)");

    println!(
        "\n🔢 ADDITION OPERATIONS BENCHMARK - Small/Medium\n{}",
        "=".repeat(50)
    );

    let test_sizes = vec![("Small (1K)", 1_000), ("Medium (10K)", 10_000)];

    for (size_name, size) in test_sizes {
        println!("\n📊 Testing {size_name} elements:");
        println!("{}", "-".repeat(30));

        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];

        group.throughput(Throughput::Elements(size as u64));

        // Scalar baseline
        group.bench_function(format!("{size_name}/Scalar"), |bench| {
            bench.iter(|| a.as_slice().scalar_add(b.as_slice()))
        });

        // SIMD implementation
        group.bench_function(format!("{size_name}/SIMD"), |bench| {
            bench.iter(|| a.as_slice().simd_add(b.as_slice()))
        });

        // Parallel SIMD for larger sizes
        group.bench_function(format!("{size_name}/Parallel_SIMD"), |bench| {
            bench.iter(|| a.as_slice().par_simd_add(b.as_slice()))
        });
    }

    group.finish();
}

/// Addition Operations Comparison - Large Sizes
fn addition_comparison_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("🔢 Addition Operations (Large)");

    println!(
        "\n🔢 ADDITION OPERATIONS BENCHMARK - Large\n{}",
        "=".repeat(50)
    );

    let test_sizes = vec![("Large (100K)", 100_000)];

    for (size_name, size) in test_sizes {
        println!("\n📊 Testing {size_name} elements:");
        println!("{}", "-".repeat(30));

        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];

        group.throughput(Throughput::Elements(size as u64));

        // Scalar baseline
        group.bench_function(format!("{size_name}/Scalar"), |bench| {
            bench.iter(|| a.as_slice().scalar_add(b.as_slice()))
        });

        // SIMD implementation
        group.bench_function(format!("{size_name}/SIMD"), |bench| {
            bench.iter(|| a.as_slice().simd_add(b.as_slice()))
        });

        // Parallel SIMD for larger sizes
        group.bench_function(format!("{size_name}/Parallel_SIMD"), |bench| {
            bench.iter(|| a.as_slice().par_simd_add(b.as_slice()))
        });
    }

    group.finish();
}

/// Addition Operations Comparison - XLarge Sizes
fn addition_comparison_xlarge(c: &mut Criterion) {
    let mut group = c.benchmark_group("🔢 Addition Operations (XLarge)");

    println!(
        "\n🔢 ADDITION OPERATIONS BENCHMARK - XLarge\n{}",
        "=".repeat(50)
    );

    let test_sizes = vec![("XLarge (1M)", 1_000_000)];

    for (size_name, size) in test_sizes {
        println!("\n📊 Testing {size_name} elements:");
        println!("{}", "-".repeat(30));

        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];

        group.throughput(Throughput::Elements(size as u64));

        // Scalar baseline
        group.bench_function(format!("{size_name}/Scalar"), |bench| {
            bench.iter(|| a.as_slice().scalar_add(b.as_slice()))
        });

        // SIMD implementation
        group.bench_function(format!("{size_name}/SIMD"), |bench| {
            bench.iter(|| a.as_slice().simd_add(b.as_slice()))
        });

        // Parallel SIMD for larger sizes
        group.bench_function(format!("{size_name}/Parallel_SIMD"), |bench| {
            bench.iter(|| a.as_slice().par_simd_add(b.as_slice()))
        });
    }

    group.finish();
}

/// Mathematical Functions Comparison
fn math_functions_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("🧮 Mathematical Functions");

    println!("\n🧮 MATHEMATICAL FUNCTIONS BENCHMARK\n{}", "=".repeat(50));

    let test_cases = vec![
        ("Cosine_1K", 1_000),
        ("Cosine_10K", 10_000),
        ("Cosine_100K", 100_000),
    ];

    for (test_name, size) in test_cases {
        println!("\n📊 Testing {}:", test_name.replace('_', " "));
        println!("{}", "-".repeat(30));

        // Generate reasonable test data for trig functions
        let angles: Vec<f32> = (0..size)
            .map(|i| (i as f32) * 0.001) // Small angles for better precision
            .collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_function(format!("{test_name}/Scalar"), |bench| {
            bench.iter(|| scalar_cos(&angles))
        });

        group.bench_function(format!("{test_name}/SIMD"), |bench| {
            bench.iter(|| angles.as_slice().cos())
        });
    }

    group.finish();
}

/// Platform Information Display
fn display_platform_info() {
    println!("\n🖥️  PLATFORM INFORMATION\n{}", "=".repeat(50));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        println!("Architecture: x86/x86_64 (Intel/AMD)");
        println!("SIMD ISA: AVX2 (256-bit vectors)");
        println!("Vector Width: 8 × f32 elements");
        println!("Alignment: 32-byte optimal");

        if std::is_x86_feature_detected!("avx2") {
            println!("✅ AVX2 support: DETECTED");
        } else {
            println!("❌ AVX2 support: NOT DETECTED");
        }

        if std::is_x86_feature_detected!("fma") {
            println!("✅ FMA support: DETECTED");
        } else {
            println!("❌ FMA support: NOT DETECTED");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("Architecture: ARM64 (AArch64)");
        println!("SIMD ISA: NEON (128-bit vectors)");
        println!("Vector Width: 4 × f32 elements");
        println!("Alignment: 16-byte optimal");
        println!("✅ NEON support: ALWAYS AVAILABLE");
    }

    println!("Rust Target: {}", std::env::consts::ARCH);
    println!("Optimization: Release mode with target-cpu features");
}

fn custom_runner(c: &mut Criterion) {
    // Display platform information first
    display_platform_info();

    // Run organized benchmarks with separate plots for each size category
    addition_comparison_small_medium(c);
    addition_comparison_large(c);
    addition_comparison_xlarge(c);
    math_functions_comparison(c);

    println!("\n🎉 BENCHMARK COMPLETE!\n{}", "=".repeat(50));
    println!("📊 Detailed results available in: target/criterion/");
    println!("🌐 Open target/criterion/reports/index.html for interactive charts");
    println!();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = custom_runner
}

criterion_main!(benches);
