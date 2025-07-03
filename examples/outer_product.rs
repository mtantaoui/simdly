use std::arch::x86_64::*;
use std::time::Instant;

// Maximum performance implementation using micro-architectural optimizations
// Exploits superscalar execution, port utilization, and instruction fusion
#[target_feature(enable = "avx2")]
pub unsafe fn outer_product_max_performance(a: __m256, b: __m256, result: &mut [__m256; 8]) {
    // Pre-compute both halves to maximize instruction-level parallelism
    let a_lo = _mm256_permute2f128_ps::<0x00>(a, a); // duplicate lower 128 bits
    let a_hi = _mm256_permute2f128_ps::<0x11>(a, a); // duplicate upper 128 bits

    // Interleave operations to maximize port utilization on modern CPUs
    // Intel CPUs have multiple execution ports - we want to use them all

    // Group 1: Prepare broadcasts for elements 0,1,4,5 (interleaved)
    let a0 = _mm256_permute_ps::<0x00>(a_lo); // Port 5
    let a4 = _mm256_permute_ps::<0x00>(a_hi); // Port 5 (parallel)
    let a1 = _mm256_permute_ps::<0x55>(a_lo); // Port 5
    let a5 = _mm256_permute_ps::<0x55>(a_hi); // Port 5 (parallel)

    // Group 2: Compute multiplications for rows 0,4,1,5 (interleaved)
    result[0] = _mm256_mul_ps(a0, b); // Port 0/1
    result[4] = _mm256_mul_ps(a4, b); // Port 0/1 (parallel)
    result[1] = _mm256_mul_ps(a1, b); // Port 0/1
    result[5] = _mm256_mul_ps(a5, b); // Port 0/1 (parallel)

    // Group 3: Prepare broadcasts for elements 2,3,6,7 (interleaved)
    let a2 = _mm256_permute_ps::<0xAA>(a_lo); // Port 5
    let a6 = _mm256_permute_ps::<0xAA>(a_hi); // Port 5 (parallel)
    let a3 = _mm256_permute_ps::<0xFF>(a_lo); // Port 5
    let a7 = _mm256_permute_ps::<0xFF>(a_hi); // Port 5 (parallel)

    // Group 4: Compute multiplications for rows 2,6,3,7 (interleaved)
    result[2] = _mm256_mul_ps(a2, b); // Port 0/1
    result[6] = _mm256_mul_ps(a6, b); // Port 0/1 (parallel)
    result[3] = _mm256_mul_ps(a3, b); // Port 0/1
    result[7] = _mm256_mul_ps(a7, b); // Port 0/1 (parallel)
}

// Cache-optimized version for when result matrix will be used immediately
// Uses temporal locality optimizations and prefetching
#[target_feature(enable = "avx2")]
pub unsafe fn outer_product_cache_optimized(a: __m256, b: __m256, result: &mut [__m256; 8]) {
    // Prefetch result memory locations
    _mm_prefetch(result.as_ptr() as *const i8, _MM_HINT_T0);
    _mm_prefetch(result.as_ptr().add(4) as *const i8, _MM_HINT_T0);

    // Use optimized computation pattern
    let a_lo = _mm256_permute2f128_ps::<0x00>(a, a);
    let a_hi = _mm256_permute2f128_ps::<0x11>(a, a);

    // Compute all broadcasts first (better for cache)
    let broadcasts = [
        _mm256_permute_ps::<0x00>(a_lo),
        _mm256_permute_ps::<0x55>(a_lo),
        _mm256_permute_ps::<0xAA>(a_lo),
        _mm256_permute_ps::<0xFF>(a_lo),
        _mm256_permute_ps::<0x00>(a_hi),
        _mm256_permute_ps::<0x55>(a_hi),
        _mm256_permute_ps::<0xAA>(a_hi),
        _mm256_permute_ps::<0xFF>(a_hi),
    ];

    // Vectorized multiplication (compiler will optimize)
    for i in 0..8 {
        result[i] = _mm256_mul_ps(broadcasts[i], b);
    }
}

// Specialized version using FMA (Fused Multiply-Add) for Haswell+ CPUs
#[target_feature(enable = "avx2,fma")]
pub unsafe fn outer_product_fma(
    a: __m256,
    b: __m256,
    result: &mut [__m256; 8],
    accumulator: &[__m256; 8],
) {
    let a_lo = _mm256_permute2f128_ps::<0x00>(a, a);
    let a_hi = _mm256_permute2f128_ps::<0x11>(a, a);

    // Use FMA to compute result = a[i] * b + accumulator[i] in one operation
    result[0] = _mm256_fmadd_ps(_mm256_permute_ps::<0x00>(a_lo), b, accumulator[0]);
    result[1] = _mm256_fmadd_ps(_mm256_permute_ps::<0x55>(a_lo), b, accumulator[1]);
    result[2] = _mm256_fmadd_ps(_mm256_permute_ps::<0xAA>(a_lo), b, accumulator[2]);
    result[3] = _mm256_fmadd_ps(_mm256_permute_ps::<0xFF>(a_lo), b, accumulator[3]);
    result[4] = _mm256_fmadd_ps(_mm256_permute_ps::<0x00>(a_hi), b, accumulator[4]);
    result[5] = _mm256_fmadd_ps(_mm256_permute_ps::<0x55>(a_hi), b, accumulator[5]);
    result[6] = _mm256_fmadd_ps(_mm256_permute_ps::<0xAA>(a_hi), b, accumulator[6]);
    result[7] = _mm256_fmadd_ps(_mm256_permute_ps::<0xFF>(a_hi), b, accumulator[7]);
}

// // Ultimate performance version with manual instruction scheduling
// // Uses inline assembly with optimal instruction ordering
// #[target_feature(enable = "avx2")]
// pub unsafe fn outer_product_ultimate_performance(a: __m256, b: __m256, result: &mut [__m256; 8]) {
//     std::arch::asm!(
//         // Load inputs
//         "vmovaps {a}, %ymm0",                           // ymm0 = a
//         "vmovaps {b}, %ymm1",                           // ymm1 = b

//         // Prepare both halves early
//         "vperm2f128 $0x00, %ymm0, %ymm0, %ymm2",       // ymm2 = a_lo
//         "vperm2f128 $0x11, %ymm0, %ymm0, %ymm3",       // ymm3 = a_hi

//         // Interleaved computation for maximum throughput
//         "vpermilps $0x00, %ymm2, %ymm4",               // ymm4 = broadcast a[0]
//         "vpermilps $0x00, %ymm3, %ymm5",               // ymm5 = broadcast a[4]
//         "vmulps %ymm1, %ymm4, %ymm4",                  // ymm4 = result[0]
//         "vmulps %ymm1, %ymm5, %ymm5",                  // ymm5 = result[4]

//         "vpermilps $0x55, %ymm2, %ymm6",               // ymm6 = broadcast a[1]
//         "vpermilps $0x55, %ymm3, %ymm7",               // ymm7 = broadcast a[5]
//         "vmovaps %ymm4, {result0}",                     // store result[0]
//         "vmovaps %ymm5, {result4}",                     // store result[4]

//         "vmulps %ymm1, %ymm6, %ymm6",                  // ymm6 = result[1]
//         "vmulps %ymm1, %ymm7, %ymm7",                  // ymm7 = result[5]
//         "vpermilps $0xAA, %ymm2, %ymm4",               // ymm4 = broadcast a[2]
//         "vpermilps $0xAA, %ymm3, %ymm5",               // ymm5 = broadcast a[6]

//         "vmovaps %ymm6, {result1}",                     // store result[1]
//         "vmovaps %ymm7, {result5}",                     // store result[5]
//         "vmulps %ymm1, %ymm4, %ymm4",                  // ymm4 = result[2]
//         "vmulps %ymm1, %ymm5, %ymm5",                  // ymm5 = result[6]

//         "vpermilps $0xFF, %ymm2, %ymm6",               // ymm6 = broadcast a[3]
//         "vpermilps $0xFF, %ymm3, %ymm7",               // ymm7 = broadcast a[7]
//         "vmovaps %ymm4, {result2}",                     // store result[2]
//         "vmovaps %ymm5, {result6}",                     // store result[6]

//         "vmulps %ymm1, %ymm6, %ymm6",                  // ymm6 = result[3]
//         "vmulps %ymm1, %ymm7, %ymm7",                  // ymm7 = result[7]
//         "vmovaps %ymm6, {result3}",                     // store result[3]
//         "vmovaps %ymm7, {result7}",                     // store result[7]

//         a = in(ymm_reg) a,
//         b = in(ymm_reg) b,
//         result0 = out(ymm_reg) result[0],
//         result1 = out(ymm_reg) result[1],
//         result2 = out(ymm_reg) result[2],
//         result3 = out(ymm_reg) result[3],
//         result4 = out(ymm_reg) result[4],
//         result5 = out(ymm_reg) result[5],
//         result6 = out(ymm_reg) result[6],
//         result7 = out(ymm_reg) result[7],
//         out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _,
//         out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
//     );
// }

// Generic wrapper functions for easy usage
pub fn outer_product_max_perf(a: [f32; 8], b: [f32; 8]) -> [[f32; 8]; 8] {
    unsafe {
        let a_vec = _mm256_loadu_ps(a.as_ptr());
        let b_vec = _mm256_loadu_ps(b.as_ptr());
        let mut result = [_mm256_setzero_ps(); 8];

        outer_product_max_performance(a_vec, b_vec, &mut result);

        let mut output = [[0.0f32; 8]; 8];
        for i in 0..8 {
            _mm256_storeu_ps(output[i].as_mut_ptr(), result[i]);
        }
        output
    }
}

// pub fn outer_product_ultimate_perf(a: [f32; 8], b: [f32; 8]) -> [[f32; 8]; 8] {
//     unsafe {
//         let a_vec = _mm256_loadu_ps(a.as_ptr());
//         let b_vec = _mm256_loadu_ps(b.as_ptr());
//         let mut result = [_mm256_setzero_ps(); 8];

//         outer_product_ultimate_performance(a_vec, b_vec, &mut result);

//         let mut output = [[0.0f32; 8]; 8];
//         for i in 0..8 {
//             _mm256_storeu_ps(output[i].as_mut_ptr(), result[i]);
//         }
//         output
//     }
// }

// Benchmark all versions
fn benchmark_all() {
    const ITERATIONS: usize = 10_000_000;

    let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    unsafe {
        let a = _mm256_loadu_ps(a_data.as_ptr());
        let b = _mm256_loadu_ps(b_data.as_ptr());
        let mut result = [_mm256_setzero_ps(); 8];
        let accumulator = [_mm256_setzero_ps(); 8];

        // Benchmark max performance version
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            outer_product_max_performance(a, b, &mut result);
        }
        let duration = start.elapsed();
        println!(
            "Max performance: {:.3} ms ({:.2} ops/sec)",
            duration.as_millis(),
            ITERATIONS as f64 / duration.as_secs_f64()
        );

        // Benchmark cache optimized version
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            outer_product_cache_optimized(a, b, &mut result);
        }
        let duration = start.elapsed();
        println!(
            "Cache optimized: {:.3} ms ({:.2} ops/sec)",
            duration.as_millis(),
            ITERATIONS as f64 / duration.as_secs_f64()
        );

        // Benchmark FMA version
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            outer_product_fma(a, b, &mut result, &accumulator);
        }
        let duration = start.elapsed();
        println!(
            "FMA version: {:.3} ms ({:.2} ops/sec)",
            duration.as_millis(),
            ITERATIONS as f64 / duration.as_secs_f64()
        );
    }
}

// Example usage and test
fn main() {
    // Check if AVX2 is available
    if !is_x86_feature_detected!("avx2") {
        println!("AVX2 not supported on this CPU");
        return;
    }

    println!("Maximum performance outer product benchmark in Rust:");
    println!("CPU features: AVX2, FMA");
    println!("Iterations: 10,000,000\n");

    // Test correctness
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    let result = outer_product_max_perf(a, b);

    println!("Sample outer product result (first 4x4):");
    for i in 0..4 {
        for j in 0..4 {
            print!("{:.3} ", result[i][j]);
        }
        println!();
    }
    println!();

    benchmark_all();
}
