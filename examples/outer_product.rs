use std::arch::x86_64::*;
// use std::time::Instant;

/// High-performance outer product implementation using AVX2 SIMD instructions.
///
/// Computes the outer product of two 8-element vectors: result[i][j] = a[i] * b[j]
///
/// # Safety
///
/// This function is unsafe because it uses AVX2 intrinsics. The caller must ensure:
///
/// - **CPU Support**: The target CPU must support AVX2 instructions. Use
///   `is_x86_feature_detected!("avx2")` to check before calling.
/// - **Vector Validity**: Input vectors `a` and `b` must contain valid f32 values
///   (not uninitialized memory).
/// - **Result Array**: The `result` array must be properly initialized and have
///   exactly 8 elements.
/// - **Memory Safety**: All memory accessed through the vectors and result array
///   must be valid and properly aligned (though unaligned access is supported).
/// - **No Data Races**: This function is not thread-safe for the same `result` array.
///   Concurrent access to the same result array from multiple threads will cause
///   undefined behavior.
///
/// # Arguments
///
/// * `a` - First input vector as __m256 (8 f32 elements)
/// * `b` - Second input vector as __m256 (8 f32 elements)  
/// * `result` - Output array of 8 __m256 vectors representing the 8x8 result matrix
#[target_feature(enable = "avx2")]
pub unsafe fn simd_outer_product(a: __m256, b: __m256, result: &mut [__m256; 8]) {
    // Split vector 'a' into lower and upper halves for broadcasting
    let a_lo = _mm256_permute2f128_ps::<0x00>(a, a); // [a0,a1,a2,a3,a0,a1,a2,a3]
    let a_hi = _mm256_permute2f128_ps::<0x11>(a, a); // [a4,a5,a6,a7,a4,a5,a6,a7]

    // Broadcast each element of 'a' and multiply with 'b' to get each row
    result[0] = _mm256_mul_ps(_mm256_permute_ps::<0x00>(a_lo), b); // a[0] * b
    result[1] = _mm256_mul_ps(_mm256_permute_ps::<0x55>(a_lo), b); // a[1] * b
    result[2] = _mm256_mul_ps(_mm256_permute_ps::<0xAA>(a_lo), b); // a[2] * b
    result[3] = _mm256_mul_ps(_mm256_permute_ps::<0xFF>(a_lo), b); // a[3] * b
    result[4] = _mm256_mul_ps(_mm256_permute_ps::<0x00>(a_hi), b); // a[4] * b
    result[5] = _mm256_mul_ps(_mm256_permute_ps::<0x55>(a_hi), b); // a[5] * b
    result[6] = _mm256_mul_ps(_mm256_permute_ps::<0xAA>(a_hi), b); // a[6] * b
    result[7] = _mm256_mul_ps(_mm256_permute_ps::<0xFF>(a_hi), b); // a[7] * b
}

/// Safe wrapper for computing outer product of two f32 arrays.
///
/// # Arguments
/// * `a` - First input vector (8 f32 elements)
/// * `b` - Second input vector (8 f32 elements)
///
/// # Returns
/// 8x8 matrix where result[i][j] = a[i] * b[j]
pub fn outer_product_f32_8x8(a: [f32; 8], b: [f32; 8]) -> [[f32; 8]; 8] {
    if !is_x86_feature_detected!("avx2") {
        panic!("AVX2 instruction set not supported on this CPU");
    }

    unsafe {
        let a_vec = _mm256_loadu_ps(a.as_ptr());
        let b_vec = _mm256_loadu_ps(b.as_ptr());
        let mut result = [_mm256_setzero_ps(); 8];

        simd_outer_product(a_vec, b_vec, &mut result);

        let mut output = [[0.0f32; 8]; 8];
        for i in 0..8 {
            _mm256_storeu_ps(output[i].as_mut_ptr(), result[i]);
        }
        output
    }
}

// /// Naive reference implementation for comparison
// fn naive_outer_product(a: [f32; 8], b: [f32; 8]) -> [[f32; 8]; 8] {
//     let mut result = [[0.0f32; 8]; 8];
//     for i in 0..8 {
//         for j in 0..8 {
//             result[i][j] = a[i] * b[j];
//         }
//     }
//     result
// }

// /// Performance benchmark
// fn benchmark_outer_product() {
//     const ITERATIONS: usize = 1_000_000;
//     let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//     let b = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

//     // Benchmark SIMD version
//     let start = Instant::now();
//     for _ in 0..ITERATIONS {
//         let _result = outer_product_f32_8x8(a, b);
//     }
//     let simd_time = start.elapsed();

//     // Benchmark naive version
//     let start = Instant::now();
//     for _ in 0..ITERATIONS {
//         let _result = naive_outer_product(a, b);
//     }
//     let naive_time = start.elapsed();

//     println!("SIMD version: {:.1} ms", simd_time.as_millis());
//     println!("Naive version: {:.1} ms", naive_time.as_millis());
//     println!(
//         "Speedup: {:.1}x",
//         naive_time.as_secs_f64() / simd_time.as_secs_f64()
//     );
// }

fn main() {
    if !is_x86_feature_detected!("avx2") {
        println!("AVX2 not supported on this CPU");
        return;
    }

    println!("=== AVX2 Outer Product Example ===\n");

    // Basic example
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    println!("Vector a: {:?}", a);
    println!("Vector b: {:?}", b);

    let _result = outer_product_f32_8x8(a, b);

    // println!("\nOuter product result (first 4x4 submatrix):");
    // for i in 0..4 {
    //     for j in 0..4 {
    //         print!("{:6.2} ", result[i][j]);
    //     }
    //     println!();
    // }

    // // Verify correctness
    // println!("\nVerification:");
    // println!(
    //     "result[0][0] = {} (expected: {})",
    //     result[0][0],
    //     a[0] * b[0]
    // );
    // println!(
    //     "result[1][2] = {} (expected: {})",
    //     result[1][2],
    //     a[1] * b[2]
    // );
    // println!(
    //     "result[7][7] = {} (expected: {})",
    //     result[7][7],
    //     a[7] * b[7]
    // );

    // // Performance comparison
    // println!("\n=== Performance Benchmark ===");
    // benchmark_outer_product();
}
