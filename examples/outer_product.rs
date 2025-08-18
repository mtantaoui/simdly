use std::arch::x86_64::*;
use std::time::Instant;

/// High-performance outer product implementation using AVX2 SIMD instructions.
///
/// Computes the outer product of two 8-element vectors: result[i][j] = a[i] * b[j]
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

/// Naive reference implementation for comparison
fn naive_outer_product(a: [f32; 8], b: [f32; 8]) -> [[f32; 8]; 8] {
    let mut result = [[0.0f32; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            result[i][j] = a[i] * b[j];
        }
    }
    result
}

/// Performance benchmark
fn benchmark_outer_product() {
    const ITERATIONS: usize = 1_000_000;
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    // Benchmark SIMD version
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _result = outer_product_f32_8x8(a, b);
    }
    let simd_time = start.elapsed();

    // Benchmark naive version
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _result = naive_outer_product(a, b);
    }
    let naive_time = start.elapsed();

    println!("SIMD version: {:.1} ms", simd_time.as_millis());
    println!("Naive version: {:.1} ms", naive_time.as_millis());
    println!("Speedup: {:.1}x", naive_time.as_secs_f64() / simd_time.as_secs_f64());
}

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

    let result = outer_product_f32_8x8(a, b);
    
    println!("\nOuter product result (first 4x4 submatrix):");
    for i in 0..4 {
        for j in 0..4 {
            print!("{:6.2} ", result[i][j]);
        }
        println!();
    }

    // Verify correctness
    println!("\nVerification:");
    println!("result[0][0] = {} (expected: {})", result[0][0], a[0] * b[0]);
    println!("result[1][2] = {} (expected: {})", result[1][2], a[1] * b[2]);
    println!("result[7][7] = {} (expected: {})", result[7][7], a[7] * b[7]);

    // Performance comparison
    println!("\n=== Performance Benchmark ===");
    benchmark_outer_product();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < f32::EPSILON * 10.0
    }

    #[test]
    fn test_outer_product_correctness() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        let simd_result = outer_product_f32_8x8(a, b);
        let naive_result = naive_outer_product(a, b);

        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    approx_eq(simd_result[i][j], naive_result[i][j]),
                    "Mismatch at [{}, {}]: {} vs {}",
                    i, j, simd_result[i][j], naive_result[i][j]
                );
            }
        }
    }

    #[test]
    fn test_specific_values() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a = [2.0, 3.0, 1.0, 4.0, 0.5, 1.5, 2.5, 3.5];
        let b = [1.0, 0.0, 2.0, 0.5, 3.0, 1.5, 0.25, 4.0];

        let result = outer_product_f32_8x8(a, b);

        assert!(approx_eq(result[0][0], 2.0)); // 2.0 * 1.0
        assert!(approx_eq(result[0][2], 4.0)); // 2.0 * 2.0
        assert!(approx_eq(result[1][3], 1.5)); // 3.0 * 0.5
        assert!(approx_eq(result[7][7], 14.0)); // 3.5 * 4.0
    }

    #[test]
    fn test_zero_vector() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a = [0.0; 8];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result = outer_product_f32_8x8(a, b);

        for i in 0..8 {
            for j in 0..8 {
                assert!(approx_eq(result[i][j], 0.0));
            }
        }
    }

    #[test]
    fn test_identity_vector() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let result = outer_product_f32_8x8(a, b);

        // First row should equal vector b
        for j in 0..8 {
            assert!(approx_eq(result[0][j], b[j]));
        }

        // Other rows should be zero
        for i in 1..8 {
            for j in 0..8 {
                assert!(approx_eq(result[i][j], 0.0));
            }
        }
    }
}