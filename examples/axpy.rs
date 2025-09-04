use simdly::simd::SimdMath;

/// AXPY operation: y = alpha * x + y (scalar implementation)
///
/// Performs the AXPY (A times X Plus Y) operation where:
/// - alpha: scalar multiplier
/// - x: input vector  
/// - y: input/output vector (modified in-place)
///
/// This is the scalar reference implementation for comparison with SIMD version.
fn axpy_scalar(alpha: f32, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    for i in 0..x.len() {
        y[i] = alpha * x[i] + y[i]; // y = alpha * x + y
    }
}

/// AXPY operation using SIMD FMA (vectorized implementation)
///
/// Performs vectorized AXPY operation using SIMD instructions.
/// Uses the FMA operation: y.fma(alpha_vec, x) = alpha_vec * x + y
/// where y acts as the accumulator (addend) in the FMA operation.
fn axpy_simd(alpha: f32, x: Vec<f32>, y: Vec<f32>) -> Vec<f32> {
    assert_eq!(x.len(), y.len(), "Vectors must have same length");

    // Create alpha vector for broadcasting
    let alpha_vec = vec![alpha; x.len()];

    // Use FMA: y.fma(alpha_vec, x) = alpha_vec * x + y
    // This computes: alpha * x + y (where y is the addend)
    y.fma(alpha_vec, x)
}

fn main() {
    println!("AXPY (A*X Plus Y) Example");
    println!("========================\n");

    // Example vectors
    let alpha = 2.5;
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![0.5, 1.0, 1.5, 2.0];

    println!("alpha = {}", alpha);
    println!("x = {:?}", x);
    println!("y = {:?}", y);

    // Compute AXPY: alpha * x + y
    let result = axpy_simd(alpha, x.clone(), y.clone());

    println!("\nResult (alpha * x + y):");
    println!("result = {:?}", result);

    // Show step-by-step computation
    println!("\nStep-by-step:");
    for i in 0..x.len() {
        println!("  {} * {} + {} = {}", alpha, x[i], y[i], result[i]);
    }

    println!("\n✓ AXPY computation completed!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axpy_basic() {
        let alpha = 2.0;
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = axpy(alpha, x, y);
        let expected = vec![6.0, 9.0, 12.0]; // 2*1+4, 2*2+5, 2*3+6

        assert_eq!(result, expected);
    }

    #[test]
    fn test_axpy_zero_alpha() {
        let alpha = 0.0;
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];

        let result = axpy(alpha, x, y.clone());

        // When alpha is 0, result should be y
        assert_eq!(result, y);
    }

    #[test]
    fn test_axpy_negative_values() {
        let alpha = -1.5;
        let x = vec![2.0, -1.0, 3.0];
        let y = vec![1.0, 2.0, -1.0];

        let result = axpy(alpha, x, y);
        let expected = vec![-2.0, 3.5, -5.5]; // -1.5*2+1, -1.5*(-1)+2, -1.5*3+(-1)

        assert_eq!(result, expected);
    }

    #[test]
    fn test_axpy_empty_vectors() {
        let alpha = 2.0;
        let x: Vec<f32> = vec![];
        let y: Vec<f32> = vec![];

        let result = axpy(alpha, x, y);

        assert!(result.is_empty());
    }

    #[test]
    fn test_axpy_single_element() {
        let alpha = 3.0;
        let x = vec![4.0];
        let y = vec![1.0];

        let result = axpy(alpha, x, y);
        let expected = vec![13.0]; // 3*4+1

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Vectors must have same length")]
    fn test_axpy_mismatched_lengths() {
        let alpha = 1.0;
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];

        axpy(alpha, x, y);
    }
}
