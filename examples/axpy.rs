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

/// Demonstrates the FMA signature and how it applies to AXPY operations.
/// 
/// Shows that both scalar and SIMD implementations produce identical results
/// when computing y = alpha * x + y using different approaches.
fn demo_fma_usage() {
    println!("=== FMA Usage Demo with AXPY ===\n");

    let alpha = 2.5;
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y_scalar = vec![0.5, 1.0, 1.5, 2.0];
    let y_simd = vec![0.5, 1.0, 1.5, 2.0];

    println!("AXPY: y = alpha * x + y");
    println!("alpha = {}", alpha);
    println!("x = {:?}", x);
    println!("y (initial) = {:?}", y_simd);

    // Scalar version
    axpy_scalar(alpha, &x, &mut y_scalar);
    println!("\nScalar result: {:?}", y_scalar);

    // SIMD version using new FMA signature
    let y_simd_result = axpy_simd(alpha, x.clone(), y_simd.clone());
    println!("SIMD result:   {:?}", y_simd_result);

    // Verify they match
    assert_eq!(y_scalar, y_simd_result);
    println!("✓ Scalar and SIMD results match!");

    // Show the computation step by step
    println!("\nStep-by-step computation:");
    for i in 0..x.len() {
        println!(
            "y[{}] = {} * {} + {} = {}",
            i, alpha, x[i], y_simd[i], y_scalar[i]
        );
    }

    println!("\n✓ FMA signature demonstration completed!\n");
}

/// Test FMA accumulation pattern
fn test_fma_accumulation() {
    println!("=== FMA Accumulation Pattern ===\n");

    let a = vec![2.0, 3.0];
    let b = vec![0.5, 1.0];
    let mut accumulator = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix: [[1,2], [3,4]]

    println!("Computing outer product accumulation using FMA");
    println!("Initial matrix (2x2):");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:4.1} ", accumulator[i * 2 + j]);
        }
        println!();
    }

    // Simulate outer product accumulation: result[i][j] += a[i] * b[j]
    for i in 0..a.len() {
        for j in 0..b.len() {
            let idx = i * b.len() + j;
            let old_value = accumulator[idx];

            // New FMA usage: accumulator.fma(multiplier, multiplicand) = multiplier * multiplicand + accumulator
            let a_scalar = vec![a[i]];
            let b_scalar = vec![b[j]];
            let acc_scalar = vec![old_value];
            let result = acc_scalar.fma(a_scalar, b_scalar);
            accumulator[idx] = result[0];

            println!(
                "acc[{}][{}] = acc.fma(a[{}], b[{}]) = {} * {} + {} = {}",
                i, j, i, j, a[i], b[j], old_value, accumulator[idx]
            );
        }
    }

    println!("\nFinal accumulated matrix:");
    for i in 0..2 {
        for j in 0..2 {
            print!("{:4.1} ", accumulator[i * 2 + j]);
        }
        println!();
    }

    // Verify manually:
    // acc[0][0] = 2.0 * 0.5 + 1.0 = 2.0
    // acc[0][1] = 2.0 * 1.0 + 2.0 = 4.0
    // acc[1][0] = 3.0 * 0.5 + 3.0 = 4.5
    // acc[1][1] = 3.0 * 1.0 + 4.0 = 7.0
    let expected = vec![2.0, 4.0, 4.5, 7.0];
    assert_eq!(accumulator, expected);

    println!("✓ FMA accumulation test passed!\n");
}

/// Test different FMA usage patterns
fn test_fma_patterns() {
    println!("=== FMA Usage Patterns ===\n");

    // Note: We're using platform-agnostic SIMD operations

    // Pattern 1: Standard FMA with vectors
    println!("Pattern 1: Vector FMA");
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 3.0, 4.0, 5.0];
    let c = vec![0.5, 0.5, 0.5, 0.5];

    let result1 = c.fma(a.clone(), b.clone());
    println!("c.fma(a, b) = a * b + c = {:?}", result1);

    // Pattern 2: Broadcasting scalar
    println!("\nPattern 2: Scalar broadcast");
    let scalar = vec![2.0; 4];
    let result2 = c.fma(scalar, a);
    println!("c.fma(scalar, a) = scalar * a + c = {:?}", result2);

    // Pattern 3: In-place accumulation simulation
    println!("\nPattern 3: Accumulation simulation");
    let mut acc = vec![10.0, 20.0, 30.0, 40.0];
    let factor1 = vec![0.1, 0.2, 0.3, 0.4];
    let factor2 = vec![5.0, 5.0, 5.0, 5.0];

    println!("Before: acc = {:?}", acc);
    acc = acc.fma(factor1, factor2);
    println!("After acc.fma(f1, f2): {:?}", acc);

    println!("✓ All FMA patterns tested successfully!\n");
}

fn main() {
    println!("🧮 AXPY and FMA Demonstration\n");

    demo_fma_usage();
    test_fma_accumulation();
    test_fma_patterns();

    println!("✅ All AXPY and FMA tests completed successfully! 🎉");
}
