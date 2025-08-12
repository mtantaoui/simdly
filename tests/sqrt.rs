//! Precision comparison tests between scalar and SIMD square root implementations.
//!
//! This test suite validates that the SIMD square root implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use simdly::simd::SimdMath;

/// Test precision of SIMD sqrt against scalar sqrt for various input ranges.
#[test]
fn test_sqrt_precision_comparison() {
    // Test cases covering different ranges and edge cases
    // Only positive values as sqrt is undefined for negative inputs
    let test_cases = [
        // Small positive values
        vec![0.0f32, 0.01, 0.1, 0.25],
        // Unit values
        vec![1.0f32, 2.0, 3.0, 4.0],
        // Perfect squares
        vec![9.0f32, 16.0, 25.0, 36.0],
        // Larger values
        vec![100.0f32, 256.0, 1024.0, 10000.0],
        // Very small positive values
        vec![1e-8f32, 1e-6, 1e-4, 1e-2],
        // Random positive values
        vec![1.5f32, 2.7, 5.3, 7.8],
        // Large values
        vec![50000.0f32, 100000.0, 1000000.0],
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
        println!("Testing case {}: {:?}", i + 1, test_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = test_case.iter().map(|x| x.sqrt()).collect();

        // Compute SIMD results
        let simd_results = test_case.sqrt();

        // Compare results element by element
        assert_eq!(
            scalar_results.len(),
            simd_results.len(),
            "Result vectors have different lengths"
        );

        for (j, (&scalar_val, &simd_val)) in
            scalar_results.iter().zip(simd_results.iter()).enumerate()
        {
            let input_val = test_case[j];
            let absolute_error = (scalar_val - simd_val).abs();
            let relative_error = if scalar_val != 0.0 {
                absolute_error / scalar_val.abs()
            } else {
                absolute_error
            };

            println!(
                "  Input: {input_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs Error: {absolute_error:.2e}, Rel Error: {relative_error:.2e}"
            );

            // Assertion with reasonable tolerance for f32 precision
            // sqrt implementations should be very accurate
            assert!(
                absolute_error < 1e-5 || relative_error < 1e-5,
                "Precision error too large for input {input_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated positive inputs.
#[test]
fn test_sqrt_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random positive inputs in range [0, 1000000]
    let inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random::<f32>() * 1000000.0)
        .collect();

    let scalar_results: Vec<f32> = inputs.iter().map(|x| x.sqrt()).collect();
    let simd_results = inputs.sqrt();

    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut error_count = 0;

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = inputs[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        max_abs_error = max_abs_error.max(absolute_error);
        max_rel_error = max_rel_error.max(relative_error);

        // Count significant errors
        if absolute_error > 1e-6 && relative_error > 1e-6 {
            error_count += 1;
            if error_count <= 10 {
                // Print first 10 significant errors
                println!(
                    "Large error #{error_count}: Input: {input_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs: {absolute_error:.2e}, Rel: {relative_error:.2e}"
                );
            }
        }

        // Individual assertion with reasonable tolerance for sqrt
        assert!(
            absolute_error < 1e-4 || relative_error < 1e-4,
            "Precision error too large at index {i}: input={input_val}, scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
        );
    }

    println!("Random precision test summary:");
    println!("  Test size: {test_size}");
    println!("  Max absolute error: {max_abs_error:.2e}");
    println!("  Max relative error: {max_rel_error:.2e}");
    println!("  Significant errors (>1e-6): {error_count}");

    // Overall precision requirements
    assert!(
        max_abs_error < 1e-4,
        "Maximum absolute error too large: {max_abs_error:.2e}"
    );
    assert!(
        max_rel_error < 1e-4,
        "Maximum relative error too large: {max_rel_error:.2e}"
    );
    assert!(
        error_count < test_size / 10,
        "Too many significant errors: {error_count}/{test_size}"
    );
}

/// Test edge cases and special values.
#[test]
fn test_sqrt_edge_cases() {
    let edge_cases = vec![
        // Exact values where sqrt should be known
        (0.0f32, 0.0f32),    // sqrt(0) = 0
        (1.0f32, 1.0f32),    // sqrt(1) = 1
        (4.0f32, 2.0f32),    // sqrt(4) = 2
        (9.0f32, 3.0f32),    // sqrt(9) = 3
        (16.0f32, 4.0f32),   // sqrt(16) = 4
        (25.0f32, 5.0f32),   // sqrt(25) = 5
        (100.0f32, 10.0f32), // sqrt(100) = 10
    ];

    for (input, expected) in edge_cases {
        let input_vec = vec![input];
        let scalar_result = input_vec.iter().map(|x| x.sqrt()).collect::<Vec<f32>>()[0];
        let simd_result = input_vec.sqrt()[0];

        println!("Edge case: sqrt({input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
        println!("  Scalar: {scalar_result:.8}, SIMD: {simd_result:.8}");

        // Compare against expected value (allowing for minimal numerical error)
        let scalar_error = (scalar_result - expected).abs();
        let simd_error = (simd_result - expected).abs();

        println!("  Scalar error: {scalar_error:.2e}, SIMD error: {simd_error:.2e}");

        // Both should be very close to expected value
        assert!(
            scalar_error < 1e-6,
            "Scalar sqrt error too large for {input}: {scalar_error:.2e}"
        );
        assert!(
            simd_error < 1e-5,
            "SIMD sqrt error too large for {input}: {simd_error:.2e}"
        );

        // SIMD should be close to scalar
        let simd_vs_scalar_error = (simd_result - scalar_result).abs();
        assert!(
            simd_vs_scalar_error < 1e-5,
            "SIMD vs scalar error too large for {input}: {simd_vs_scalar_error:.2e}"
        );
    }
}

/// Test precision with very small positive values near zero.
#[test]
fn test_sqrt_precision_near_zero() {
    let small_values: Vec<f32> = vec![
        1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.01, 0.1, 0.25, 0.5,
    ];

    let scalar_results: Vec<f32> = small_values.iter().map(|x| x.sqrt()).collect();
    let simd_results = small_values.sqrt();

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = small_values[i];
        let absolute_error = (scalar_val - simd_val).abs();

        println!(
            "Small value: sqrt({input_val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}"
        );

        // For small positive values, sqrt should be accurate
        let expected = input_val.sqrt();
        assert!(
            (scalar_val - expected).abs() < 1e-7,
            "Scalar sqrt should be accurate for small positive input"
        );
        assert!(
            (simd_val - expected).abs() < 1e-6,
            "SIMD sqrt should be accurate for small positive input"
        );

        // SIMD should match scalar closely
        assert!(
            absolute_error < 1e-6,
            "SIMD precision error too large for small input {input_val}: {absolute_error:.2e}"
        );
    }
}
