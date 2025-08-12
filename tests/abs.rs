//! Precision comparison tests between scalar and SIMD absolute value implementations.
//!
//! This test suite validates that the SIMD absolute value implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use simdly::simd::SimdMath;

/// Test precision of SIMD abs against scalar abs for various input ranges.
#[test]
fn test_abs_precision_comparison() {
    // Test cases covering different ranges and edge cases
    let test_cases = [
        // Small positive values
        vec![0.0f32, 0.1, 0.2, 0.3],
        // Regular positive values
        vec![1.0f32, 5.0, 10.0, 25.0],
        // Large positive values
        vec![100.0f32, 1000.0, 10000.0, 100000.0],
        // Small negative values
        vec![-0.1f32, -0.2, -0.3, -0.5],
        // Regular negative values
        vec![-1.0f32, -5.0, -10.0, -25.0],
        // Large negative values
        vec![-100.0f32, -1000.0, -10000.0, -100000.0],
        // Mixed positive and negative
        vec![-10.0f32, -1.0, 0.0, 1.0, 10.0],
        // Very small values near zero
        vec![-1e-6f32, -1e-7, 1e-7, 1e-6],
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
        println!("Testing case {}: {:?}", i + 1, test_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = test_case.iter().map(|x| x.abs()).collect();

        // Compute SIMD results
        let simd_results = test_case.abs();

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
            // abs() should be exact for floating point values
            assert!(
                absolute_error < 1e-6 || relative_error < 1e-6,
                "Precision error too large for input {input_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs across the full range.
#[test]
fn test_abs_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in range [-1000000, 1000000] to test various magnitudes
    let inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-1000000.0..=1000000.0))
        .collect();

    let scalar_results: Vec<f32> = inputs.iter().map(|x| x.abs()).collect();
    let simd_results = inputs.abs();

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

        // Count significant errors (should be zero for abs function)
        if absolute_error > 1e-7 && relative_error > 1e-7 {
            error_count += 1;
            if error_count <= 10 {
                // Print first 10 significant errors
                println!(
                    "Large error #{error_count}: Input: {input_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs: {absolute_error:.2e}, Rel: {relative_error:.2e}"
                );
            }
        }

        // Individual assertion with very strict tolerance for abs function
        assert!(
            absolute_error < 1e-6,
            "Precision error too large at index {i}: input={input_val}, scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
        );
    }

    println!("Random precision test summary:");
    println!("  Test size: {test_size}");
    println!("  Max absolute error: {max_abs_error:.2e}");
    println!("  Max relative error: {max_rel_error:.2e}");
    println!("  Significant errors (>1e-7): {error_count}");

    // Overall precision requirements - abs should be exact
    assert!(
        max_abs_error < 1e-6,
        "Maximum absolute error too large: {max_abs_error:.2e}"
    );
    assert!(
        max_rel_error < 1e-6,
        "Maximum relative error too large: {max_rel_error:.2e}"
    );
    assert!(
        error_count == 0,
        "Any significant errors are unacceptable for abs function: {error_count}/{test_size}"
    );
}

/// Test edge cases and special values.
#[test]
fn test_abs_edge_cases() {
    let edge_cases = vec![
        // Exact values where absolute value should be known
        (0.0f32, 0.0f32),    // abs(0) = 0
        (1.0f32, 1.0f32),    // abs(1) = 1
        (-1.0f32, 1.0f32),   // abs(-1) = 1
        (42.0f32, 42.0f32),  // abs(42) = 42
        (-42.0f32, 42.0f32), // abs(-42) = 42
    ];

    for (input, expected) in edge_cases {
        let input_vec = vec![input];
        let scalar_result = input_vec.iter().map(|x| x.abs()).collect::<Vec<f32>>()[0];
        let simd_result = input_vec.abs()[0];

        println!("Edge case: abs({input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
        println!("  Scalar: {scalar_result:.8}, SIMD: {simd_result:.8}");

        // Compare against expected value (should be exact)
        let scalar_error = (scalar_result - expected).abs();
        let simd_error = (simd_result - expected).abs();

        println!("  Scalar error: {scalar_error:.2e}, SIMD error: {simd_error:.2e}");

        // Both should be exactly equal to expected value
        assert!(
            scalar_error < 1e-7,
            "Scalar abs error too large for {input}: {scalar_error:.2e}"
        );
        assert!(
            simd_error < 1e-7,
            "SIMD abs error too large for {input}: {simd_error:.2e}"
        );

        // SIMD should be exactly equal to scalar
        let simd_vs_scalar_error = (simd_result - scalar_result).abs();
        assert!(
            simd_vs_scalar_error < 1e-7,
            "SIMD vs scalar error too large for {input}: {simd_vs_scalar_error:.2e}"
        );
    }
}

/// Test precision with very small values near zero.
#[test]
fn test_abs_precision_near_zero() {
    let small_values: Vec<f32> = vec![
        1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, -1e-8, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3,
        -1e-2, -1e-1,
    ];

    let scalar_results: Vec<f32> = small_values.iter().map(|x| x.abs()).collect();
    let simd_results = small_values.abs();

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = small_values[i];
        let absolute_error = (scalar_val - simd_val).abs();

        println!(
            "Small value: abs({input_val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}"
        );

        // For small values, abs(x) should equal |x| exactly
        let expected = input_val.abs();
        assert!(
            (scalar_val - expected).abs() < 1e-8,
            "Scalar abs should be exact for small input"
        );
        assert!(
            (simd_val - expected).abs() < 1e-8,
            "SIMD abs should be exact for small input"
        );

        // SIMD should match scalar exactly
        assert!(
            absolute_error < 1e-8,
            "SIMD precision error too large for small input {input_val}: {absolute_error:.2e}"
        );
    }
}
