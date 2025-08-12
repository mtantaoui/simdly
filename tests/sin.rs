//! Precision comparison tests between scalar and SIMD sine implementations.
//!
//! This test suite validates that the SIMD sine implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use std::f32::consts::{PI, TAU};

use simdly::simd::SimdMath;

/// Test precision of SIMD sine against scalar sine for various input ranges.
#[test]
fn test_sine_precision_comparison() {
    // Test cases covering different ranges and edge cases
    let test_cases = vec![
        // Small angles near zero
        vec![0.0f32, 0.1, 0.2, 0.3],
        // Quarter circle
        vec![0.5f32, 1.0, 1.2, 1.5],
        // Around π/2
        vec![1.4f32, 1.5, 1.57, 1.6],
        // Around π
        vec![3.0f32, 3.1, PI, 3.2],
        // Around 3π/2
        vec![4.5f32, 4.7, 4.71, 4.8],
        // Around 2π
        vec![6.0f32, 6.2, TAU, 6.3],
        // Negative values
        vec![-0.5f32, -1.0, -1.57, -PI],
        // Larger values
        vec![10.0f32, 15.0, 20.0, 25.0],
        // Mixed range
        vec![-10.0f32, -5.0, 0.0, 5.0],
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
        println!("Testing case {}: {:?}", i + 1, test_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = test_case.iter().map(|x| x.sin()).collect();

        // Compute SIMD results
        let simd_results = test_case.sin();

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
            // SIMD implementations may use polynomial approximations with slightly different precision
            assert!(
                absolute_error < 1e-5 || relative_error < 1e-5,
                "Precision error too large for input {input_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs across the full range.
#[test]
fn test_sine_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in range [-4π, 4π] to test various periods
    let inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-4.0 * std::f32::consts::PI..=4.0 * std::f32::consts::PI))
        .collect();

    let scalar_results: Vec<f32> = inputs.iter().map(|x| x.sin()).collect();
    let simd_results = inputs.sin();

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

        // Individual assertion with relaxed tolerance for SIMD approximations
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
        max_rel_error < 2.1e-4,
        "Maximum relative error too large: {max_rel_error:.2e}"
    );
    assert!(
        error_count < test_size / 10,
        "Too many significant errors: {error_count}/{test_size}"
    );
}

/// Test edge cases and special values.
#[test]
fn test_sine_edge_cases() {
    let edge_cases = vec![
        // Exact values where sine should be known
        (0.0f32, 0.0f32),                            // sin(0) = 0
        (std::f32::consts::PI / 2.0, 1.0f32),        // sin(π/2) = 1
        (std::f32::consts::PI, 0.0f32),              // sin(π) = 0
        (3.0 * std::f32::consts::PI / 2.0, -1.0f32), // sin(3π/2) = -1
        (2.0 * std::f32::consts::PI, 0.0f32),        // sin(2π) = 0
    ];

    for (input, expected) in edge_cases {
        let input_vec = vec![input];
        let scalar_result = input_vec.iter().map(|x| x.sin()).collect::<Vec<f32>>()[0];
        let simd_result = input_vec.sin()[0];

        println!("Edge case: sin({input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
        println!("  Scalar: {scalar_result:.8}, SIMD: {simd_result:.8}");

        // Compare against expected value (allowing for some numerical error)
        let scalar_error = (scalar_result - expected).abs();
        let simd_error = (simd_result - expected).abs();

        println!("  Scalar error: {scalar_error:.2e}, SIMD error: {simd_error:.2e}");

        // Both should be reasonably close to expected value
        assert!(
            scalar_error < 1e-6,
            "Scalar sine error too large for {input}: {scalar_error:.2e}"
        );
        assert!(
            simd_error < 1e-5,
            "SIMD sine error too large for {input}: {simd_error:.2e}"
        );

        // SIMD should be close to scalar
        let simd_vs_scalar_error = (simd_result - scalar_result).abs();
        assert!(
            simd_vs_scalar_error < 1e-5,
            "SIMD vs scalar error too large for {input}: {simd_vs_scalar_error:.2e}"
        );
    }
}

/// Test precision with very small values near zero.
#[test]
fn test_sine_precision_near_zero() {
    let small_values: Vec<f32> = vec![
        1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, -1e-8, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3,
        -1e-2, -1e-1,
    ];

    let scalar_results: Vec<f32> = small_values.iter().map(|x| x.sin()).collect();
    let simd_results = small_values.sin();

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = small_values[i];
        let absolute_error = (scalar_val - simd_val).abs();

        println!(
            "Small value: sin({input_val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}"
        );

        // For small values, sin(x) ≈ x, so both should be very close to the input
        assert!(
            (scalar_val - input_val).abs() < input_val.abs() * 0.1,
            "Scalar sine should be close to input for small values"
        );
        assert!(
            (simd_val - input_val).abs() < input_val.abs() * 0.1,
            "SIMD sine should be close to input for small values"
        );

        // SIMD should match scalar closely
        assert!(
            absolute_error < 1e-6,
            "SIMD precision error too large for small input {input_val}: {absolute_error:.2e}"
        );
    }
}
