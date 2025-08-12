//! Precision comparison tests between scalar and SIMD arcsine implementations.
//!
//! This test suite validates that the SIMD arcsine implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use std::f32::consts::PI;

use simdly::simd::SimdMath;

/// Test precision of SIMD asin against scalar asin for various input ranges.
#[test]
fn test_asin_precision_comparison() {
    // Test cases covering the valid domain [-1, 1] for asin
    let test_cases = vec![
        // Values near zero
        vec![0.0f32, 0.1, 0.2, 0.3],
        // Values near the middle
        vec![0.4f32, 0.5, 0.6, 0.7],
        // Values near the boundary
        vec![0.8f32, 0.9, 0.95, 0.99],
        // Exact boundary values
        vec![-1.0f32, -0.99, -0.95, -0.9],
        // Negative values near zero
        vec![-0.1f32, -0.2, -0.3, -0.4],
        // Negative values in middle
        vec![-0.5f32, -0.6, -0.7, -0.8],
        // Mixed values
        vec![-0.5f32, -0.25, 0.0, 0.25, 0.5],
        // Values that should give known results
        vec![0.5f32, std::f32::consts::FRAC_1_SQRT_2],
        // Boundary values
        vec![1.0f32, -1.0],
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
        println!("Testing case {}: {:?}", i + 1, test_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = test_case.iter().map(|x| x.asin()).collect();

        // Compute SIMD results
        let simd_results = test_case.asin();

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
            // asin can be sensitive near the boundaries
            assert!(
                absolute_error < 1e-4 || relative_error < 1e-4,
                "Precision error too large for input {input_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs in the valid domain.
#[test]
fn test_asin_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in valid domain [-1, 1] for asin
    let inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-1.0..=1.0))
        .collect();

    let scalar_results: Vec<f32> = inputs.iter().map(|x| x.asin()).collect();
    let simd_results = inputs.asin();

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
        if absolute_error > 1e-5 && relative_error > 1e-5 {
            error_count += 1;
            if error_count <= 10 {
                // Print first 10 significant errors
                println!(
                    "Large error #{error_count}: Input: {input_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs: {absolute_error:.2e}, Rel: {relative_error:.2e}"
                );
            }
        }

        // Individual assertion with reasonable tolerance for asin
        assert!(
            absolute_error < 1e-3 || relative_error < 1e-3,
            "Precision error too large at index {i}: input={input_val}, scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
        );
    }

    println!("Random precision test summary:");
    println!("  Test size: {test_size}");
    println!("  Max absolute error: {max_abs_error:.2e}");
    println!("  Max relative error: {max_rel_error:.2e}");
    println!("  Significant errors (>1e-5): {error_count}");

    // Overall precision requirements
    assert!(
        max_abs_error < 1e-3,
        "Maximum absolute error too large: {max_abs_error:.2e}"
    );
    assert!(
        max_rel_error < 1e-2,
        "Maximum relative error too large: {max_rel_error:.2e}"
    );
    assert!(
        error_count < test_size / 10,
        "Too many significant errors: {error_count}/{test_size}"
    );
}

/// Test edge cases and special values.
#[test]
fn test_asin_edge_cases() {
    let edge_cases = vec![
        // Exact values where asin should be known
        (0.0f32, 0.0f32),                           // asin(0) = 0
        (1.0f32, PI / 2.0),                         // asin(1) = π/2
        (-1.0f32, -PI / 2.0),                       // asin(-1) = -π/2
        (0.5f32, PI / 6.0),                         // asin(0.5) = π/6
        (-0.5f32, -PI / 6.0),                       // asin(-0.5) = -π/6
        (std::f32::consts::FRAC_1_SQRT_2, PI / 4.0), // asin(1/√2) = π/4
        (-std::f32::consts::FRAC_1_SQRT_2, -PI / 4.0), // asin(-1/√2) = -π/4
    ];

    for (input, expected) in edge_cases {
        let input_vec = vec![input];
        let scalar_result = input_vec.iter().map(|x| x.asin()).collect::<Vec<f32>>()[0];
        let simd_result = input_vec.asin()[0];

        println!("Edge case: asin({input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
        println!("  Scalar: {scalar_result:.8}, SIMD: {simd_result:.8}");

        // Compare against expected value (allowing for some numerical error)
        let scalar_error = (scalar_result - expected).abs();
        let simd_error = (simd_result - expected).abs();
        let scalar_rel_error = if expected != 0.0 { scalar_error / expected.abs() } else { scalar_error };
        let simd_rel_error = if expected != 0.0 { simd_error / expected.abs() } else { simd_error };

        println!("  Scalar error: {scalar_error:.2e} (rel: {scalar_rel_error:.2e}), SIMD error: {simd_error:.2e} (rel: {simd_rel_error:.2e})");

        // Both should be reasonably close to expected value
        if expected != 0.0 {
            assert!(
                scalar_rel_error < 1e-5,
                "Scalar asin relative error too large for {input}: {scalar_rel_error:.2e}"
            );
            assert!(
                simd_rel_error < 1e-4,
                "SIMD asin relative error too large for {input}: {simd_rel_error:.2e}"
            );
        } else {
            assert!(
                scalar_error < 1e-6,
                "Scalar asin absolute error too large for {input}: {scalar_error:.2e}"
            );
            assert!(
                simd_error < 1e-5,
                "SIMD asin absolute error too large for {input}: {simd_error:.2e}"
            );
        }

        // SIMD should be close to scalar
        let simd_vs_scalar_error = (simd_result - scalar_result).abs();
        let simd_vs_scalar_rel_error = if scalar_result != 0.0 { 
            simd_vs_scalar_error / scalar_result.abs() 
        } else { 
            simd_vs_scalar_error 
        };
        
        if scalar_result != 0.0 {
            assert!(
                simd_vs_scalar_rel_error < 1e-4,
                "SIMD vs scalar relative error too large for {input}: {simd_vs_scalar_rel_error:.2e}"
            );
        } else {
            assert!(
                simd_vs_scalar_error < 1e-5,
                "SIMD vs scalar absolute error too large for {input}: {simd_vs_scalar_error:.2e}"
            );
        }
    }
}

/// Test precision with very small values near zero.
#[test]
fn test_asin_precision_near_zero() {
    let small_values: Vec<f32> = vec![
        1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, -1e-8, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3,
        -1e-2, -1e-1,
    ];

    let scalar_results: Vec<f32> = small_values.iter().map(|x| x.asin()).collect();
    let simd_results = small_values.asin();

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = small_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Small value: asin({input_val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // For small values, asin(x) ≈ x, so both should be very close to the input
        assert!(
            (scalar_val - input_val).abs() < input_val.abs() * 0.01,
            "Scalar asin should be close to input for small values"
        );
        assert!(
            (simd_val - input_val).abs() < input_val.abs() * 0.01,
            "SIMD asin should be close to input for small values"
        );

        // SIMD should match scalar closely
        if scalar_val != 0.0 {
            assert!(
                relative_error < 1e-5,
                "SIMD precision error too large for small input {input_val}: {relative_error:.2e}"
            );
        } else {
            assert!(
                absolute_error < 1e-6,
                "SIMD precision error too large for small input {input_val}: {absolute_error:.2e}"
            );
        }
    }
}