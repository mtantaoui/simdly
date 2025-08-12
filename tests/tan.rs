//! Precision comparison tests between scalar and SIMD tangent implementations.
//!
//! This test suite validates that the SIMD tangent implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use std::f32::consts::{PI, TAU};

use simdly::simd::SimdMath;

/// Test precision of SIMD tangent against scalar tangent for various input ranges.
#[test]
fn test_tangent_precision_comparison() {
    // Test cases covering different ranges and edge cases
    // Avoiding values close to π/2 + nπ where tangent is undefined
    let test_cases = [
        // Small angles near zero
        vec![0.0f32, 0.1, 0.2, 0.3],
        // Quarter circle (avoiding π/2)
        vec![0.5f32, 1.0, 1.2, 1.4],
        // Around π (avoiding 3π/2)
        vec![3.0f32, 3.1, PI, 3.3],
        // Around 2π
        vec![6.0f32, 6.2, TAU, 6.4],
        // Negative values
        vec![-0.5f32, -1.0, -1.4, -3.0],
        // Larger values (avoiding singularities)
        vec![7.0f32, 8.0, 9.0, 10.0],
        // Mixed range
        vec![-2.0f32, -1.0, 0.0, 1.0, 2.0],
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
        println!("Testing case {}: {:?}", i + 1, test_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = test_case.iter().map(|x| x.tan()).collect();

        // Compute SIMD results
        let simd_results = test_case.tan();

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
            // Tangent can be sensitive near singularities, so we use relaxed tolerances
            assert!(
                absolute_error < 1e-4 || relative_error < 1e-4,
                "Precision error too large for input {input_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs across safe ranges.
#[test]
fn test_tangent_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs avoiding singularities (π/2 + nπ)
    // Use range [-1.4, 1.4] to avoid first singularity at π/2 ≈ 1.57
    let inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-1.4..=1.4))
        .collect();

    let scalar_results: Vec<f32> = inputs.iter().map(|x| x.tan()).collect();
    let simd_results = inputs.tan();

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

        // Individual assertion with relaxed tolerance for SIMD approximations
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

    // Overall precision requirements - relaxed for tangent due to sensitivity
    assert!(
        max_abs_error < 1e-3,
        "Maximum absolute error too large: {max_abs_error:.2e}"
    );
    assert!(
        max_rel_error < 1e-2,
        "Maximum relative error too large: {max_rel_error:.2e}"
    );
    assert!(
        error_count < test_size / 5,
        "Too many significant errors: {error_count}/{test_size}"
    );
}

/// Test edge cases and special values.
#[test]
fn test_tangent_edge_cases() {
    let edge_cases = vec![
        // Exact values where tangent should be known
        (0.0f32, 0.0f32),               // tan(0) = 0
        (std::f32::consts::PI, 0.0f32), // tan(π) = 0
        (2.0 * std::f32::consts::PI, 0.0f32), // tan(2π) = 0
                                        // Note: Avoiding π/2 and 3π/2 where tangent is undefined
    ];

    for (input, expected) in edge_cases {
        let input_vec = vec![input];
        let scalar_result = input_vec.iter().map(|x| x.tan()).collect::<Vec<f32>>()[0];
        let simd_result = input_vec.tan()[0];

        println!("Edge case: tan({input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
        println!("  Scalar: {scalar_result:.8}, SIMD: {simd_result:.8}");

        // Compare against expected value (allowing for some numerical error)
        let scalar_error = (scalar_result - expected).abs();
        let simd_error = (simd_result - expected).abs();

        println!("  Scalar error: {scalar_error:.2e}, SIMD error: {simd_error:.2e}");

        // Both should be reasonably close to expected value
        assert!(
            scalar_error < 1e-6,
            "Scalar tangent error too large for {input}: {scalar_error:.2e}"
        );
        assert!(
            simd_error < 1e-5,
            "SIMD tangent error too large for {input}: {simd_error:.2e}"
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
fn test_tangent_precision_near_zero() {
    let small_values: Vec<f32> = vec![
        1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, -1e-8, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3,
        -1e-2, -1e-1,
    ];

    let scalar_results: Vec<f32> = small_values.iter().map(|x| x.tan()).collect();
    let simd_results = small_values.tan();

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = small_values[i];
        let absolute_error = (scalar_val - simd_val).abs();

        println!(
            "Small value: tan({input_val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}"
        );

        // For small values, tan(x) ≈ x, so both should be very close to the input
        assert!(
            (scalar_val - input_val).abs() < input_val.abs() * 0.1,
            "Scalar tangent should be close to input for small values"
        );
        assert!(
            (simd_val - input_val).abs() < input_val.abs() * 0.1,
            "SIMD tangent should be close to input for small values"
        );

        // SIMD should match scalar closely
        assert!(
            absolute_error < 1e-6,
            "SIMD precision error too large for small input {input_val}: {absolute_error:.2e}"
        );
    }
}
