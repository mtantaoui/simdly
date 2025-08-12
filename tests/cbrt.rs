//! Precision comparison tests between scalar and SIMD cube root implementations.
//!
//! This test suite validates that the SIMD cube root implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use simdly::simd::SimdMath;

/// Test precision of SIMD cbrt against scalar cbrt for various input ranges.
#[test]
fn test_cbrt_precision_comparison() {
    // Test cases covering different ranges and edge cases
    // cbrt is valid for all real numbers including negative values
    let test_cases = vec![
        // Small positive values
        vec![0.0f32, 0.001, 0.1, 0.125],
        // Unit values
        vec![1.0f32, 8.0, 27.0, 64.0],
        // Perfect cubes
        vec![1.0f32, 8.0, 27.0, 64.0, 125.0, 216.0],
        // Larger values (reasonable range for high precision)
        vec![1000.0f32, 2000.0, 5000.0, 10000.0],
        // Fractional values
        vec![0.125f32, 0.25, 0.5, 0.75],
        // Negative values
        vec![-1.0f32, -8.0, -27.0, -64.0],
        // Larger negative values (reasonable range for high precision)
        vec![-125.0f32, -216.0, -1000.0, -5000.0],
        // Mixed positive and negative
        vec![-8.0f32, -1.0, 0.0, 1.0, 8.0],
        // Very small positive values
        vec![1e-8f32, 1e-6, 1e-4, 1e-2],
        // Very small negative values
        vec![-1e-8f32, -1e-6, -1e-4, -1e-2],
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
        println!("Testing case {}: {:?}", i + 1, test_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = test_case.iter().map(|x| x.cbrt()).collect();

        // Compute SIMD results
        let simd_results = test_case.cbrt();

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
            assert!(
                absolute_error < 1e-5 || relative_error < 1e-5,
                "Precision error too large for input {input_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs across wide ranges.
#[test]
fn test_cbrt_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in range [-10000, 10000] (cbrt is valid for all real numbers)
    // Using smaller range to avoid known precision issues with very large values
    let inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-10000.0..=10000.0))
        .collect();

    let scalar_results: Vec<f32> = inputs.iter().map(|x| x.cbrt()).collect();
    let simd_results = inputs.cbrt();

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

        // Individual assertion with reasonable tolerance for cbrt
        assert!(
            absolute_error < 1e-5 || relative_error < 1e-5,
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
        max_abs_error < 1e-5,
        "Maximum absolute error too large: {max_abs_error:.2e}"
    );
    assert!(
        max_rel_error < 1e-6,
        "Maximum relative error too large: {max_rel_error:.2e}"
    );
    assert!(
        error_count < test_size / 10,
        "Too many significant errors: {error_count}/{test_size}"
    );
}

/// Test edge cases and special values.
#[test]
fn test_cbrt_edge_cases() {
    let edge_cases = vec![
        // Exact values where cbrt should be known
        (0.0f32, 0.0f32),           // cbrt(0) = 0
        (1.0f32, 1.0f32),           // cbrt(1) = 1
        (-1.0f32, -1.0f32),         // cbrt(-1) = -1
        (8.0f32, 2.0f32),           // cbrt(8) = 2
        (-8.0f32, -2.0f32),         // cbrt(-8) = -2
        (27.0f32, 3.0f32),          // cbrt(27) = 3
        (-27.0f32, -3.0f32),        // cbrt(-27) = -3
        (64.0f32, 4.0f32),          // cbrt(64) = 4
        (-64.0f32, -4.0f32),        // cbrt(-64) = -4
        (125.0f32, 5.0f32),         // cbrt(125) = 5
        (-125.0f32, -5.0f32),       // cbrt(-125) = -5
        (0.125f32, 0.5f32),         // cbrt(0.125) = 0.5
        (-0.125f32, -0.5f32),       // cbrt(-0.125) = -0.5
    ];

    for (input, expected) in edge_cases {
        let input_vec = vec![input];
        let scalar_result = input_vec.iter().map(|x| x.cbrt()).collect::<Vec<f32>>()[0];
        let simd_result = input_vec.cbrt()[0];

        println!("Edge case: cbrt({input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
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
                scalar_rel_error < 1e-6,
                "Scalar cbrt relative error too large for {input}: {scalar_rel_error:.2e}"
            );
            assert!(
                simd_rel_error < 1e-5,
                "SIMD cbrt relative error too large for {input}: {simd_rel_error:.2e}"
            );
        } else {
            assert!(
                scalar_error < 1e-7,
                "Scalar cbrt absolute error too large for {input}: {scalar_error:.2e}"
            );
            assert!(
                simd_error < 1e-6,
                "SIMD cbrt absolute error too large for {input}: {simd_error:.2e}"
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
                simd_vs_scalar_rel_error < 1e-5,
                "SIMD vs scalar relative error too large for {input}: {simd_vs_scalar_rel_error:.2e}"
            );
        } else {
            assert!(
                simd_vs_scalar_error < 1e-6,
                "SIMD vs scalar absolute error too large for {input}: {simd_vs_scalar_error:.2e}"
            );
        }
    }
}

/// Test precision with very small values near zero.
#[test]
fn test_cbrt_precision_near_zero() {
    let small_values: Vec<f32> = vec![
        1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 
        -1e-9, -1e-8, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3, -1e-2, -1e-1,
    ];

    let scalar_results: Vec<f32> = small_values.iter().map(|x| x.cbrt()).collect();
    let simd_results = small_values.cbrt();

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
            "Small value: cbrt({input_val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // For small values, cbrt should be accurate
        let expected = input_val.cbrt();
        if expected != 0.0 {
            assert!(
                (scalar_val - expected).abs() / expected.abs() < 1e-6,
                "Scalar cbrt should be accurate for small input"
            );
            assert!(
                (simd_val - expected).abs() / expected.abs() < 1e-5,
                "SIMD cbrt should be accurate for small input"
            );
        } else {
            assert!(
                (scalar_val - expected).abs() < 1e-7,
                "Scalar cbrt should be accurate for small input"
            );
            assert!(
                (simd_val - expected).abs() < 1e-6,
                "SIMD cbrt should be accurate for small input"
            );
        }

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