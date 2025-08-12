//! Precision comparison tests between scalar and SIMD exponential implementations.
//!
//! This test suite validates that the SIMD exponential implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use simdly::simd::SimdMath;

/// Test precision of SIMD exp against scalar exp for various input ranges.
#[test]
fn test_exp_precision_comparison() {
    // Test cases covering different ranges and edge cases
    let test_cases = [
        // Small values near zero
        vec![0.0f32, 0.1, 0.2, 0.3],
        // Unit values
        vec![1.0f32, 2.0, 3.0, 4.0],
        // Negative values
        vec![-0.5f32, -1.0, -2.0, -3.0],
        // Mixed range
        vec![-2.0f32, -1.0, 0.0, 1.0, 2.0],
        // Moderate values (avoiding overflow)
        vec![5.0f32, 6.0, 7.0, 8.0],
        // Larger negative values
        vec![-10.0f32, -5.0, -8.0, -15.0],
        // Very small values
        vec![1e-6f32, 1e-4, 1e-2],
        // Values that should give known results
        vec![std::f32::consts::LN_2, std::f32::consts::LN_10],
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
        println!("Testing case {}: {:?}", i + 1, test_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = test_case.iter().map(|x| x.exp()).collect();

        // Compute SIMD results
        let simd_results = test_case.exp();

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
            // exp can grow rapidly, so we use relative error primarily
            assert!(
                absolute_error < 1e-3 || relative_error < 1e-4,
                "Precision error too large for input {input_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs in safe ranges.
#[test]
fn test_exp_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in range [-10, 10] to avoid overflow/underflow
    let inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-10.0..=10.0))
        .collect();

    let scalar_results: Vec<f32> = inputs.iter().map(|x| x.exp()).collect();
    let simd_results = inputs.exp();

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
        if relative_error > 1e-4 && absolute_error > 1e-3 {
            error_count += 1;
            if error_count <= 10 {
                // Print first 10 significant errors
                println!(
                    "Large error #{error_count}: Input: {input_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs: {absolute_error:.2e}, Rel: {relative_error:.2e}"
                );
            }
        }

        // Individual assertion with reasonable tolerance for exp
        assert!(
            absolute_error < 1e-1 || relative_error < 1e-3,
            "Precision error too large at index {i}: input={input_val}, scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
        );
    }

    println!("Random precision test summary:");
    println!("  Test size: {test_size}");
    println!("  Max absolute error: {max_abs_error:.2e}");
    println!("  Max relative error: {max_rel_error:.2e}");
    println!("  Significant errors (rel>1e-4 && abs>1e-3): {error_count}");

    // Overall precision requirements - relaxed for exp due to rapid growth
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
fn test_exp_edge_cases() {
    let edge_cases = vec![
        // Exact values where exp should be known
        (0.0f32, 1.0f32),                        // exp(0) = 1
        (1.0f32, std::f32::consts::E),           // exp(1) = e
        (std::f32::consts::LN_2, 2.0f32),        // exp(ln(2)) = 2
        (std::f32::consts::LN_10, 10.0f32),      // exp(ln(10)) = 10
        (-1.0f32, 1.0f32 / std::f32::consts::E), // exp(-1) = 1/e
    ];

    for (input, expected) in edge_cases {
        let input_vec = vec![input];
        let scalar_result = input_vec.iter().map(|x| x.exp()).collect::<Vec<f32>>()[0];
        let simd_result = input_vec.exp()[0];

        println!("Edge case: exp({input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
        println!("  Scalar: {scalar_result:.8}, SIMD: {simd_result:.8}");

        // Compare against expected value (allowing for some numerical error)
        let scalar_error = (scalar_result - expected).abs();
        let simd_error = (simd_result - expected).abs();
        let scalar_rel_error = if expected != 0.0 {
            scalar_error / expected.abs()
        } else {
            scalar_error
        };
        let simd_rel_error = if expected != 0.0 {
            simd_error / expected.abs()
        } else {
            simd_error
        };

        println!("  Scalar error: {scalar_error:.2e} (rel: {scalar_rel_error:.2e}), SIMD error: {simd_error:.2e} (rel: {simd_rel_error:.2e})");

        // Both should be reasonably close to expected value
        assert!(
            scalar_rel_error < 1e-5,
            "Scalar exp relative error too large for {input}: {scalar_rel_error:.2e}"
        );
        assert!(
            simd_rel_error < 1e-4,
            "SIMD exp relative error too large for {input}: {simd_rel_error:.2e}"
        );

        // SIMD should be close to scalar
        let simd_vs_scalar_error = (simd_result - scalar_result).abs();
        let simd_vs_scalar_rel_error = if scalar_result != 0.0 {
            simd_vs_scalar_error / scalar_result.abs()
        } else {
            simd_vs_scalar_error
        };
        assert!(
            simd_vs_scalar_rel_error < 1e-4,
            "SIMD vs scalar relative error too large for {input}: {simd_vs_scalar_rel_error:.2e}"
        );
    }
}

/// Test precision with very small values near zero.
#[test]
fn test_exp_precision_near_zero() {
    let small_values: Vec<f32> = vec![
        1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, -1e-8, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3,
        -1e-2, -1e-1,
    ];

    let scalar_results: Vec<f32> = small_values.iter().map(|x| x.exp()).collect();
    let simd_results = small_values.exp();

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
            "Small value: exp({input_val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // For small values, exp(x) ≈ 1 + x
        // Check that results are in reasonable bounds for small inputs
        assert!(
            scalar_val > 0.8 && scalar_val < 1.2,
            "Scalar exp should be in reasonable range for small input"
        );
        assert!(
            simd_val > 0.8 && simd_val < 1.2,
            "SIMD exp should be in reasonable range for small input"
        );

        // For very small inputs, verify mathematical relationship
        if input_val.abs() > 1e-6 {
            // For larger small values, verify directional correctness
            if input_val > 0.0 {
                assert!(
                    scalar_val >= 1.0,
                    "Scalar exp should be >= 1 for positive input"
                );
                assert!(
                    simd_val >= 1.0,
                    "SIMD exp should be >= 1 for positive input"
                );
            } else {
                assert!(
                    scalar_val <= 1.0,
                    "Scalar exp should be <= 1 for negative input"
                );
                assert!(
                    simd_val <= 1.0,
                    "SIMD exp should be <= 1 for negative input"
                );
            }
        }

        // SIMD should match scalar closely
        assert!(
            relative_error < 1e-5,
            "SIMD precision error too large for small input {input_val}: {relative_error:.2e}"
        );
    }
}
