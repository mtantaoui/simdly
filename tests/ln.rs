//! Precision comparison tests between scalar and SIMD natural logarithm implementations.
//!
//! This test suite validates that the SIMD natural logarithm implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use simdly::simd::SimdMath;

/// Test precision of SIMD ln against scalar ln for various input ranges.
#[test]
fn test_ln_precision_comparison() {
    // Test cases covering different ranges and edge cases
    // Only positive values as ln is undefined for negative inputs and zero
    let test_cases = [
        // Small positive values near 1
        vec![0.5f32, 0.8, 1.0, 1.2, 1.5],
        // Unit values
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0],
        // Powers of e
        vec![
            std::f32::consts::E,
            std::f32::consts::E * std::f32::consts::E,
        ],
        // Powers of 2
        vec![2.0f32, 4.0, 8.0, 16.0, 32.0],
        // Powers of 10
        vec![10.0f32, 100.0, 1000.0],
        // Small positive values
        vec![0.1f32, 0.01, 0.001, 0.0001],
        // Fractional values
        vec![0.25f32, 0.333, 0.666, 0.75],
        // Larger values
        vec![50.0f32, 100.0, 500.0, 1000.0],
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
        println!("Testing case {}: {:?}", i + 1, test_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = test_case.iter().map(|x| x.ln()).collect();

        // Compute SIMD results
        let simd_results = test_case.ln();

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
                absolute_error < 1e-4 || relative_error < 1e-4,
                "Precision error too large for input {input_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated positive inputs.
#[test]
fn test_ln_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random positive inputs in range [0.001, 1000] to avoid extremes
    let inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(0.001..=1000.0))
        .collect();

    let scalar_results: Vec<f32> = inputs.iter().map(|x| x.ln()).collect();
    let simd_results = inputs.ln();

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

        // Individual assertion with reasonable tolerance for ln
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
        max_rel_error < 1e-3,
        "Maximum relative error too large: {max_rel_error:.2e}"
    );
    assert!(
        error_count < test_size / 10,
        "Too many significant errors: {error_count}/{test_size}"
    );
}

/// Test edge cases and special values.
#[test]
fn test_ln_edge_cases() {
    let edge_cases = vec![
        // Exact values where ln should be known
        (1.0f32, 0.0f32),                      // ln(1) = 0
        (std::f32::consts::E, 1.0f32),         // ln(e) = 1
        (std::f32::consts::E.powi(2), 2.0f32), // ln(e²) = 2
        (std::f32::consts::E.powi(3), 3.0f32), // ln(e³) = 3
        (10.0f32, std::f32::consts::LN_10),    // ln(10) = ln(10)
        (2.0f32, std::f32::consts::LN_2),      // ln(2) = ln(2)
    ];

    for (input, expected) in edge_cases {
        let input_vec = vec![input];
        let scalar_result = input_vec.iter().map(|x| x.ln()).collect::<Vec<f32>>()[0];
        let simd_result = input_vec.ln()[0];

        println!("Edge case: ln({input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
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
        if expected != 0.0 {
            assert!(
                scalar_rel_error < 1e-5,
                "Scalar ln relative error too large for {input}: {scalar_rel_error:.2e}"
            );
            assert!(
                simd_rel_error < 1e-4,
                "SIMD ln relative error too large for {input}: {simd_rel_error:.2e}"
            );
        } else {
            assert!(
                scalar_error < 1e-6,
                "Scalar ln absolute error too large for {input}: {scalar_error:.2e}"
            );
            assert!(
                simd_error < 1e-5,
                "SIMD ln absolute error too large for {input}: {simd_error:.2e}"
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

/// Test precision with values near 1 (where ln is most sensitive).
#[test]
fn test_ln_precision_near_one() {
    let near_one_values: Vec<f32> = vec![
        0.9, 0.95, 0.99, 0.999, 1.001, 1.01, 1.05, 1.1, 1.5, 2.0, 0.5, 0.1, 0.01, 10.0, 100.0,
    ];

    let scalar_results: Vec<f32> = near_one_values.iter().map(|x| x.ln()).collect();
    let simd_results = near_one_values.ln();

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = near_one_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Near-one value: ln({input_val:.6}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // SIMD should match scalar closely
        if scalar_val != 0.0 {
            assert!(
                relative_error < 1e-4,
                "SIMD precision error too large for input {input_val}: {relative_error:.2e}"
            );
        } else {
            assert!(
                absolute_error < 1e-5,
                "SIMD precision error too large for input {input_val}: {absolute_error:.2e}"
            );
        }
    }
}
