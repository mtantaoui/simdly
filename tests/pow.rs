//! Precision comparison tests between scalar and SIMD power function implementations.
//!
//! This test suite validates that the SIMD pow implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use simdly::simd::SimdMath;

/// Test precision of SIMD pow against scalar pow for various input ranges.
#[test]
fn test_pow_precision_comparison() {
    // Test cases covering different ranges for pow function
    let test_cases = vec![
        // Basic integer powers
        (vec![2.0f32, 3.0, 4.0, 5.0], vec![2.0f32, 2.0, 2.0, 2.0]),
        // Unit base with various powers
        (vec![1.0f32, 1.0, 1.0, 1.0], vec![0.0f32, 1.0, 2.0, 100.0]),
        // Fractional powers
        (vec![4.0f32, 9.0, 16.0, 25.0], vec![0.5f32, 0.5, 0.5, 0.5]),
        // Negative powers
        (vec![2.0f32, 3.0, 4.0, 5.0], vec![-1.0f32, -2.0, -1.0, -2.0]),
        // Mixed cases
        (vec![2.0f32, 0.5, 10.0, 0.1], vec![3.0f32, 2.0, 0.5, -1.0]),
        // Small bases
        (vec![0.1f32, 0.2, 0.5, 0.9], vec![2.0f32, 3.0, 2.0, 2.0]),
        // Powers near zero
        (
            vec![2.0f32, 3.0, 4.0, 5.0],
            vec![0.1f32, 0.01, 0.001, 0.0001],
        ),
        // Powers near one
        (vec![2.0f32, 3.0, 4.0, 5.0], vec![1.1f32, 0.9, 1.01, 0.99]),
    ];

    for (i, (base_case, exp_case)) in test_cases.iter().enumerate() {
        println!(
            "Testing case {}: base={:?}, exp={:?}",
            i + 1,
            base_case,
            exp_case
        );

        // Compute scalar results
        let scalar_results: Vec<f32> = base_case
            .iter()
            .zip(exp_case.iter())
            .map(|(b, e)| b.powf(*e))
            .collect();

        // Compute SIMD results
        let simd_results = base_case.pow(exp_case.clone());

        // Compare results element by element
        assert_eq!(
            scalar_results.len(),
            simd_results.len(),
            "Result vectors have different lengths"
        );

        for (j, (&scalar_val, &simd_val)) in
            scalar_results.iter().zip(simd_results.iter()).enumerate()
        {
            let base_val = base_case[j];
            let exp_val = exp_case[j];
            let absolute_error = (scalar_val - simd_val).abs();
            let relative_error = if scalar_val != 0.0 {
                absolute_error / scalar_val.abs()
            } else {
                absolute_error
            };

            println!(
                "  Base: {base_val:.6}, Exp: {exp_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs Error: {absolute_error:.2e}, Rel Error: {relative_error:.2e}"
            );

            // Assertion with reasonable tolerance for f32 precision
            // Power function can have significant errors for certain combinations
            assert!(
                absolute_error < 1e-3 || relative_error < 1e-3,
                "Precision error too large for base={base_val}, exp={exp_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs across reasonable ranges.
#[test]
fn test_pow_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in reasonable ranges to avoid overflow/underflow
    let base_inputs: Vec<f32> = (0..test_size)
        .map(|_| {
            let val: f32 = rng.random_range(0.1..=10.0); // Positive bases to avoid complex results
            val
        })
        .collect();

    let exp_inputs: Vec<f32> = (0..test_size)
        .map(|_| {
            let val: f32 = rng.random_range(-3.0..=3.0); // Limited exponent range
            val
        })
        .collect();

    let scalar_results: Vec<f32> = base_inputs
        .iter()
        .zip(exp_inputs.iter())
        .map(|(b, e)| b.powf(*e))
        .collect();
    let simd_results = base_inputs.as_slice().pow(exp_inputs.as_slice());

    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut error_count = 0;

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let base_val = base_inputs[i];
        let exp_val = exp_inputs[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        max_abs_error = max_abs_error.max(absolute_error);
        max_rel_error = max_rel_error.max(relative_error);

        // Count significant errors
        if absolute_error > 1e-4 && relative_error > 1e-4 {
            error_count += 1;
            if error_count <= 10 {
                // Print first 10 significant errors
                println!(
                    "Large error #{error_count}: Base: {base_val:.6}, Exp: {exp_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs: {absolute_error:.2e}, Rel: {relative_error:.2e}"
                );
            }
        }

        // Individual assertion with reasonable tolerance for pow
        assert!(
            absolute_error < 1e-2 || relative_error < 1e-2,
            "Precision error too large at index {i}: base={base_val}, exp={exp_val}, scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
        );
    }

    println!("Random precision test summary:");
    println!("  Test size: {test_size}");
    println!("  Max absolute error: {max_abs_error:.2e}");
    println!("  Max relative error: {max_rel_error:.2e}");
    println!("  Significant errors (>1e-4): {error_count}");

    // Overall precision requirements - relaxed for power function due to complexity
    assert!(
        max_abs_error < 1e-1,
        "Maximum absolute error too large: {max_abs_error:.2e}"
    );
    assert!(
        max_rel_error < 1e-1,
        "Maximum relative error too large: {max_rel_error:.2e}"
    );
    assert!(
        error_count < test_size / 5,
        "Too many significant errors: {error_count}/{test_size}"
    );
}

/// Test edge cases and special values.
#[test]
fn test_pow_edge_cases() {
    let edge_cases = vec![
        // Known power values
        (2.0f32, 0.0f32, 1.0f32),    // 2^0 = 1
        (2.0f32, 1.0f32, 2.0f32),    // 2^1 = 2
        (2.0f32, 2.0f32, 4.0f32),    // 2^2 = 4
        (2.0f32, 3.0f32, 8.0f32),    // 2^3 = 8
        (3.0f32, 2.0f32, 9.0f32),    // 3^2 = 9
        (4.0f32, 0.5f32, 2.0f32),    // 4^0.5 = 2
        (9.0f32, 0.5f32, 3.0f32),    // 9^0.5 = 3
        (8.0f32, 1.0 / 3.0, 2.0f32), // 8^(1/3) = 2
        (1.0f32, 100.0f32, 1.0f32),  // 1^100 = 1
        (2.0f32, -1.0f32, 0.5f32),   // 2^(-1) = 0.5
        (2.0f32, -2.0f32, 0.25f32),  // 2^(-2) = 0.25
        (0.5f32, 2.0f32, 0.25f32),   // 0.5^2 = 0.25
    ];

    for (base, exp, expected) in edge_cases {
        let base_vec = vec![base];
        let exp_vec = vec![exp];
        let scalar_result = base_vec
            .iter()
            .zip(exp_vec.iter())
            .map(|(b, e)| b.powf(*e))
            .collect::<Vec<f32>>()[0];
        let simd_result = base_vec.pow(exp_vec)[0];

        println!(
            "Edge case: pow({base:.6}, {exp:.6}) = {scalar_result:.8} (expected: {expected:.8})"
        );
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
                "Scalar pow relative error too large for ({base}, {exp}): {scalar_rel_error:.2e}"
            );
            assert!(
                simd_rel_error < 1e-4,
                "SIMD pow relative error too large for ({base}, {exp}): {simd_rel_error:.2e}"
            );
        } else {
            assert!(
                scalar_error < 1e-6,
                "Scalar pow absolute error too large for ({base}, {exp}): {scalar_error:.2e}"
            );
            assert!(
                simd_error < 1e-5,
                "SIMD pow absolute error too large for ({base}, {exp}): {simd_error:.2e}"
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
                "SIMD vs scalar relative error too large for ({base}, {exp}): {simd_vs_scalar_rel_error:.2e}"
            );
        } else {
            assert!(
                simd_vs_scalar_error < 1e-5,
                "SIMD vs scalar absolute error too large for ({base}, {exp}): {simd_vs_scalar_error:.2e}"
            );
        }
    }
}

/// Test precision with small bases near unity.
#[test]
fn test_pow_precision_near_unity() {
    let base_values: Vec<f32> = vec![0.9, 0.95, 0.99, 0.999, 1.001, 1.01, 1.05, 1.1];

    let exp_values: Vec<f32> = vec![2.0, 3.0, 0.5, -1.0, -2.0, 10.0, 0.1, 100.0];

    let scalar_results: Vec<f32> = base_values
        .iter()
        .zip(exp_values.iter())
        .map(|(b, e)| b.powf(*e))
        .collect();
    let simd_results = base_values.pow(exp_values.clone());

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let base_val = base_values[i];
        let exp_val = exp_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Near unity: pow({base_val:.6}, {exp_val:.6}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // SIMD should match scalar closely for values near unity
        if scalar_val != 0.0 {
            assert!(
                relative_error < 1e-5,
                "SIMD precision error too large for near-unity inputs base={base_val}, exp={exp_val}: {relative_error:.2e}"
            );
        } else {
            assert!(
                absolute_error < 1e-6,
                "SIMD precision error too large for near-unity inputs base={base_val}, exp={exp_val}: {absolute_error:.2e}"
            );
        }
    }
}

/// Test precision with integer exponents.
#[test]
fn test_pow_precision_integer_exponents() {
    let base_values: Vec<f32> = vec![2.0, 3.0, 1.5, 0.5, 10.0, 0.1, 1.25, 0.8];

    let exp_values: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0];

    let scalar_results: Vec<f32> = base_values
        .iter()
        .zip(exp_values.iter())
        .map(|(b, e)| b.powf(*e))
        .collect();
    let simd_results = base_values.pow(exp_values.clone());

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let base_val = base_values[i];
        let exp_val = exp_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Integer exp: pow({base_val:.6}, {exp_val:.1}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // Integer exponents should be very accurate
        if scalar_val != 0.0 {
            assert!(
                relative_error < 1e-6,
                "SIMD precision error too large for integer exponent base={base_val}, exp={exp_val}: {relative_error:.2e}"
            );
        } else {
            assert!(
                absolute_error < 1e-7,
                "SIMD precision error too large for integer exponent base={base_val}, exp={exp_val}: {absolute_error:.2e}"
            );
        }
    }
}
