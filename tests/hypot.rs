//! Precision comparison tests between scalar and SIMD 2D hypotenuse implementations.
//!
//! This test suite validates that the SIMD hypot implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use simdly::simd::SimdMath;

/// Test precision of SIMD hypot against scalar hypot for various input ranges.
#[test]
fn test_hypot_precision_comparison() {
    // Test cases covering different ranges for hypot function (all real pairs)
    let test_cases = vec![
        // Pythagorean triples
        (vec![3.0f32, 5.0, 8.0, 7.0], vec![4.0f32, 12.0, 15.0, 24.0]),
        // Simple cases
        (vec![1.0f32, 2.0, 3.0, 4.0], vec![1.0f32, 2.0, 3.0, 4.0]),
        // Large values
        (
            vec![100.0f32, 1000.0, 300.0, 400.0],
            vec![200.0f32, 500.0, 400.0, 300.0],
        ),
        // Small values
        (
            vec![0.01f32, 0.001, 0.1, 0.05],
            vec![0.02f32, 0.002, 0.2, 0.12],
        ),
        // Mixed positive/negative
        (
            vec![3.0f32, -5.0, 8.0, -7.0],
            vec![-4.0f32, 12.0, -15.0, 24.0],
        ),
        // One coordinate zero
        (vec![0.0f32, 5.0, 0.0, 7.0], vec![4.0f32, 0.0, 15.0, 0.0]),
        // Nearly equal coordinates
        (
            vec![1.0f32, 2.0, 3.0, 4.0],
            vec![1.001f32, 2.001, 2.999, 3.999],
        ),
        // Very different magnitudes
        (
            vec![1.0f32, 1000.0, 0.001, 1.0],
            vec![1000.0f32, 1.0, 1.0, 0.001],
        ),
    ];

    for (i, (x_case, y_case)) in test_cases.iter().enumerate() {
        println!("Testing case {}: x={:?}, y={:?}", i + 1, x_case, y_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = x_case
            .iter()
            .zip(y_case.iter())
            .map(|(x, y)| x.hypot(*y))
            .collect();

        // Compute SIMD results
        let simd_results = x_case.hypot(y_case.clone());

        // Compare results element by element
        assert_eq!(
            scalar_results.len(),
            simd_results.len(),
            "Result vectors have different lengths"
        );

        for (j, (&scalar_val, &simd_val)) in
            scalar_results.iter().zip(simd_results.iter()).enumerate()
        {
            let x_val = x_case[j];
            let y_val = y_case[j];
            let absolute_error = (scalar_val - simd_val).abs();
            let relative_error = if scalar_val != 0.0 {
                absolute_error / scalar_val.abs()
            } else {
                absolute_error
            };

            println!(
                "  X: {x_val:.6}, Y: {y_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs Error: {absolute_error:.2e}, Rel Error: {relative_error:.2e}"
            );

            // Assertion with reasonable tolerance for f32 precision
            assert!(
                absolute_error < 1e-4 || relative_error < 1e-5,
                "Precision error too large for inputs x={x_val}, y={y_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs across wide ranges.
#[test]
fn test_hypot_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in range [-100, 100]
    let x_inputs: Vec<f32> = (0..test_size)
        .map(|_| {
            let val: f32 = rng.random_range(-100.0..=100.0);
            val
        })
        .collect();

    let y_inputs: Vec<f32> = (0..test_size)
        .map(|_| {
            let val: f32 = rng.random_range(-100.0..=100.0);
            val
        })
        .collect();

    let scalar_results: Vec<f32> = x_inputs
        .iter()
        .zip(y_inputs.iter())
        .map(|(x, y)| x.hypot(*y))
        .collect();
    let simd_results = x_inputs.as_slice().hypot(y_inputs.as_slice());

    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut error_count = 0;

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let x_val = x_inputs[i];
        let y_val = y_inputs[i];
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
                    "Large error #{error_count}: X: {x_val:.6}, Y: {y_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs: {absolute_error:.2e}, Rel: {relative_error:.2e}"
                );
            }
        }

        // Individual assertion with reasonable tolerance for hypot
        assert!(
            absolute_error < 1e-3 || relative_error < 1e-4,
            "Precision error too large at index {i}: x={x_val}, y={y_val}, scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
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
fn test_hypot_edge_cases() {
    let edge_cases = vec![
        // Pythagorean triples (exact results)
        (3.0f32, 4.0f32, 5.0f32),   // 3-4-5 triangle
        (5.0f32, 12.0f32, 13.0f32), // 5-12-13 triangle
        (8.0f32, 15.0f32, 17.0f32), // 8-15-17 triangle
        (7.0f32, 24.0f32, 25.0f32), // 7-24-25 triangle
        // Simple cases
        (0.0f32, 0.0f32, 0.0f32),        // hypot(0, 0) = 0
        (1.0f32, 0.0f32, 1.0f32),        // hypot(1, 0) = 1
        (0.0f32, 1.0f32, 1.0f32),        // hypot(0, 1) = 1
        (1.0f32, 1.0f32, 2.0f32.sqrt()), // hypot(1, 1) = √2
        // Negative values (should be same as positive)
        (-3.0f32, -4.0f32, 5.0f32), // hypot(-3, -4) = 5
        (3.0f32, -4.0f32, 5.0f32),  // hypot(3, -4) = 5
        (-3.0f32, 4.0f32, 5.0f32),  // hypot(-3, 4) = 5
        // Very small values
        (1e-10f32, 1e-10f32, (2.0f32).sqrt() * 1e-10),
    ];

    for (x, y, expected) in edge_cases {
        let x_vec = vec![x];
        let y_vec = vec![y];
        let scalar_result = x_vec
            .iter()
            .zip(y_vec.iter())
            .map(|(x, y)| x.hypot(*y))
            .collect::<Vec<f32>>()[0];
        let simd_result = x_vec.hypot(y_vec)[0];

        println!("Edge case: hypot({x:.6}, {y:.6}) = {scalar_result:.8} (expected: {expected:.8})");
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
                scalar_rel_error < 1e-6,
                "Scalar hypot relative error too large for ({x}, {y}): {scalar_rel_error:.2e}"
            );
            assert!(
                simd_rel_error < 1e-5,
                "SIMD hypot relative error too large for ({x}, {y}): {simd_rel_error:.2e}"
            );
        } else {
            assert!(
                scalar_error < 1e-7,
                "Scalar hypot absolute error too large for ({x}, {y}): {scalar_error:.2e}"
            );
            assert!(
                simd_error < 1e-6,
                "SIMD hypot absolute error too large for ({x}, {y}): {simd_error:.2e}"
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
                "SIMD vs scalar relative error too large for ({x}, {y}): {simd_vs_scalar_rel_error:.2e}"
            );
        } else {
            assert!(
                simd_vs_scalar_error < 1e-6,
                "SIMD vs scalar absolute error too large for ({x}, {y}): {simd_vs_scalar_error:.2e}"
            );
        }
    }
}

/// Test precision with very small values near zero.
#[test]
fn test_hypot_precision_near_zero() {
    let small_x_values: Vec<f32> = vec![1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];

    let small_y_values: Vec<f32> = vec![1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];

    let scalar_results: Vec<f32> = small_x_values
        .iter()
        .zip(small_y_values.iter())
        .map(|(x, y)| x.hypot(*y))
        .collect();
    let simd_results = small_x_values.hypot(small_y_values.clone());

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let x_val = small_x_values[i];
        let y_val = small_y_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Small values: hypot({x_val:.2e}, {y_val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // For small values, hypot should be accurate
        // SIMD should match scalar closely
        if scalar_val != 0.0 {
            assert!(
                relative_error < 1e-6,
                "SIMD precision error too large for small inputs x={x_val}, y={y_val}: {relative_error:.2e}"
            );
        } else {
            assert!(
                absolute_error < 1e-7,
                "SIMD precision error too large for small inputs x={x_val}, y={y_val}: {absolute_error:.2e}"
            );
        }
    }
}

/// Test precision with very large values that might cause overflow.
#[test]
fn test_hypot_precision_large_values() {
    let large_x_values: Vec<f32> = vec![1e6, 1e7, 1e8, 1e9, 1e10, 1e20, 1e30, 1e35];

    let large_y_values: Vec<f32> = vec![1e6, 1e7, 1e8, 1e9, 1e10, 1e20, 1e30, 1e35];

    let scalar_results: Vec<f32> = large_x_values
        .iter()
        .zip(large_y_values.iter())
        .map(|(x, y)| x.hypot(*y))
        .collect();
    let simd_results = large_x_values.hypot(large_y_values.clone());

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let x_val = large_x_values[i];
        let y_val = large_y_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Large values: hypot({x_val:.1e}, {y_val:.1e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // For large values, both results should be finite (no overflow)
        assert!(scalar_val.is_finite(), "Scalar result should be finite");
        assert!(simd_val.is_finite(), "SIMD result should be finite");

        // SIMD should match scalar reasonably well
        if scalar_val.is_finite() && simd_val.is_finite() {
            assert!(
                relative_error < 1e-4,
                "SIMD precision error too large for large inputs x={x_val}, y={y_val}: {relative_error:.2e}"
            );
        }
    }
}

/// Test commutativity: hypot(x, y) should equal hypot(y, x).
#[test]
fn test_hypot_commutativity() {
    let x_values: Vec<f32> = vec![3.0, 5.0, 8.0, 1.0, -2.0, 0.1, 100.0, 0.001];

    let y_values: Vec<f32> = vec![4.0, 12.0, 15.0, 1.0, 3.0, 0.2, 200.0, 0.002];

    let result1 = x_values.hypot(y_values.clone());
    let result2 = y_values.hypot(x_values.clone());

    for (i, (&val1, &val2)) in result1.iter().zip(result2.iter()).enumerate() {
        let x_val = x_values[i];
        let y_val = y_values[i];

        println!(
            "Commutativity test: hypot({x_val:.6}, {y_val:.6}) = {val1:.8}, hypot({y_val:.6}, {x_val:.6}) = {val2:.8}"
        );

        let error = (val1 - val2).abs();
        let rel_error = if val1 != 0.0 {
            error / val1.abs()
        } else {
            error
        };

        assert!(
            error < 1e-7 || rel_error < 1e-7,
            "Hypot should be commutative for ({x_val}, {y_val}): error={error:.2e}, rel_error={rel_error:.2e}"
        );
    }
}
