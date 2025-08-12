//! Precision comparison tests between scalar and SIMD two-argument arctangent implementations.
//!
//! This test suite validates that the SIMD atan2 implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_6, PI};

#[allow(clippy::excessive_precision)]
const SQRT_3: f32 = 1.7320508075688772935274463415058723669428052538103806280558069794;

use simdly::simd::SimdMath;

/// Test precision of SIMD atan2 against scalar atan2 for various input ranges.
#[test]
fn test_atan2_precision_comparison() {
    // Test cases covering different ranges for atan2 (valid for all real pairs except (0,0))
    let test_cases = vec![
        // Standard quadrant tests
        (vec![1.0f32, 1.0, -1.0, -1.0], vec![1.0f32, -1.0, 1.0, -1.0]),
        // First quadrant (positive x, positive y)
        (vec![1.0f32, 2.0, 3.0, 0.5], vec![1.0f32, 1.0, 1.0, 2.0]),
        // Second quadrant (negative x, positive y)
        (vec![1.0f32, 2.0, 3.0, 0.5], vec![-1.0f32, -1.0, -1.0, -2.0]),
        // Third quadrant (negative x, negative y)
        (
            vec![-1.0f32, -2.0, -3.0, -0.5],
            vec![-1.0f32, -1.0, -1.0, -2.0],
        ),
        // Fourth quadrant (positive x, negative y)
        (vec![-1.0f32, -2.0, -3.0, -0.5], vec![1.0f32, 1.0, 1.0, 2.0]),
        // Special angle cases
        (
            vec![0.0f32, 1.0, SQRT_3, 1.0],
            vec![1.0f32, 0.0, 1.0, SQRT_3],
        ),
        // Large values in different quadrants
        (
            vec![100.0f32, 1000.0, -100.0, -1000.0],
            vec![50.0f32, 500.0, -50.0, -500.0],
        ),
        // Small values near zero
        (
            vec![1e-6f32, 1e-4, -1e-6, -1e-4],
            vec![1e-3f32, 1e-5, 1e-3, 1e-5],
        ),
    ];

    for (i, (y_case, x_case)) in test_cases.iter().enumerate() {
        println!("Testing case {}: y={:?}, x={:?}", i + 1, y_case, x_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = y_case
            .iter()
            .zip(x_case.iter())
            .map(|(y, x)| y.atan2(*x))
            .collect();

        // Compute SIMD results
        let simd_results = y_case.atan2(x_case.clone());

        // Compare results element by element
        assert_eq!(
            scalar_results.len(),
            simd_results.len(),
            "Result vectors have different lengths"
        );

        for (j, (&scalar_val, &simd_val)) in
            scalar_results.iter().zip(simd_results.iter()).enumerate()
        {
            let y_val = y_case[j];
            let x_val = x_case[j];
            let absolute_error = (scalar_val - simd_val).abs();
            let relative_error = if scalar_val != 0.0 {
                absolute_error / scalar_val.abs()
            } else {
                absolute_error
            };

            println!(
                "  Y: {y_val:.6}, X: {x_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs Error: {absolute_error:.2e}, Rel Error: {relative_error:.2e}"
            );

            // Assertion with reasonable tolerance for f32 precision
            assert!(
                absolute_error < 1e-4 || relative_error < 1e-4,
                "Precision error too large for inputs y={y_val}, x={x_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs across wide ranges.
#[test]
fn test_atan2_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in range [-100, 100] (atan2 is valid for all real pairs except (0,0))
    let y_inputs: Vec<f32> = (0..test_size)
        .map(|_| {
            let val: f32 = rng.random_range(-100.0..=100.0);
            // Avoid exact zero to prevent (0,0) case
            if val.abs() < 1e-6 {
                if val >= 0.0 {
                    1e-6
                } else {
                    -1e-6
                }
            } else {
                val
            }
        })
        .collect();

    let x_inputs: Vec<f32> = (0..test_size)
        .map(|_| {
            let val: f32 = rng.random_range(-100.0..=100.0);
            // Avoid exact zero to prevent (0,0) case
            if val.abs() < 1e-6 {
                if val >= 0.0 {
                    1e-6
                } else {
                    -1e-6
                }
            } else {
                val
            }
        })
        .collect();

    let scalar_results: Vec<f32> = y_inputs
        .iter()
        .zip(x_inputs.iter())
        .map(|(y, x)| y.atan2(*x))
        .collect();
    let simd_results = y_inputs.atan2(x_inputs.clone());

    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut error_count = 0;

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let y_val = y_inputs[i];
        let x_val = x_inputs[i];
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
                    "Large error #{error_count}: Y: {y_val:.6}, X: {x_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs: {absolute_error:.2e}, Rel: {relative_error:.2e}"
                );
            }
        }

        // Individual assertion with reasonable tolerance for atan2
        assert!(
            absolute_error < 1e-3 || relative_error < 1e-3,
            "Precision error too large at index {i}: y={y_val}, x={x_val}, scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
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
fn test_atan2_edge_cases() {
    let edge_cases = vec![
        // First quadrant - known angles
        (1.0f32, 1.0f32, FRAC_PI_4), // atan2(1, 1) = π/4
        (SQRT_3, 1.0f32, PI / 3.0),  // atan2(√3, 1) = π/3 (60°)
        (1.0f32, SQRT_3, FRAC_PI_6), // atan2(1, √3) = π/6 (30°)
        // Second quadrant
        (1.0f32, -1.0f32, 3.0 * FRAC_PI_4), // atan2(1, -1) = 3π/4
        (SQRT_3, -1.0f32, 2.0 * PI / 3.0),  // atan2(√3, -1) = 2π/3 (120°)
        // Third quadrant
        (-1.0f32, -1.0f32, -3.0 * FRAC_PI_4), // atan2(-1, -1) = -3π/4
        (-SQRT_3, -1.0f32, -2.0 * PI / 3.0),  // atan2(-√3, -1) = -2π/3
        // Fourth quadrant
        (-1.0f32, 1.0f32, -FRAC_PI_4), // atan2(-1, 1) = -π/4
        (-1.0f32, SQRT_3, -FRAC_PI_6), // atan2(-1, √3) = -π/6
        // Axes cases
        (1.0f32, 0.0f32, FRAC_PI_2), // atan2(1, 0) = π/2 (positive y-axis)
        (-1.0f32, 0.0f32, -FRAC_PI_2), // atan2(-1, 0) = -π/2 (negative y-axis)
        (0.0f32, 1.0f32, 0.0f32),    // atan2(0, 1) = 0 (positive x-axis)
        (0.0f32, -1.0f32, PI),       // atan2(0, -1) = π (negative x-axis)
    ];

    for (y_input, x_input, expected) in edge_cases {
        let input_y = vec![y_input];
        let input_x = vec![x_input];
        let scalar_result = input_y
            .iter()
            .zip(input_x.iter())
            .map(|(y, x)| y.atan2(*x))
            .collect::<Vec<f32>>()[0];
        let simd_result = input_y.atan2(input_x)[0];

        println!("Edge case: atan2({y_input:.6}, {x_input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
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
                "Scalar atan2 relative error too large for ({y_input}, {x_input}): {scalar_rel_error:.2e}"
            );
            assert!(
                simd_rel_error < 1e-4,
                "SIMD atan2 relative error too large for ({y_input}, {x_input}): {simd_rel_error:.2e}"
            );
        } else {
            assert!(
                scalar_error < 1e-6,
                "Scalar atan2 absolute error too large for ({y_input}, {x_input}): {scalar_error:.2e}"
            );
            assert!(
                simd_error < 1e-5,
                "SIMD atan2 absolute error too large for ({y_input}, {x_input}): {simd_error:.2e}"
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
                "SIMD vs scalar relative error too large for ({y_input}, {x_input}): {simd_vs_scalar_rel_error:.2e}"
            );
        } else {
            assert!(
                simd_vs_scalar_error < 1e-5,
                "SIMD vs scalar absolute error too large for ({y_input}, {x_input}): {simd_vs_scalar_error:.2e}"
            );
        }
    }
}

/// Test precision with very small values (avoiding the (0,0) singularity).
#[test]
fn test_atan2_precision_near_zero() {
    let small_y_values: Vec<f32> = vec![
        1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, -1e-8, -1e-7, -1e-6, -1e-5, -1e-4, -1e-3,
        -1e-2, -1e-1,
    ];

    let small_x_values: Vec<f32> = vec![
        1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3,
        1e-3,
    ];

    let scalar_results: Vec<f32> = small_y_values
        .iter()
        .zip(small_x_values.iter())
        .map(|(y, x)| y.atan2(*x))
        .collect();
    let simd_results = small_y_values.atan2(small_x_values.clone());

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let y_val = small_y_values[i];
        let x_val = small_x_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Small values: atan2({y_val:.2e}, {x_val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // For small y with larger x, atan2(y, x) ≈ y/x ≈ atan(y/x)
        // SIMD should match scalar closely
        if scalar_val != 0.0 {
            assert!(
                relative_error < 1e-5,
                "SIMD precision error too large for small inputs y={y_val}, x={x_val}: {relative_error:.2e}"
            );
        } else {
            assert!(
                absolute_error < 1e-6,
                "SIMD precision error too large for small inputs y={y_val}, x={x_val}: {absolute_error:.2e}"
            );
        }
    }
}

/// Test precision with large values in different quadrants.
#[test]
fn test_atan2_precision_large_values() {
    let large_y_values: Vec<f32> = vec![
        100.0, 1000.0, 10000.0, -100.0, -1000.0, -10000.0, 1000.0, -1000.0,
    ];

    let large_x_values: Vec<f32> = vec![
        1000.0, 10000.0, 100000.0, -1000.0, -10000.0, -100000.0, -10000.0, 10000.0,
    ];

    let scalar_results: Vec<f32> = large_y_values
        .iter()
        .zip(large_x_values.iter())
        .map(|(y, x)| y.atan2(*x))
        .collect();
    let simd_results = large_y_values.atan2(large_x_values.clone());

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let y_val = large_y_values[i];
        let x_val = large_x_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Large values: atan2({y_val:.1e}, {x_val:.1e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // For large values, results should be in expected ranges based on quadrant
        let expected_quadrant = if y_val > 0.0 && x_val > 0.0 {
            "First (0 to π/2)"
        } else if y_val > 0.0 && x_val < 0.0 {
            "Second (π/2 to π)"
        } else if y_val < 0.0 && x_val < 0.0 {
            "Third (-π to -π/2)"
        } else {
            "Fourth (-π/2 to 0)"
        };

        println!("  Expected quadrant: {expected_quadrant}");

        // SIMD should match scalar closely
        assert!(
            relative_error < 1e-4,
            "SIMD precision error too large for large inputs y={y_val}, x={x_val}: {relative_error:.2e}"
        );
    }
}
