//! Precision comparison tests between scalar and SIMD 4D hypotenuse implementations.
//!
//! This test suite validates that the SIMD hypot4 implementation maintains
//! acceptable precision compared to scalar 4D distance calculation.

use simdly::simd::SimdMath;

/// Test precision of SIMD hypot4 against scalar sqrt(x²+y²+z²+w²) for various input ranges.
#[test]
fn test_hypot4_precision_comparison() {
    // Test cases covering different ranges for 4D hypot function
    let test_cases = vec![
        // Simple unit vectors and known cases
        (
            vec![1.0f32, 0.0, 2.0, 3.0],
            vec![0.0f32, 1.0, 3.0, 4.0],
            vec![0.0f32, 0.0, 6.0, 5.0],
            vec![0.0f32, 0.0, 0.0, 12.0],
        ),
        // 4D "Pythagorean-like" cases
        (
            vec![1.0f32, 2.0, 1.0, 2.0],
            vec![2.0f32, 3.0, 2.0, 4.0],
            vec![2.0f32, 6.0, 3.0, 4.0],
            vec![4.0f32, 2.0, 4.0, 12.0],
        ),
        // Large values
        (
            vec![100.0f32, 200.0, 300.0, 150.0],
            vec![200.0f32, 300.0, 400.0, 200.0],
            vec![300.0f32, 100.0, 500.0, 250.0],
            vec![400.0f32, 500.0, 600.0, 300.0],
        ),
        // Small values
        (
            vec![0.01f32, 0.02, 0.001, 0.1],
            vec![0.02f32, 0.03, 0.002, 0.2],
            vec![0.03f32, 0.01, 0.003, 0.15],
            vec![0.04f32, 0.05, 0.004, 0.25],
        ),
        // Mixed positive/negative
        (
            vec![3.0f32, -4.0, 5.0, -6.0],
            vec![-4.0f32, 3.0, -12.0, 8.0],
            vec![5.0f32, -5.0, 13.0, -10.0],
            vec![-12.0f32, 13.0, -5.0, 9.0],
        ),
        // One coordinate zero
        (
            vec![0.0f32, 3.0, 0.0, 4.0],
            vec![4.0f32, 0.0, 5.0, 0.0],
            vec![5.0f32, 4.0, 0.0, 3.0],
            vec![12.0f32, 5.0, 13.0, 0.0],
        ),
        // Two coordinates zero
        (
            vec![0.0f32, 0.0, 3.0, 0.0],
            vec![0.0f32, 4.0, 0.0, 0.0],
            vec![5.0f32, 0.0, 4.0, 5.0],
            vec![0.0f32, 0.0, 0.0, 12.0],
        ),
        // All coordinates equal
        (
            vec![1.0f32, 2.0, 3.0, 0.5],
            vec![1.0f32, 2.0, 3.0, 0.5],
            vec![1.0f32, 2.0, 3.0, 0.5],
            vec![1.0f32, 2.0, 3.0, 0.5],
        ),
    ];

    for (i, (x_case, y_case, z_case, w_case)) in test_cases.iter().enumerate() {
        println!(
            "Testing case {}: x={:?}, y={:?}, z={:?}, w={:?}",
            i + 1,
            x_case,
            y_case,
            z_case,
            w_case
        );

        // Compute scalar results using sqrt(x²+y²+z²+w²)
        let scalar_results: Vec<f32> = x_case
            .iter()
            .zip(y_case.iter())
            .zip(z_case.iter())
            .zip(w_case.iter())
            .map(|(((x, y), z), w)| (x * x + y * y + z * z + w * w).sqrt())
            .collect();

        // Compute SIMD results
        let simd_results = x_case.hypot4(y_case.clone(), z_case.clone(), w_case.clone());

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
            let z_val = z_case[j];
            let w_val = w_case[j];
            let absolute_error = (scalar_val - simd_val).abs();
            let relative_error = if scalar_val != 0.0 {
                absolute_error / scalar_val.abs()
            } else {
                absolute_error
            };

            println!(
                "  X: {x_val:.6}, Y: {y_val:.6}, Z: {z_val:.6}, W: {w_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs Error: {absolute_error:.2e}, Rel Error: {relative_error:.2e}"
            );

            // Assertion with reasonable tolerance for f32 precision
            assert!(
                absolute_error < 1e-4 || relative_error < 1e-5,
                "Precision error too large for inputs x={x_val}, y={y_val}, z={z_val}, w={w_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs across wide ranges.
#[test]
fn test_hypot4_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in range [-100, 100]
    let x_inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-100.0..=100.0))
        .collect();
    let y_inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-100.0..=100.0))
        .collect();
    let z_inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-100.0..=100.0))
        .collect();
    let w_inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-100.0..=100.0))
        .collect();

    let scalar_results: Vec<f32> = x_inputs
        .iter()
        .zip(y_inputs.iter())
        .zip(z_inputs.iter())
        .zip(w_inputs.iter())
        .map(|(((x, y), z), w)| (x * x + y * y + z * z + w * w).sqrt())
        .collect();
    let simd_results = x_inputs.as_slice().hypot4(
        y_inputs.as_slice(),
        z_inputs.as_slice(),
        w_inputs.as_slice(),
    );

    let mut max_abs_error = 0.0f32;
    let mut max_rel_error = 0.0f32;
    let mut error_count = 0;

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let x_val = x_inputs[i];
        let y_val = y_inputs[i];
        let z_val = z_inputs[i];
        let w_val = w_inputs[i];
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
                    "Large error #{error_count}: X: {x_val:.6}, Y: {y_val:.6}, Z: {z_val:.6}, W: {w_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs: {absolute_error:.2e}, Rel: {relative_error:.2e}"
                );
            }
        }

        // Individual assertion with reasonable tolerance for hypot4
        assert!(
            absolute_error < 1e-3 || relative_error < 1e-4,
            "Precision error too large at index {i}: x={x_val}, y={y_val}, z={z_val}, w={w_val}, scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}, rel_error={relative_error:.2e}"
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
fn test_hypot4_edge_cases() {
    let edge_cases = vec![
        // Simple cases
        (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32), // hypot4(0, 0, 0, 0) = 0
        (1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32), // hypot4(1, 0, 0, 0) = 1
        (0.0f32, 1.0f32, 0.0f32, 0.0f32, 1.0f32), // hypot4(0, 1, 0, 0) = 1
        (0.0f32, 0.0f32, 1.0f32, 0.0f32, 1.0f32), // hypot4(0, 0, 1, 0) = 1
        (0.0f32, 0.0f32, 0.0f32, 1.0f32, 1.0f32), // hypot4(0, 0, 0, 1) = 1
        (1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32), // hypot4(1, 1, 1, 1) = 2
        // Known 4D distances
        (1.0f32, 2.0f32, 2.0f32, 4.0f32, 5.0f32), // hypot4(1, 2, 2, 4) = 5
        (2.0f32, 3.0f32, 6.0f32, 2.0f32, 7.280109f32), // hypot4(2, 3, 6, 2) = 7
        (3.0f32, 4.0f32, 0.0f32, 0.0f32, 5.0f32), // hypot4(3, 4, 0, 0) = 5
        // Negative values (should be same as positive)
        (-1.0f32, -1.0f32, -1.0f32, -1.0f32, 2.0f32), // hypot4(-1, -1, -1, -1) = 2
        (3.0f32, -4.0f32, 0.0f32, 0.0f32, 5.0f32),    // hypot4(3, -4, 0, 0) = 5
        // Large equal values
        (50.0f32, 50.0f32, 50.0f32, 50.0f32, 100.0f32), // hypot4(50, 50, 50, 50) = 100
        // Very small values
        (1e-10f32, 1e-10f32, 1e-10f32, 1e-10f32, 2e-10f32),
    ];

    for (x, y, z, w, expected) in edge_cases {
        let x_vec = vec![x];
        let y_vec = vec![y];
        let z_vec = vec![z];
        let w_vec = vec![w];
        let scalar_result = x_vec
            .iter()
            .zip(y_vec.iter())
            .zip(z_vec.iter())
            .zip(w_vec.iter())
            .map(|(((x, y), z), w)| (x * x + y * y + z * z + w * w).sqrt())
            .collect::<Vec<f32>>()[0];
        let simd_result = x_vec.hypot4(y_vec, z_vec, w_vec)[0];

        println!("Edge case: hypot4({x:.6}, {y:.6}, {z:.6}, {w:.6}) = {scalar_result:.8} (expected: {expected:.8})");
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
                "Scalar hypot4 relative error too large for ({x}, {y}, {z}, {w}): {scalar_rel_error:.2e}"
            );
            assert!(
                simd_rel_error < 1e-5,
                "SIMD hypot4 relative error too large for ({x}, {y}, {z}, {w}): {simd_rel_error:.2e}"
            );
        } else {
            assert!(
                scalar_error < 1e-7,
                "Scalar hypot4 absolute error too large for ({x}, {y}, {z}, {w}): {scalar_error:.2e}"
            );
            assert!(
                simd_error < 1e-6,
                "SIMD hypot4 absolute error too large for ({x}, {y}, {z}, {w}): {simd_error:.2e}"
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
                "SIMD vs scalar relative error too large for ({x}, {y}, {z}, {w}): {simd_vs_scalar_rel_error:.2e}"
            );
        } else {
            assert!(
                simd_vs_scalar_error < 1e-6,
                "SIMD vs scalar absolute error too large for ({x}, {y}, {z}, {w}): {simd_vs_scalar_error:.2e}"
            );
        }
    }
}

/// Test coordinate permutation invariance.
#[test]
fn test_hypot4_permutation_invariance() {
    let x_values: Vec<f32> = vec![3.0, 5.0, 1.0, -2.0];
    let y_values: Vec<f32> = vec![4.0, 12.0, 1.0, 3.0];
    let z_values: Vec<f32> = vec![5.0, 9.0, 1.0, 1.0];
    let w_values: Vec<f32> = vec![12.0, 8.0, 1.0, 4.0];

    let result_xyzw = x_values.hypot4(y_values.clone(), z_values.clone(), w_values.clone());
    let result_yzwx = y_values.hypot4(z_values.clone(), w_values.clone(), x_values.clone());
    let result_zwxy = z_values.hypot4(w_values.clone(), x_values.clone(), y_values.clone());

    for i in 0..x_values.len() {
        let x_val = x_values[i];
        let y_val = y_values[i];
        let z_val = z_values[i];
        let w_val = w_values[i];
        let val_xyzw = result_xyzw[i];
        let val_yzwx = result_yzwx[i];
        let val_zwxy = result_zwxy[i];

        println!(
            "Permutation test: hypot4({x_val:.6}, {y_val:.6}, {z_val:.6}, {w_val:.6}) = {val_xyzw:.8}"
        );

        let error_xyzw_yzwx = (val_xyzw - val_yzwx).abs();
        let error_xyzw_zwxy = (val_xyzw - val_zwxy).abs();
        let rel_error_xyzw_yzwx = if val_xyzw != 0.0 {
            error_xyzw_yzwx / val_xyzw.abs()
        } else {
            error_xyzw_yzwx
        };
        let rel_error_xyzw_zwxy = if val_xyzw != 0.0 {
            error_xyzw_zwxy / val_xyzw.abs()
        } else {
            error_xyzw_zwxy
        };

        assert!(
            error_xyzw_yzwx < 1e-6 || rel_error_xyzw_yzwx < 1e-6,
            "Hypot4 should be permutation-invariant for ({x_val}, {y_val}, {z_val}, {w_val}): xyzw vs yzwx error={error_xyzw_yzwx:.2e}, rel_error={rel_error_xyzw_yzwx:.2e}"
        );

        assert!(
            error_xyzw_zwxy < 1e-6 || rel_error_xyzw_zwxy < 1e-6,
            "Hypot4 should be permutation-invariant for ({x_val}, {y_val}, {z_val}, {w_val}): xyzw vs zwxy error={error_xyzw_zwxy:.2e}, rel_error={rel_error_xyzw_zwxy:.2e}"
        );
    }
}

/// Test precision with very small values near zero.
#[test]
fn test_hypot4_precision_near_zero() {
    let small_values: Vec<f32> = vec![1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];

    let scalar_results: Vec<f32> = small_values
        .iter()
        .map(|&val| (val * val + val * val + val * val + val * val).sqrt()) // x=y=z=w for simplicity
        .collect();
    let simd_results = small_values.hypot4(
        small_values.clone(),
        small_values.clone(),
        small_values.clone(),
    );

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let val = small_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Small values: hypot4({val:.2e}, {val:.2e}, {val:.2e}, {val:.2e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // For small values, hypot4 should be accurate
        // SIMD should match scalar closely
        if scalar_val != 0.0 {
            assert!(
                relative_error < 1e-6,
                "SIMD precision error too large for small inputs val={val}: {relative_error:.2e}"
            );
        } else {
            assert!(
                absolute_error < 1e-7,
                "SIMD precision error too large for small inputs val={val}: {absolute_error:.2e}"
            );
        }
    }
}

/// Test precision with very large values that might cause overflow.
#[test]
fn test_hypot4_precision_large_values() {
    let large_values: Vec<f32> = vec![
        1e6, 1e7, 1e8, 1e9, 1e10, 1e15, 1e18,
        // 1e19
    ];

    let scalar_results: Vec<f32> = large_values
        .iter()
        .map(|&val| (val * val + val * val + val * val + val * val).sqrt()) // x=y=z=w for simplicity
        .collect();
    let simd_results = large_values.hypot4(
        large_values.clone(),
        large_values.clone(),
        large_values.clone(),
    );

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let val = large_values[i];
        let absolute_error = (scalar_val - simd_val).abs();
        let relative_error = if scalar_val != 0.0 {
            absolute_error / scalar_val.abs()
        } else {
            absolute_error
        };

        println!(
            "Large values: hypot4({val:.1e}, {val:.1e}, {val:.1e}, {val:.1e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}, Rel: {relative_error:.2e}"
        );

        // For large values, both results should be finite (no overflow)
        assert!(scalar_val.is_finite(), "Scalar result should be finite");
        assert!(simd_val.is_finite(), "SIMD result should be finite");

        // SIMD should match scalar reasonably well
        if scalar_val.is_finite() && simd_val.is_finite() {
            assert!(
                relative_error < 1e-4,
                "SIMD precision error too large for large inputs val={val}: {relative_error:.2e}"
            );
        }
    }
}
