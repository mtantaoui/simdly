//! Precision comparison tests between scalar and SIMD ceiling implementations.
//!
//! This test suite validates that the SIMD ceiling implementation maintains
//! acceptable precision compared to the standard library scalar implementation.

use simdly::simd::SimdMath;

/// Test precision of SIMD ceil against scalar ceil for various input ranges.
#[test]
fn test_ceil_precision_comparison() {
    // Test cases covering different ranges and edge cases
    let test_cases = vec![
        // Small positive values
        vec![0.0f32, 0.1, 0.5, 0.9],
        // Values around 1
        vec![0.9f32, 1.0, 1.1, 1.9],
        // Integer values
        vec![0.0f32, 1.0, 2.0, 3.0, 10.0],
        // Larger positive values
        vec![10.1f32, 10.5, 10.9, 100.5],
        // Small negative values
        vec![-0.1f32, -0.5, -0.9],
        // Values around -1
        vec![-0.9f32, -1.0, -1.1, -1.9],
        // Negative integer values
        vec![-1.0f32, -2.0, -3.0, -10.0],
        // Larger negative values
        vec![-10.1f32, -10.5, -10.9, -100.5],
        // Mixed values
        vec![-2.7f32, -1.2, 0.0, 1.8, 3.3],
        // Very small fractional parts
        vec![1.0001f32, 1.9999, -1.0001, -1.9999],
        // Large values
        vec![1000.1f32, 1000.9, -1000.1, -1000.9],
    ];

    for (i, test_case) in test_cases.iter().enumerate() {
        println!("Testing case {}: {:?}", i + 1, test_case);

        // Compute scalar results
        let scalar_results: Vec<f32> = test_case.iter().map(|x| x.ceil()).collect();

        // Compute SIMD results
        let simd_results = test_case.ceil();

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

            println!(
                "  Input: {input_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs Error: {absolute_error:.2e}"
            );

            // Ceil should be exact for floating point values
            assert!(
                absolute_error < 1e-6,
                "Precision error too large for input {input_val}: scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}"
            );
        }
    }
}

/// Test precision with randomly generated inputs across wide ranges.
#[test]
fn test_ceil_precision_random_inputs() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(12345);
    let test_size = 1000;

    // Generate random inputs in range [-1000, 1000]
    let inputs: Vec<f32> = (0..test_size)
        .map(|_| rng.random_range(-1000.0..=1000.0))
        .collect();

    let scalar_results: Vec<f32> = inputs.iter().map(|x| x.ceil()).collect();
    let simd_results = inputs.ceil();

    let mut max_abs_error = 0.0f32;
    let mut error_count = 0;

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = inputs[i];
        let absolute_error = (scalar_val - simd_val).abs();

        max_abs_error = max_abs_error.max(absolute_error);

        // Count any errors (ceil should be exact)
        if absolute_error > 1e-7 {
            error_count += 1;
            if error_count <= 10 {
                // Print first 10 errors
                println!(
                    "Error #{error_count}: Input: {input_val:.6}, Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Abs: {absolute_error:.2e}"
                );
            }
        }

        // Individual assertion with very strict tolerance for ceil
        assert!(
            absolute_error < 1e-6,
            "Precision error too large at index {i}: input={input_val}, scalar={scalar_val}, simd={simd_val}, abs_error={absolute_error:.2e}"
        );
    }

    println!("Random precision test summary:");
    println!("  Test size: {test_size}");
    println!("  Max absolute error: {max_abs_error:.2e}");
    println!("  Any errors (>1e-7): {error_count}");

    // Overall precision requirements - ceil should be exact
    assert!(
        max_abs_error < 1e-6,
        "Maximum absolute error too large: {max_abs_error:.2e}"
    );
    assert!(
        error_count == 0,
        "Any errors are unacceptable for ceil function: {error_count}/{test_size}"
    );
}

/// Test edge cases and special values.
#[test]
fn test_ceil_edge_cases() {
    let edge_cases = vec![
        // Exact values where ceil should be known
        (0.0f32, 0.0f32),           // ceil(0) = 0
        (1.0f32, 1.0f32),           // ceil(1) = 1
        (-1.0f32, -1.0f32),         // ceil(-1) = -1
        (1.5f32, 2.0f32),           // ceil(1.5) = 2
        (-1.5f32, -1.0f32),         // ceil(-1.5) = -1
        (2.9f32, 3.0f32),           // ceil(2.9) = 3
        (-2.9f32, -2.0f32),         // ceil(-2.9) = -2
        (0.1f32, 1.0f32),           // ceil(0.1) = 1
        (-0.1f32, 0.0f32),          // ceil(-0.1) = 0
        (0.9f32, 1.0f32),           // ceil(0.9) = 1
        (-0.9f32, 0.0f32),          // ceil(-0.9) = 0
        (42.0f32, 42.0f32),         // ceil(42) = 42
        (-42.0f32, -42.0f32),       // ceil(-42) = -42
    ];

    for (input, expected) in edge_cases {
        let input_vec = vec![input];
        let scalar_result = input_vec.iter().map(|x| x.ceil()).collect::<Vec<f32>>()[0];
        let simd_result = input_vec.ceil()[0];

        println!("Edge case: ceil({input:.6}) = {scalar_result:.8} (expected: {expected:.8})");
        println!("  Scalar: {scalar_result:.8}, SIMD: {simd_result:.8}");

        // Compare against expected value (should be exact)
        let scalar_error = (scalar_result - expected).abs();
        let simd_error = (simd_result - expected).abs();

        println!("  Scalar error: {scalar_error:.2e}, SIMD error: {simd_error:.2e}");

        // Both should be exactly equal to expected value
        assert!(
            scalar_error < 1e-7,
            "Scalar ceil error too large for {input}: {scalar_error:.2e}"
        );
        assert!(
            simd_error < 1e-7,
            "SIMD ceil error too large for {input}: {simd_error:.2e}"
        );

        // SIMD should be exactly equal to scalar
        let simd_vs_scalar_error = (simd_result - scalar_result).abs();
        assert!(
            simd_vs_scalar_error < 1e-7,
            "SIMD vs scalar error too large for {input}: {simd_vs_scalar_error:.2e}"
        );
    }
}

/// Test precision with fractional values near integer boundaries.
#[test]
fn test_ceil_precision_near_integers() {
    let boundary_values: Vec<f32> = vec![
        0.9999, 1.0001, 1.9999, 2.0001, -0.0001, -0.9999, -1.0001, -1.9999, -2.0001,
        9.9999, 10.0001, -9.9999, -10.0001,
        99.9999, 100.0001, -99.9999, -100.0001,
    ];

    let scalar_results: Vec<f32> = boundary_values.iter().map(|x| x.ceil()).collect();
    let simd_results = boundary_values.ceil();

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = boundary_values[i];
        let absolute_error = (scalar_val - simd_val).abs();

        println!(
            "Boundary value: ceil({input_val:.6}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}"
        );

        // Ceil should be exact even near integer boundaries
        assert!(
            absolute_error < 1e-7,
            "SIMD precision error too large for boundary input {input_val}: {absolute_error:.2e}"
        );

        // Verify the result makes mathematical sense
        assert!(
            scalar_val >= input_val && scalar_val <= input_val + 1.0,
            "Ceil result should be between input and input+1"
        );
        assert!(
            simd_val >= input_val && simd_val <= input_val + 1.0,
            "Ceil result should be between input and input+1"
        );
    }
}

/// Test precision with very large values.
#[test]
fn test_ceil_precision_large_values() {
    let large_values: Vec<f32> = vec![
        1000000.1, 1000000.9, -1000000.1, -1000000.9,
        f32::MAX - 1.0, -f32::MAX + 1.0,
    ];

    let scalar_results: Vec<f32> = large_values.iter().map(|x| x.ceil()).collect();
    let simd_results = large_values.ceil();

    for (i, (&scalar_val, &simd_val)) in scalar_results.iter().zip(simd_results.iter()).enumerate()
    {
        let input_val = large_values[i];
        let absolute_error = (scalar_val - simd_val).abs();

        println!(
            "Large value: ceil({input_val:.1e}) -> Scalar: {scalar_val:.8}, SIMD: {simd_val:.8}, Error: {absolute_error:.2e}"
        );

        // Ceil should be exact even for large values
        assert!(
            absolute_error < 1e-5, // Slightly relaxed for very large values due to floating point precision limits
            "SIMD precision error too large for large input {input_val}: {absolute_error:.2e}"
        );
    }
}