use simdly::simd::SimdMath;

#[test]
fn test_parallel_abs_small_array() {
    let data = vec![1.0f32, -2.0, 3.0, -4.0];
    let result = data.as_slice().par_abs();
    let expected = vec![1.0f32, 2.0, 3.0, 4.0];
    
    for (actual, expected) in result.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6, "par_abs failed: {} != {}", actual, expected);
    }
}

#[test] 
fn test_parallel_abs_large_array() {
    let data: Vec<f32> = (0..100_000).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
    let result = data.as_slice().par_abs();
    
    for (i, &val) in result.iter().enumerate() {
        let expected = i as f32;
        assert!((val - expected).abs() < 1e-6, "par_abs large array failed at index {}: {} != {}", i, val, expected);
    }
}

#[test]
fn test_parallel_cos_methods() {
    let data = vec![0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI];
    
    // Test slice method
    let result_slice = data.as_slice().par_cos();
    assert!((result_slice[0] - 1.0).abs() < 1e-5, "cos(0) should be ~1.0");
    assert!(result_slice[1].abs() < 1e-5, "cos(π/2) should be ~0.0");  
    assert!((result_slice[2] + 1.0).abs() < 1e-5, "cos(π) should be ~-1.0");
    
    // Test Vec method
    let result_vec = data.par_cos();
    assert!((result_vec[0] - 1.0).abs() < 1e-5, "Vec cos(0) should be ~1.0");
}

#[test]
fn test_parallel_sin_methods() {
    let data = vec![0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI];
    
    // Test slice method  
    let result_slice = data.as_slice().par_sin();
    assert!(result_slice[0].abs() < 1e-5, "sin(0) should be ~0.0");
    assert!((result_slice[1] - 1.0).abs() < 1e-5, "sin(π/2) should be ~1.0");
    assert!(result_slice[2].abs() < 1e-5, "sin(π) should be ~0.0");
    
    // Test Vec method
    let result_vec = data.par_sin();
    assert!(result_vec[0].abs() < 1e-5, "Vec sin(0) should be ~0.0");
}

#[test]
fn test_parallel_sqrt_methods() {
    let data = vec![0.0f32, 1.0, 4.0, 9.0, 16.0];
    let expected = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
    
    // Test slice method
    let result_slice = data.as_slice().par_sqrt();
    for (actual, expected) in result_slice.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6, "par_sqrt failed: {} != {}", actual, expected);
    }
    
    // Test Vec method
    let result_vec = data.par_sqrt();
    for (actual, expected) in result_vec.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6, "Vec par_sqrt failed: {} != {}", actual, expected);
    }
}

#[test]
fn test_parallel_exp_methods() {
    let data = vec![0.0f32, 1.0, 2.0];
    
    // Test slice method
    let result_slice = data.as_slice().par_exp();
    assert!((result_slice[0] - 1.0).abs() < 1e-5, "exp(0) should be ~1.0");
    assert!((result_slice[1] - std::f32::consts::E).abs() < 1e-4, "exp(1) should be ~e");
    
    // Test Vec method
    let result_vec = data.par_exp();
    assert!((result_vec[0] - 1.0).abs() < 1e-5, "Vec exp(0) should be ~1.0");
}

#[test]
fn test_parallel_ln_methods() {
    let data = vec![1.0f32, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E];
    
    // Test slice method
    let result_slice = data.as_slice().par_ln();
    assert!(result_slice[0].abs() < 1e-5, "ln(1) should be ~0.0");
    assert!((result_slice[1] - 1.0).abs() < 1e-4, "ln(e) should be ~1.0");
    assert!((result_slice[2] - 2.0).abs() < 1e-4, "ln(e²) should be ~2.0");
    
    // Test Vec method
    let result_vec = data.par_ln();
    assert!(result_vec[0].abs() < 1e-5, "Vec ln(1) should be ~0.0");
}

#[test]
fn test_parallel_atan2_methods() {
    let y_data = vec![1.0f32, 1.0, -1.0, -1.0];
    let x_data = vec![1.0f32, -1.0, -1.0, 1.0];
    
    // Test slice method
    let result_slice = y_data.as_slice().par_atan2(x_data.as_slice());
    
    // atan2(1, 1) = π/4
    assert!((result_slice[0] - std::f32::consts::PI / 4.0).abs() < 1e-4);
    
    // Test Vec method  
    let result_vec = y_data.par_atan2(x_data);
    assert!((result_vec[0] - std::f32::consts::PI / 4.0).abs() < 1e-4);
}

#[test]
fn test_parallel_floor_ceil_methods() {
    let data = vec![1.1f32, 2.7, -1.5, -2.9];
    let expected_floor = vec![1.0f32, 2.0, -2.0, -3.0];
    let expected_ceil = vec![2.0f32, 3.0, -1.0, -2.0];
    
    // Test floor
    let result_floor = data.as_slice().par_floor();
    for (actual, expected) in result_floor.iter().zip(expected_floor.iter()) {
        assert!((actual - expected).abs() < 1e-6, "par_floor failed: {} != {}", actual, expected);
    }
    
    // Test ceil
    let result_ceil = data.as_slice().par_ceil();
    for (actual, expected) in result_ceil.iter().zip(expected_ceil.iter()) {
        assert!((actual - expected).abs() < 1e-6, "par_ceil failed: {} != {}", actual, expected);
    }
}

#[test]
fn test_match_statement_threshold_selection() {
    // Test that small arrays use regular SIMD (threshold-based selection)
    let small_data = vec![1.0f32; 10];
    let result_small = small_data.as_slice().par_abs();
    assert_eq!(result_small.len(), 10);
    
    // Test that large arrays can use parallel SIMD 
    let large_data = vec![1.0f32; 50_000];
    let result_large = large_data.as_slice().par_abs();
    assert_eq!(result_large.len(), 50_000);
    
    // Both should produce correct results
    for &val in result_small.iter() {
        assert!((val - 1.0).abs() < 1e-6);
    }
    
    for &val in result_large.iter() {
        assert!((val - 1.0).abs() < 1e-6);
    }
}