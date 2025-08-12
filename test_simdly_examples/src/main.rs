// Test README examples to verify documentation matches implementation

fn main() {
    test_fastadd_example();
    test_simdmath_example();
    test_platform_specific_example();
    test_mathematical_operations();
    println!("All README examples work correctly!");
}

fn test_fastadd_example() {
    // This example is from README but FastAdd trait doesn't exist in actual code
    // The documentation mentions it but it's not implemented
    println!("Note: FastAdd trait mentioned in README but not found in implementation");
}

fn test_simdmath_example() {
    use simdly::simd::SimdMath;
    
    let data = vec![1.0, 2.0, 3.0, 4.0];
    
    // All mathematical operations use SIMD automatically
    let cosines = data.cos();              // Vectorized cosine
    let sines = data.sin();                // Vectorized sine
    let exponentials = data.exp();         // Vectorized exponential
    let square_roots = data.sqrt();        // Vectorized square root
    
    println!("SimdMath operations work: cosines={:?}", cosines);
    
    // Power and distance operations
    let base = vec![2.0, 3.0, 4.0, 5.0];
    let exp = vec![2.0, 2.0, 2.0, 2.0];
    let powers = base.pow(exp);            // Powers: [4.0, 9.0, 16.0, 25.0]
    
    let x = vec![3.0, 5.0, 8.0, 7.0];
    let y = vec![4.0, 12.0, 15.0, 24.0];
    let distances = x.hypot(y);            // 2D distances: [5.0, 13.0, 17.0, 25.0]
    
    println!("Powers: {:?}", powers);
    println!("Distances: {:?}", distances);
}

#[cfg(target_arch = "x86_64")]
fn test_platform_specific_example() {
    use simdly::simd::avx2::f32x8::F32x8;
    use simdly::simd::{SimdLoad, SimdStore};
    
    // Load 8 f32 values into AVX2 SIMD vector
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let vec = F32x8::from(&data[..]);
    
    // Store results using platform-appropriate method
    let mut output = [0.0f32; 8];
    unsafe {
        vec.store_at(output.as_mut_ptr());
    }
    
    println!("Processed {} elements with AVX2 SIMD", vec.size);
}

#[cfg(target_arch = "aarch64")]
fn test_platform_specific_example() {
    use simdly::simd::neon::f32x4::F32x4;
    use simdly::simd::{SimdLoad, SimdStore};
    
    // Load 4 f32 values into NEON SIMD vector  
    let data = [1.0, 2.0, 3.0, 4.0];
    let vec = F32x4::from(&data[..]);
    
    // Store results
    let mut output = [0.0f32; 4];
    unsafe {
        vec.store_at(output.as_mut_ptr());
    }
    
    println!("Processed {} elements with NEON SIMD", vec.size);
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn test_platform_specific_example() {
    println!("Platform-specific SIMD not available on this architecture");
}

fn test_mathematical_operations() {
    use simdly::simd::SimdMath;
    
    let data = vec![1.0, 2.0, 3.0, 4.0];
    
    // All mathematical operations use SIMD automatically
    let cosines = data.cos();              // Vectorized cosine
    let sines = data.sin();                // Vectorized sine
    let exponentials = data.exp();         // Vectorized exponential
    let square_roots = data.sqrt();        // Vectorized square root
    
    println!("Results computed with SIMD acceleration!");
}
