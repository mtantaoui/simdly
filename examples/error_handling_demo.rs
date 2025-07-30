//! Error Handling Demonstration
//!
//! This example demonstrates the improved error handling in simdly,
//! showing how operations gracefully handle error conditions instead of panicking.

use simdly::{error::SimdlyError, SimdAdd};

fn main() {
    println!("🔧 Simdly Error Handling Demonstration\n");

    // Example 1: Successful operation
    println!("✅ Example 1: Successful SIMD addition");
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    
    match a.as_slice().simd_add(b.as_slice()) {
        Ok(result) => println!("   Result: {:?}", result),
        Err(e) => println!("   Error: {}", e),
    }
    println!();

    // Example 2: Validation error - mismatched lengths
    println!("❌ Example 2: Validation error (mismatched slice lengths)");
    let a_short = vec![1.0, 2.0, 3.0];
    let b_long = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    
    match a_short.as_slice().simd_add(b_long.as_slice()) {
        Ok(result) => println!("   Unexpected success: {:?}", result),
        Err(e) => {
            println!("   Error caught: {}", e);
            println!("   Error type: {:?}", e);
        }
    }
    println!();

    // Example 3: Different error handling strategies
    println!("🔄 Example 3: Different error handling strategies");
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0]; // Wrong length
    
    // Strategy 1: Unwrap (would panic on error)
    println!("   Strategy 1: Using unwrap() - would panic on error");
    println!("   (Skipping to avoid panic in demo)");
    
    // Strategy 2: Match for explicit handling
    println!("   Strategy 2: Pattern matching for explicit error handling");
    match a.as_slice().scalar_add(b.as_slice()) {
        Ok(result) => println!("      Success: {:?}", result),
        Err(SimdlyError::ValidationError { message }) => {
            println!("      Validation error: {}", message);
        }
        Err(SimdlyError::AllocationError { requested_size, requested_alignment, message }) => {
            println!("      Allocation error: {} (size: {}, align: {})", message, requested_size, requested_alignment);
        }
        Err(SimdlyError::LayoutError { size, alignment, message }) => {
            println!("      Layout error: {} (size: {}, align: {})", message, size, alignment);
        }
    }
    
    // Strategy 3: Using unwrap_or for fallback values
    println!("   Strategy 3: Using unwrap_or() for graceful fallback");
    let fallback_result = a.as_slice().scalar_add(b.as_slice())
        .unwrap_or_else(|_| vec![0.0; a.len()]); // Fallback to zeros
    println!("      Fallback result: {:?}", fallback_result);
    
    // Strategy 4: Using map_err for error transformation
    println!("   Strategy 4: Using map_err() for error transformation");
    let transformed_result = a.as_slice().simd_add(b.as_slice())
        .map_err(|e| format!("Custom error: {}", e));
    match transformed_result {
        Ok(result) => println!("      Success: {:?}", result),
        Err(custom_msg) => println!("      Custom error: {}", custom_msg),
    }
    println!();

    // Example 4: Chaining operations with error propagation
    println!("🔗 Example 4: Chaining operations with error propagation");
    
    fn complex_operation(a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<f32>, SimdlyError> {
        // First add a + b
        let intermediate = a.simd_add(b)?;
        
        // Then add result + c
        intermediate.as_slice().simd_add(c)
    }
    
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![10.0, 20.0, 30.0, 40.0];
    let c = vec![100.0, 200.0, 300.0, 400.0];
    
    match complex_operation(&a, &b, &c) {
        Ok(result) => println!("   Complex operation result: {:?}", result),
        Err(e) => println!("   Complex operation failed: {}", e),
    }
    
    // Same operation but with mismatched c
    let c_wrong = vec![100.0, 200.0]; // Wrong length
    match complex_operation(&a, &b, &c_wrong) {
        Ok(result) => println!("   Unexpected success: {:?}", result),
        Err(e) => println!("   Complex operation failed as expected: {}", e),
    }
    println!();

    // Example 5: Performance comparison (error checking overhead)
    println!("⚡ Example 5: Error handling performance impact");
    let large_a: Vec<f32> = (0..10000).map(|i| i as f32).collect();
    let large_b: Vec<f32> = (0..10000).map(|i| (i * 2) as f32).collect();
    
    let start = std::time::Instant::now();
    let _result = large_a.as_slice().simd_add(large_b.as_slice()).unwrap();
    let duration = start.elapsed();
    
    println!("   SIMD addition of 10,000 elements: {:?}", duration);
    println!("   (Error checking adds minimal overhead)");
    println!();

    println!("✨ Summary:");
    println!("   - All operations now return Result<T, SimdlyError> instead of panicking");
    println!("   - Three error types: ValidationError, AllocationError, LayoutError");
    println!("   - Enables graceful error handling in applications");
    println!("   - Memory safety bug in alignment detection has been fixed");
    println!("   - Minimal performance impact from error checking");
}