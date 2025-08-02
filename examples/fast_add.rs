use simdly::FastAdd;

fn main() {
    // Example 1: Basic vector addition
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    
    println!("Vector A: {:?}", a);
    println!("Vector B: {:?}", b);
    
    let result = a.fast_add(b);
    println!("A + B = {:?}", result);
    
    // Example 2: Large vector addition to demonstrate SIMD performance
    let size = 10_000;
    let large_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let large_b: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();
    
    println!("\nLarge vector addition (size: {})", size);
    let start = std::time::Instant::now();
    let _large_result = large_a.fast_add(large_b);
    let duration = start.elapsed();
    println!("Fast addition completed in: {:?}", duration);
    
    // Example 3: Small vector to show scalar fallback
    let small_a = vec![1.0f32, 2.0];
    let small_b = vec![3.0f32, 4.0];
    
    println!("\nSmall vector addition:");
    println!("A: {:?}", small_a);
    println!("B: {:?}", small_b);
    let small_result = small_a.fast_add(small_b);
    println!("Result: {:?}", small_result);
}