use simdly::simd::traits::SimdAdd;

fn main() {
    let n = 24;

    let v1: Vec<f32> = (0..n).into_iter().map(|i| i as f32).collect();
    let v2: Vec<f32> = (0..n).into_iter().map(|i| i as f32).collect();

    // Call the function directly
    let v3 = v1.simd_add(&v2);
    println!("{:?}", v3); // Prints: [11.0, 22.0, 33.0]

    let v3 = v1.par_simd_add(&v2);
    println!("{:?}", v3); // Prints: [11.0, 22.0, 33.0]

    // let v3 = v1.scalar_add(&v2);
    // println!("{:?}", v3); // Prints: [11.0, 22.0, 33.0]
}
