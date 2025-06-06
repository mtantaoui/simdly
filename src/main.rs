use simdly::simd::traits::SimdAdd;

fn main() {
    let v1 = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let v2 = vec![
        10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0,
    ];

    // Call the function directly
    let v3 = v1.simd_add(&v2);
    println!("{:?}", v3); // Prints: [11.0, 22.0, 33.0]

    let v3 = v1.par_simd_add(&v2);
    println!("{:?}", v3); // Prints: [11.0, 22.0, 33.0]

    let v3 = v1.scalar_add(&v2);
    println!("{:?}", v3); // Prints: [11.0, 22.0, 33.0]
}
