use simdly::simd::traits::SimdAdd;

fn main() {
    let n: usize = 10;

    let u: Vec<f32> = vec![1.0; n];
    let v: Vec<f32> = vec![1.0; n];

    let w = u.simd_add(&v);
    println!("{:?}", w);
}
