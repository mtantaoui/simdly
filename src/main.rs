use simdly::simd::traits::SimdCos;

fn main() {
    let n: usize = 10;

    let u: Vec<f32> = (0..n).map(|i| i as f32).collect();
    // let v: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let w1 = u.simd_cos();
    let w2 = u.scalar_cos();

    for i in 0..n {
        println!("{} -- {}", w1[i], w2[i]);
    }
}
