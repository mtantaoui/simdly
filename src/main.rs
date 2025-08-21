fn main() {
    let n = 9;
    let size = n * 90;
    let result_flat = vec![0.0f32; size];

    let result_2d: Vec<Vec<f32>> = result_flat.chunks(n).map(|chunk| chunk.to_vec()).collect();

    println!("{:?}", result_2d)
}
