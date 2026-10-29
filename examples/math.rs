//! Element-wise math functions on `Vec<f32>` and `Vec<f64>` with `SliceMath`.
//!
//! ```sh
//! cargo run --example math
//! ```

use simdly::SliceMath;

fn main() {
    let x = vec![-2.5_f32, -1.0, -0.0, 0.0, 1.0, f32::NEG_INFINITY, f32::NAN];

    println!("backend: {}\n", simdly::backend());
    println!("x      = {x:?}");
    println!("|x|    = {:?}", x.abs());
    println!("√x     = {:?}", x.sqrt());
    println!("∛x     = {:?}", x.cbrt());
    println!("acos x = {:?}", x.acos());
    println!("asin x = {:?}", x.asin());
    println!("atan x = {:?}", x.atan());
    let ones = vec![-1.0_f32; x.len()];
    println!("atan2(x, -1) = {:?}", x.atan2(&ones));

    // In place, and on `f64`.
    let mut y = vec![-0.1_f64, 0.2, -0.3];
    y.abs_in_place();
    println!("\n|y|    = {y:?}");
    y.sqrt_in_place();
    println!("√|y|   = {y:?}");
}
