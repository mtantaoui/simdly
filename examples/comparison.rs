//! Element-wise comparisons on `Vec<f32>` with `SliceExt`, like `numpy.less`.
//!
//! ```sh
//! cargo run --example comparison
//! ```

use simdly::SliceExt;

fn main() {
    let a = vec![1.0_f32, 5.0, f32::NAN, 2.0, -0.0, 7.5];
    let b = vec![2.0_f32, 3.0, 1.0, 2.0, 0.0, 7.5];

    println!("backend: {}\n", simdly::backend());
    println!("a = {a:?}");
    println!("b = {b:?}\n");
    println!("a <  b  = {:?}", a.less(&b));
    println!("a <= b  = {:?}", a.less_equal(&b));
    println!("a >  b  = {:?}", a.greater(&b));
    println!("a >= b  = {:?}", a.greater_equal(&b));
    println!("a == b  = {:?}", a.equal(&b));
    println!("a != b  = {:?}", a.not_equal(&b));
    println!("\na < 2.0 = {:?}", a.less_scalar(2.0));
    println!("\nNaN compares false, except `!=`; -0.0 == 0.0.");
}
