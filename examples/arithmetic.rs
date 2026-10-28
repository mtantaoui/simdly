//! Element-wise `+ - * /` on `Vec<f32>` with `SliceExt`.
//!
//! ```sh
//! cargo run --example arithmetic
//! ```

use simdly::SliceExt;

fn main() {
    let a: Vec<f32> = (1..=10).map(|i| i as f32).collect();
    let b = vec![2.0_f32; 10];

    println!("backend: {}\n", simdly::backend());
    println!("a     = {a:?}");
    println!("b     = {b:?}\n");
    println!("a + b = {:?}", a.add(&b));
    println!("a - b = {:?}", a.sub(&b));
    println!("a * b = {:?}", a.mul(&b));
    println!("a / b = {:?}", a.div(&b));

    // In place: no allocation.
    let mut c = a.clone();
    c.mul_assign(&b);
    c.sub_assign(&a);
    println!("\na * b - a, in place = {c:?}");
}
