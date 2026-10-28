//! Element-wise `+ - * /` on `Vec<f32>` and `Vec<f64>` with `SliceExt`.
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

    // `f64` works the same way.
    let x = vec![0.1_f64, 0.2, 0.3];
    println!("\nf64: {x:?} + {x:?} = {:?}", x.add(&x));
}
