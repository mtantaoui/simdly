//! Simdly library demonstration binary.
//!
//! This is a simple placeholder main function for the simdly library.
//! The actual SIMD functionality is provided through the library modules.

fn main() {
    let a: Vec<i32> = (0..100).collect();
    let b: Vec<i32> = (1..10).collect();

    a.iter().zip(b).for_each(|(a, b)| println!("{}, {}", a, b));
}
