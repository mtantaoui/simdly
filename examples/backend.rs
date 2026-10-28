//! Prints the SIMD backend simdly uses on this CPU.
//!
//! ```sh
//! cargo run --example backend
//! ```

use simdly::Backend;

fn main() {
    println!("simdly uses the {} backend", simdly::backend());

    println!("\nBackends, fastest first:");
    for b in [
        Backend::Avx512,
        Backend::Avx2,
        Backend::Neon,
        Backend::Scalar,
    ] {
        let status = if b.is_available() { "available" } else { "-" };
        println!("  {:<7} {status}", b.name());
    }
}
