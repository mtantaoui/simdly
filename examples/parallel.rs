//! `ParSliceMath` and `ParSliceExt`: the math functions, arithmetic and
//! comparisons on several threads, for long slices.
//!
//! ```sh
//! cargo run --release --example parallel --features rayon
//! ```

use std::time::Instant;

use simdly::{PARALLEL_SIMD_THRESHOLD, ParSliceExt, ParSliceMath, SliceExt, SliceMath};

fn main() {
    let n = 16 * 1024 * 1024;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 1e-4).collect();
    let y: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 1e-4).collect();

    println!(
        "backend: {}, threads: {}",
        simdly::backend(),
        rayon::current_num_threads()
    );
    println!(
        "{n} elements; on one thread up to {PARALLEL_SIMD_THRESHOLD} (math) \
         and twice that (arithmetic, comparisons)\n"
    );

    let t = Instant::now();
    let one = x.sin();
    let t_one = t.elapsed();
    let t = Instant::now();
    let par = x.par_sin();
    let t_par = t.elapsed();
    // The same results, bit for bit.
    assert!(
        one.iter()
            .zip(&par)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    println!("sin:       one thread {t_one:>10.2?}, par_sin   {t_par:>10.2?}");

    let t = Instant::now();
    let one = x.hypot(&y);
    let t_one = t.elapsed();
    let t = Instant::now();
    let par = x.par_hypot(&y);
    let t_par = t.elapsed();
    assert_eq!(one, par);
    println!("hypot:     one thread {t_one:>10.2?}, par_hypot {t_par:>10.2?}");

    // In place, and on `f64`.
    let mut z: Vec<f64> = (0..n).map(|i| i as f64 * 1e-4).collect();
    let t = Instant::now();
    z.par_exp_in_place();
    println!("exp, f64:  par_exp_in_place {:>10.2?}", t.elapsed());

    // Arithmetic and comparisons: limited by memory bandwidth, so threads
    // help less than for `sin`.
    let t = Instant::now();
    let one = x.mul(&y);
    let t_one = t.elapsed();
    let t = Instant::now();
    let par = x.par_mul(&y);
    let t_par = t.elapsed();
    assert_eq!(one, par);
    println!("mul:       one thread {t_one:>10.2?}, par_mul   {t_par:>10.2?}");

    let mut w = x.clone();
    w.par_add_assign(&y);
    // x + y = 1, up to the rounding of x and y.
    assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-2));

    let t = Instant::now();
    let one = x.less_scalar(800.0);
    let t_one = t.elapsed();
    let t = Instant::now();
    let par = x.par_less_scalar(800.0);
    let t_par = t.elapsed();
    assert_eq!(one, par);
    let below = par.iter().filter(|&&m| m).count();
    println!("x < 800:   one thread {t_one:>10.2?}, par_less_scalar {t_par:>10.2?} ({below} true)");

    // Short slices run on the calling thread, as `SliceMath` does.
    let short = vec![0.0_f32, 1.0, 4.0];
    println!("\n√{short:?} = {:?}", short.par_sqrt());
}
