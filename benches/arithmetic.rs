//! `SliceExt` arithmetic against the plain scalar loop, from L1-resident to
//! larger-than-cache slices.
//!
//! ```sh
//! cargo bench --bench arithmetic
//! ```

use divan::{Bencher, black_box};
use simdly::SliceExt;

fn main() {
    divan::main();
}

const LENS: &[usize] = &[64, 4_096, 262_144, 4_194_304];

fn inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..len).map(|i| i as f32 * 0.5 + 1.0).collect();
    let b = (0..len).map(|i| 3.0 - i as f32 * 0.25).collect();
    (a, b)
}

/// `b` is all ones so that repeating `a op= b` keeps `a` finite and normal:
/// drifting into subnormals would slow both sides down and skew the result.
fn inputs_in_place(len: usize) -> (Vec<f32>, Vec<f32>) {
    (inputs(len).0, vec![1.0; len])
}

macro_rules! bench_op {
    ($name:ident, $assign:ident, $op:tt, $op_assign:tt) => {
        mod $name {
            use super::*;

            #[divan::bench(args = LENS)]
            fn simdly(bencher: Bencher, len: usize) {
                let (a, b) = inputs(len);
                bencher
                    .counter(divan::counter::ItemsCount::new(len))
                    .bench_local(|| black_box(&a).$name(black_box(&b)));
            }

            #[divan::bench(args = LENS)]
            fn scalar(bencher: Bencher, len: usize) {
                let (a, b) = inputs(len);
                bencher
                    .counter(divan::counter::ItemsCount::new(len))
                    .bench_local(|| {
                        black_box(&a)
                            .iter()
                            .zip(black_box(&b))
                            .map(|(x, y)| x $op y)
                            .collect::<Vec<f32>>()
                    });
            }

            #[divan::bench(args = LENS)]
            fn simdly_in_place(bencher: Bencher, len: usize) {
                let (mut a, b) = inputs_in_place(len);
                bencher
                    .counter(divan::counter::ItemsCount::new(len))
                    .bench_local(|| black_box(&mut a).$assign(black_box(&b)));
            }

            #[divan::bench(args = LENS)]
            fn scalar_in_place(bencher: Bencher, len: usize) {
                let (mut a, b) = inputs_in_place(len);
                bencher
                    .counter(divan::counter::ItemsCount::new(len))
                    .bench_local(|| {
                        for (x, y) in black_box(&mut a).iter_mut().zip(black_box(&b)) {
                            *x $op_assign *y;
                        }
                    });
            }
        }
    };
}

bench_op!(add, add_assign, +, +=);
bench_op!(sub, sub_assign, -, -=);
bench_op!(mul, mul_assign, *, *=);
bench_op!(div, div_assign, /, /=);
