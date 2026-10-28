//! `SliceExt` comparisons against the plain scalar loop, from L1-resident to
//! larger-than-cache slices.
//!
//! ```sh
//! cargo bench --bench comparison
//! ```

use divan::{Bencher, black_box};
use simdly::SliceExt;

fn main() {
    // The first call detects CPU features; keep that one-off cost out of the
    // first benchmark's calibration.
    simdly::backend();
    divan::main();
}

const LENS: &[usize] = &[64, 4_096, 262_144, 4_194_304];

/// Values on both sides of `b`, so the results are not all equal.
fn inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..len).map(|i| (i % 17) as f32).collect();
    let b = (0..len).map(|i| (i % 13) as f32).collect();
    (a, b)
}

macro_rules! bench_cmp {
    ($name:ident, $scalar_name:ident, $op:tt) => {
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
                            .collect::<Vec<bool>>()
                    });
            }

            #[divan::bench(args = LENS)]
            fn simdly_scalar_rhs(bencher: Bencher, len: usize) {
                let (a, _) = inputs(len);
                bencher
                    .counter(divan::counter::ItemsCount::new(len))
                    .bench_local(|| black_box(&a).$scalar_name(black_box(8.0)));
            }

            #[divan::bench(args = LENS)]
            fn scalar_scalar_rhs(bencher: Bencher, len: usize) {
                let (a, _) = inputs(len);
                bencher
                    .counter(divan::counter::ItemsCount::new(len))
                    .bench_local(|| {
                        let s = black_box(8.0_f32);
                        black_box(&a).iter().map(|x| *x $op s).collect::<Vec<bool>>()
                    });
            }
        }
    };
}

bench_cmp!(less, less_scalar, <);
bench_cmp!(equal, equal_scalar, ==);
