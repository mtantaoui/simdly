use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use rand::{rngs::ThreadRng, Rng};

use simdly::simd::{
    avx2::{
        f32x8,
        matmul::{
            matmul,
            // par_matmul
        },
    },
    utils::alloc_zeroed_f32_vec,
};
// use simdly::simd::{
//     avx512::{
//         f32x16,
//         matmul::{matmul, par_matmul},
//     },
//     utils::alloc_zeroed_f32_vec,
// };
// use simdly::avx2::matmul::{matmul, par_matmul}; // Added SeedableRng for reproducible tests // Use your crate name

// Helper to generate a column-major matrix as Vec<f32>
// Data is filled column by column.
fn gen_col_major_vec(rows: usize, cols: usize, rng: &mut ThreadRng) -> Vec<f32> {
    let mut data = vec![0.0f32; rows * cols];
    for j in 0..cols {
        // Iterate over columns (outer loop for memory contiguity)
        for i in 0..rows {
            // Iterate over rows
            data[j * rows + i] = rng.random_range(-1.0_f32..1.0_f32);
        }
    }
    data
}

// Function to verify results (optional, for debugging)
#[allow(dead_code, clippy::too_many_arguments)]
fn verify_results(
    m: usize,
    n: usize,
    _a_vec: &[f32],
    _b_vec: &[f32],
    c_custom_vec: &[f32],
    a_nd: &Array2<f32>,
    b_nd: &Array2<f32>,
    test_name: &str,
) {
    let c_expected_nd = a_nd.dot(b_nd);
    let mut mismatches = 0;
    const MAX_MISMATCHES_TO_PRINT: usize = 5;

    for r in 0..m {
        for c_idx in 0..n {
            let custom_val = c_custom_vec[c_idx * m + r]; // Column-major access
            let expected_val = c_expected_nd[[r, c_idx]]; // ndarray standard access (row, col)

            if (custom_val - expected_val).abs() > 1e-3 {
                // Adjust tolerance as needed
                if mismatches < MAX_MISMATCHES_TO_PRINT {
                    eprintln!(
                        "[{}] Mismatch at C[{}, {}]: custom = {:.6}, expected = {:.6}, diff = {:.6}",
                        test_name, r, c_idx, custom_val, expected_val, (custom_val - expected_val).abs()
                    );
                }
                mismatches += 1;
            }
        }
    }
    if mismatches > 0 {
        eprintln!("[{test_name}] Total mismatches: {mismatches}");
        // panic!("[{}] Verification failed with {} mismatches.", test_name, mismatches);
    } else {
        // println!("[{}] Verification successful.", test_name);
    }
}

fn benchmark_gemm(c: &mut Criterion) {
    // Use a seeded RNG for reproducible benchmark data, if desired
    // let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let mut rng = rand::rng(); // For non-deterministic data

    let mut group = c.benchmark_group("GEMM_Implementations");

    // Define matrix sizes to test (M, K, N)
    // Using square matrices for simplicity here.
    // Ensure these sizes are somewhat compatible with your block sizes (MC, NC, KC) for optimal performance.
    for &size in [64, 128, 256, 512, 1024].iter() {
        // for &size in &[64, 128].iter() { // Smaller set for quicker local tests
        let m = size;
        let k_dim = size; // Renamed to avoid conflict with `k` param in matmul functions
        let n = size;

        // --- Prepare data for your functions (column-major Vec<f32>) ---
        let a_vec = gen_col_major_vec(m, k_dim, &mut rng);
        let b_vec = gen_col_major_vec(k_dim, n, &mut rng);

        // --- Prepare data for ndarray (column-major Array2<f32>) ---
        // Create ndarray matrices also in column-major (Fortran) order.
        // This ensures ndarray interprets the Vec data (which we made column-major) correctly.
        let a_nd =
            Array2::from_shape_vec((k_dim, m), a_vec.clone()).expect("Failed to create ndarray A");
        let a_nd = a_nd.t().to_owned();

        let b_nd =
            Array2::from_shape_vec((n, k_dim), b_vec.clone()).expect("Failed to create ndarray B");
        let b_nd = b_nd.t().to_owned();

        let bench_id_str = format!("{m}x{k_dim}x{n}");

        // Clone data inside iter to ensure fresh state if modified, or for cache effects
        let mut c_vec_custom = alloc_zeroed_f32_vec(m * n, f32x8::AVX_ALIGNMENT);
        // vec![0.0f32; m * n];
        let a_clone = a_vec.clone();
        let b_clone = b_vec.clone();

        // Benchmark your sequential matmul
        group.bench_with_input(
            BenchmarkId::new("custom_matmul", &bench_id_str),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    // If c_vec_custom is not reset, results accumulate.
                    // For a fair benchmark, C should be zeroed or fresh for each call.
                    // However, matmul is C += A*B, so it should be initialized.
                    // If your kernel does C = A*B (overwrite), then no init needed.
                    // Based on your kernel (FMA), it's accumulating. So, init to 0.
                    unsafe {
                        matmul(
                            black_box(&a_clone),
                            black_box(&b_clone),
                            black_box(&mut c_vec_custom),
                            black_box(m),
                            black_box(n),
                            black_box(k_dim),
                        )
                    };
                });
                // Optional: run verification once after benchmarking this size (not timed)
                // verify_results(m, n, &a_vec, &b_vec, &c_vec_custom, &a_nd, &b_nd, &format!("custom_matmul_{}", bench_id_str));
            },
        );

        // // Benchmark your parallel matmul
        // group.bench_with_input(
        //     BenchmarkId::new("custom_par_matmul", &bench_id_str),
        //     &size,
        //     |bencher, _| {
        //         let mut c_vec_par_custom = alloc_zeroed_f32_vec(m * n, f32x8::AVX_ALIGNMENT);
        //         let a_clone = a_vec.clone();
        //         let b_clone = b_vec.clone();
        //         bencher.iter(|| {
        //             c_vec_par_custom.iter_mut().for_each(|x| *x = 0.0); // Reset C
        //             unsafe {
        //                 par_matmul(
        //                     black_box(&a_clone),
        //                     black_box(&b_clone),
        //                     black_box(&mut c_vec_par_custom),
        //                     black_box(m),
        //                     black_box(n),
        //                     black_box(k_dim),
        //                 )
        //             };
        //         });
        //         // Optional: run verification once
        //         // verify_results(m, n, &a_vec, &b_vec, &c_vec_par_custom, &a_nd, &b_nd, &format!("custom_par_matmul_{}", bench_id_str));
        //     },
        // );

        // Benchmark ndarray
        group.bench_with_input(
            BenchmarkId::new("ndarray_dot", &bench_id_str),
            &size,
            |bencher, _| {
                // ndarray's dot product allocates the result, so no manual reset of C is needed for it.
                // We clone a_nd and b_nd to ensure fair comparison if memory access patterns matter across iterations
                // let a_nd_clone = a_nd.clone();
                // let b_nd_clone = b_nd.clone();
                bencher.iter(|| {
                    let _c_nd = black_box(a_nd.dot(&b_nd));
                });
            },
        );

        // Perform verification for the smallest size to ensure correctness
        // This is done outside the bencher.iter loops.
        // if size == 64 {
        //     let mut c_val_custom = vec![0.0f32; m * n];
        //     unsafe { matmul(&a_vec, &b_vec, &mut c_val_custom, m, n, k_dim) };
        //     verify_results(
        //         m,
        //         n,
        //         &a_vec,
        //         &b_vec,
        //         &c_val_custom,
        //         &a_nd,
        //         &b_nd,
        //         &format!("VALIDATION_custom_matmul_{bench_id_str}"),
        //     );

        //     let mut c_val_par_custom = vec![0.0f32; m * n];
        //     unsafe { par_matmul(&a_vec, &b_vec, &mut c_val_par_custom, m, n, k_dim) };
        //     verify_results(
        //         m,
        //         n,
        //         &a_vec,
        //         &b_vec,
        //         &c_val_par_custom,
        //         &a_nd,
        //         &b_nd,
        //         &format!("VALIDATION_custom_par_matmul_{bench_id_str}"),
        //     );
        // }
    }
    group.finish();
}

criterion_group!(benches, benchmark_gemm);
criterion_main!(benches);
