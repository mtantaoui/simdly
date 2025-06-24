use ndarray::Array2;
use rand::{rngs::ThreadRng, Rng};
// use simdly::simd::avx2::matmul::{matmul, par_matmul};
use simdly::simd::{
    avx512::{
        f32x16,
        matmul::{matmul, par_matmul},
    },
    utils::alloc_zeroed_f32_vec,
};
// use simdly::matmul::{matmul, par_matmul}; // Added SeedableRng for reproducible tests // Use your crate name

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

fn main() {
    // Use a seeded RNG for reproducible benchmark data, if desired
    // let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let mut rng = rand::rng(); // For non-deterministic data

    // Define matrix sizes to test (M, K, N)
    // Using square matrices for simplicity here.
    // Ensure these sizes are somewhat compatible with your block sizes (MC, NC, KC) for optimal performance.
    for &size in [
        4,
        //  64, 128, 256, 512, 1024
    ]
    .iter()
    {
        // for &size in &[64, 128].iter() { // Smaller set for quicker local tests
        let m = size;
        let k_dim = size; // Renamed to avoid conflict with `k` param in matmul functions
        let n = size;

        // --- Prepare data for your functions (column-major Vec<f32>) ---
        let a_vec = gen_col_major_vec(m, k_dim, &mut rng);
        let b_vec = gen_col_major_vec(k_dim, n, &mut rng);

        let mut c_vec = alloc_zeroed_f32_vec(m * n, f32x16::AVX512_ALIGNMENT);
        let mut c_vec = alloc_zeroed_f32_vec(m * n, f32x16::AVX512_ALIGNMENT);

        unsafe { matmul(&a_vec, &b_vec, &mut c_vec, m, n, k_dim) };
        println!("{:?}", c_vec);
    }
}
