use rand::seq::SliceRandom; // For .shuffle()
use rand::Rng; // Trait for random number generators like thread_rng()

// Define the number type (f64 for double, f32 for single)
type Number = f64;

// --- Naive BLAS-like function implementations ---
// (gemm_naive and axpy_naive remain the same as in your correct version)
fn gemm_naive(
    _transa: char,
    _transb: char,
    m: usize,
    n: usize,
    k: usize,
    alpha: Number,
    a: &[Number],
    lda: usize,
    b: &[Number],
    ldb: usize,
    beta: Number,
    c: &mut [Number],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if beta == 0.0 {
            for val in c.iter_mut().take(m * n) {
                *val = 0.0;
            }
        }
        // More efficient zeroing if ldc=m
        else if beta != 1.0 {
            for j in 0..n {
                for i in 0..m {
                    c[i + j * ldc] *= beta;
                }
            }
        }
        return;
    }
    if beta == 0.0 {
        for j in 0..n {
            for i in 0..m {
                c[i + j * ldc] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for j in 0..n {
            for i in 0..m {
                c[i + j * ldc] *= beta;
            }
        }
    }
    if alpha == 0.0 {
        return;
    }
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i + p * lda] * b[p + j * ldb];
            }
            c[i + j * ldc] += alpha * sum;
        }
    }
}

fn axpy_naive(
    n_val: usize,
    alpha: Number,
    x: &[Number],
    incx: usize,
    y: &mut [Number],
    incy: usize,
) {
    if n_val == 0 || alpha == 0.0 {
        return;
    }
    if incx == 1 && incy == 1 {
        for i in 0..n_val {
            y[i] += alpha * x[i];
        }
    } else {
        let mut ix = 0;
        let mut iy = 0;
        for _ in 0..n_val {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
    }
}

// --- Helper function to print a matrix ---
fn print_matrix(name: &str, mat: &[Number], rows: usize, cols: usize) {
    println!("{} ({} x {}):", name, rows, cols);
    if rows * cols == 0 {
        println!("  (empty matrix)\n");
        return;
    }
    for i in 0..rows {
        for j in 0..cols {
            print!("{:10.4} ", mat[i + j * rows]);
        }
        println!();
    }
    println!();
}

// --- AlgoParams struct ---
#[derive(Debug)]
struct AlgoParams {
    y_factors: Vec<Vec<Number>>,
    s_levels: Vec<Vec<i32>>,
    p_levels: Vec<Vec<i32>>,
    epsilon_val: Number,
    max_rec_depth: usize,
    n_block_dim: usize,
    r_rank: usize,
    base_case_size_threshold: usize,
}

// --- Core Recursive Matrix Multiplication Function ---
fn compute_recursive_product(
    c_sub_out: &mut [Number],
    a_sub_in: &[Number],
    b_sub_in: &[Number],
    n_sub: usize,
    rec_level: usize,
    params: &AlgoParams,
) {
    let trans_gemm = 'N';
    let alpha_gemm = 1.0;
    let beta_gemm = 0.0;

    if rec_level > params.max_rec_depth || n_sub <= params.base_case_size_threshold {
        if n_sub == 0 {
            return;
        }
        gemm_naive(
            trans_gemm, trans_gemm, n_sub, n_sub, n_sub, alpha_gemm, a_sub_in, n_sub, b_sub_in,
            n_sub, beta_gemm, c_sub_out, n_sub,
        );
        return;
    }

    if n_sub % params.n_block_dim != 0 {
        eprintln!(
            "Error: n_sub ({}) not divisible by n_block_dim ({}) at rec_level {}. Fallback.",
            n_sub, params.n_block_dim, rec_level
        );
        gemm_naive(
            trans_gemm, trans_gemm, n_sub, n_sub, n_sub, alpha_gemm, a_sub_in, n_sub, b_sub_in,
            n_sub, beta_gemm, c_sub_out, n_sub,
        );
        return;
    }

    let block_sz = n_sub / params.n_block_dim;
    let block_numel = block_sz * block_sz;
    let unit_stride = 1;

    // Correctly borrow slices from params
    let s_current_level: &[i32] = &params.s_levels[rec_level - 1];
    let p_current_level: &[i32] = &params.p_levels[rec_level - 1];
    let y0: &[Number] = &params.y_factors[0];
    let y1: &[Number] = &params.y_factors[1];
    let y2: &[Number] = &params.y_factors[2];

    for j_block_idx in 0..params.n_block_dim {
        for i_block_idx in 0..params.n_block_dim {
            let mut c_block_sum = vec![0.0 as Number; block_numel];
            for r_term in 0..params.r_rank {
                let p_val_y2_dim1 = p_current_level[0 + 3 * i_block_idx];
                let p_val_y2_dim2 = p_current_level[2 + 3 * j_block_idx];
                if p_val_y2_dim1 < 1
                    || p_val_y2_dim1 as usize > params.n_block_dim
                    || p_val_y2_dim2 < 1
                    || p_val_y2_dim2 as usize > params.n_block_dim
                {
                    panic!(
                        "P index for Y2 out of bounds [{}, {}]",
                        p_val_y2_dim1, p_val_y2_dim2
                    );
                }
                let y2_offset = (p_val_y2_dim1 - 1) as usize
                    + (p_val_y2_dim2 - 1) as usize * params.n_block_dim
                    + r_term * params.n_block_dim * params.n_block_dim;
                let y2_coeff = y2[y2_offset];
                if y2_coeff == 0.0 {
                    continue;
                }

                let mut a_term_block = vec![0.0 as Number; block_numel];
                let mut b_term_block = vec![0.0 as Number; block_numel];
                for k_sum_idx in 0..params.n_block_dim {
                    for l_sum_idx in 0..params.n_block_dim {
                        let p_val_y0_dim1 = p_current_level[0 + 3 * k_sum_idx];
                        let p_val_y0_dim2 = p_current_level[1 + 3 * l_sum_idx];
                        if p_val_y0_dim1 < 1
                            || p_val_y0_dim1 as usize > params.n_block_dim
                            || p_val_y0_dim2 < 1
                            || p_val_y0_dim2 as usize > params.n_block_dim
                        {
                            panic!(
                                "P index for Y0 out of bounds [{}, {}]",
                                p_val_y0_dim1, p_val_y0_dim2
                            );
                        }
                        let y0_offset = (p_val_y0_dim1 - 1) as usize
                            + (p_val_y0_dim2 - 1) as usize * params.n_block_dim
                            + r_term * params.n_block_dim * params.n_block_dim;
                        let y0_coeff = y0[y0_offset];
                        if y0_coeff != 0.0 {
                            let s_factor_a = (s_current_level[0 + 3 * k_sum_idx]
                                * s_current_level[1 + 3 * l_sum_idx])
                                as Number;
                            let effective_coeff_a = y0_coeff * s_factor_a;
                            for c_blk in 0..block_sz {
                                for r_blk in 0..block_sz {
                                    let src_idx_a = (r_blk + k_sum_idx * block_sz)
                                        + (c_blk + l_sum_idx * block_sz) * n_sub;
                                    let dest_idx_a = r_blk + c_blk * block_sz;
                                    a_term_block[dest_idx_a] +=
                                        effective_coeff_a * a_sub_in[src_idx_a];
                                }
                            }
                        }
                        let p_val_y1_dim1 = p_current_level[1 + 3 * k_sum_idx];
                        let p_val_y1_dim2 = p_current_level[2 + 3 * l_sum_idx];
                        if p_val_y1_dim1 < 1
                            || p_val_y1_dim1 as usize > params.n_block_dim
                            || p_val_y1_dim2 < 1
                            || p_val_y1_dim2 as usize > params.n_block_dim
                        {
                            panic!(
                                "P index for Y1 out of bounds [{}, {}]",
                                p_val_y1_dim1, p_val_y1_dim2
                            );
                        }
                        let y1_offset = (p_val_y1_dim1 - 1) as usize
                            + (p_val_y1_dim2 - 1) as usize * params.n_block_dim
                            + r_term * params.n_block_dim * params.n_block_dim;
                        let y1_coeff = y1[y1_offset];
                        if y1_coeff != 0.0 {
                            let s_factor_b = (s_current_level[1 + 3 * k_sum_idx]
                                * s_current_level[2 + 3 * l_sum_idx])
                                as Number;
                            let effective_coeff_b = y1_coeff * s_factor_b;
                            for c_blk in 0..block_sz {
                                for r_blk in 0..block_sz {
                                    let src_idx_b = (r_blk + k_sum_idx * block_sz)
                                        + (c_blk + l_sum_idx * block_sz) * n_sub;
                                    let dest_idx_b = r_blk + c_blk * block_sz;
                                    b_term_block[dest_idx_b] +=
                                        effective_coeff_b * b_sub_in[src_idx_b];
                                }
                            }
                        }
                    }
                }
                let mut c_block_temp_prod = vec![0.0 as Number; block_numel];
                compute_recursive_product(
                    &mut c_block_temp_prod,
                    &a_term_block,
                    &b_term_block,
                    block_sz,
                    rec_level + 1,
                    params,
                );
                axpy_naive(
                    block_numel,
                    y2_coeff,
                    &c_block_temp_prod,
                    unit_stride,
                    &mut c_block_sum,
                    unit_stride,
                );
            }
            let s_output_factor = (s_current_level[0 + 3 * i_block_idx]
                * s_current_level[2 + 3 * j_block_idx]) as Number;
            let final_scale_factor = s_output_factor / (1.0 + params.epsilon_val);
            for c_blk in 0..block_sz {
                for r_blk in 0..block_sz {
                    let src_idx_c_block = r_blk + c_blk * block_sz;
                    let dest_idx_c_sub =
                        (r_blk + i_block_idx * block_sz) + (c_blk + j_block_idx * block_sz) * n_sub;
                    c_sub_out[dest_idx_c_sub] = final_scale_factor * c_block_sum[src_idx_c_block];
                }
            }
        }
    }
}

// --- Interface Function ---
fn rand_mat_mult_rust_interface(
    c_out: &mut Vec<Number>,
    a_in: &Vec<Number>,
    b_in: &Vec<Number>,
    n_dim: usize,
    y_factors_data: Vec<Vec<Number>>,
    s_levels_data: Vec<Vec<i32>>,
    p_levels_data: Vec<Vec<i32>>,
    epsilon_val: Number,
    max_rec_depth_val: usize,
    n_block_dim_val: usize,
    r_rank_val: usize,
    base_case_size_val: usize,
) {
    let params = AlgoParams {
        y_factors: y_factors_data,
        s_levels: s_levels_data,
        p_levels: p_levels_data,
        epsilon_val,
        max_rec_depth: max_rec_depth_val,
        n_block_dim: n_block_dim_val,
        r_rank: r_rank_val,
        base_case_size_threshold: base_case_size_val,
    };
    for elem in c_out.iter_mut() {
        *elem = 0.0;
    }
    compute_recursive_product(c_out, a_in, b_in, n_dim, 1, &params); // Pass params
}

// --- Helper function to generate Strassen's Y matrices ---
fn generate_strassen_y_factors(
    y_factors_collection: &mut Vec<Vec<Number>>, // Expects 3 Vecs (can be empty)
    n_block_dim_val: usize,
    r_rank_val: usize,
) {
    if n_block_dim_val != 2 || r_rank_val != 7 {
        eprintln!(
            "generate_strassen_y_factors: n_block_dim must be 2 and r_rank must be 7 for Strassen."
        );
        let y_factor_elements = n_block_dim_val * n_block_dim_val * r_rank_val;
        for k_idx in 0..3 {
            y_factors_collection[k_idx] = vec![0.0 as Number; y_factor_elements];
        }
        return;
    }
    let y_factor_elements = n_block_dim_val * n_block_dim_val * r_rank_val;
    for k_idx in 0..3 {
        y_factors_collection[k_idx] = vec![0.0 as Number; y_factor_elements];
    }

    // Use split_at_mut to get distinct mutable borrows
    let (first_two, last_one) = y_factors_collection.split_at_mut(2);
    let (first_one, second_one) = first_two.split_at_mut(1);
    let u = &mut first_one[0];
    let v = &mut second_one[0];
    let w = &mut last_one[0];

    u[0 + 0 * 2 + 0 * 4] = 1.0;
    u[1 + 1 * 2 + 0 * 4] = 1.0;
    v[0 + 0 * 2 + 0 * 4] = 1.0;
    v[1 + 1 * 2 + 0 * 4] = 1.0;
    w[0 + 0 * 2 + 0 * 4] = 1.0;
    w[1 + 1 * 2 + 0 * 4] = 1.0;
    u[1 + 0 * 2 + 1 * 4] = 1.0;
    u[1 + 1 * 2 + 1 * 4] = 1.0;
    v[0 + 0 * 2 + 1 * 4] = 1.0;
    w[1 + 0 * 2 + 1 * 4] = 1.0;
    w[1 + 1 * 2 + 1 * 4] = -1.0;
    u[0 + 0 * 2 + 2 * 4] = 1.0;
    v[0 + 1 * 2 + 2 * 4] = 1.0;
    v[1 + 1 * 2 + 2 * 4] = -1.0;
    w[0 + 1 * 2 + 2 * 4] = 1.0;
    w[1 + 1 * 2 + 2 * 4] = 1.0;
    u[1 + 1 * 2 + 3 * 4] = 1.0;
    v[1 + 0 * 2 + 3 * 4] = 1.0;
    v[0 + 0 * 2 + 3 * 4] = -1.0;
    w[0 + 0 * 2 + 3 * 4] = 1.0;
    w[1 + 0 * 2 + 3 * 4] = 1.0;
    u[0 + 0 * 2 + 4 * 4] = 1.0;
    u[0 + 1 * 2 + 4 * 4] = 1.0;
    v[1 + 1 * 2 + 4 * 4] = 1.0;
    w[0 + 0 * 2 + 4 * 4] = -1.0;
    w[0 + 1 * 2 + 4 * 4] = 1.0;
    u[1 + 0 * 2 + 5 * 4] = 1.0;
    u[0 + 0 * 2 + 5 * 4] = -1.0;
    v[0 + 0 * 2 + 5 * 4] = 1.0;
    v[0 + 1 * 2 + 5 * 4] = 1.0;
    w[1 + 1 * 2 + 5 * 4] = 1.0;
    u[0 + 1 * 2 + 6 * 4] = 1.0;
    u[1 + 1 * 2 + 6 * 4] = -1.0;
    v[1 + 0 * 2 + 6 * 4] = 1.0;
    v[1 + 1 * 2 + 6 * 4] = 1.0;
    w[0 + 0 * 2 + 6 * 4] = 1.0;
}

// --- Helper function to generate a random permutation ---
fn generate_random_permutation_0_based(arr: &mut [i32], n: usize, rng: &mut impl Rng) {
    if n == 0 {
        return;
    }
    for i in 0..n {
        arr[i] = i as i32;
    }
    arr.shuffle(rng);
}

/// Performs APPROXIMATE matrix multiplication C = A * B for square matrices
/// using a randomized Strassen-based algorithm.
///
/// Assumes A, B, and the returned C are stored in column-major order.
///
/// # Arguments
/// * `a_vec`: A flat vector representing matrix A (column-major).
/// * `b_vec`: A flat vector representing matrix B (column-major).
///
/// # Returns
/// * `Ok(Vec<Number>)`: A flat vector representing the approximate matrix C (column-major).
/// * `Err(&str)`: An error message if matrices are not square or dimensions don't match.
///
/// # Panics
/// * If the length of `a_vec` or `b_vec` is not a perfect square.
pub fn matmul_approximate(a_vec: &[Number], b_vec: &[Number]) -> Result<Vec<Number>, &'static str> {
    let len_a = a_vec.len();
    let len_b = b_vec.len();

    if len_a != len_b {
        return Err("Input matrices must have the same number of elements for square matrix multiplication.");
    }

    let n_float = (len_a as Number).sqrt();
    if n_float.fract() != 0.0 {
        return Err("Input vector length must be a perfect square to form a square matrix.");
    }
    let n_overall = n_float as usize;

    if n_overall * n_overall != len_a {
        return Err("Internal error: Dimension calculation failed for square matrix.");
    }

    if n_overall == 0 {
        // Handle empty matrices
        return Ok(Vec::new());
    }

    // --- Default Parameters for the Randomized Algorithm ---
    let n_block_dim = 2; // Strassen uses n=2
    let r_rank = 7; // Strassen uses R=7

    // Calculate max_rec_depth based on N and n_block_dim for full Strassen application
    // log_n_block_dim(N_overall)
    let mut max_rec_depth = 0;
    if n_overall > 0 && n_block_dim > 1 {
        max_rec_depth = (n_overall as f64).log(n_block_dim as f64).floor() as usize;
        if max_rec_depth == 0 && n_overall > 1 {
            // e.g. N=3, n_block_dim=2 -> log2(3) ~ 1.58 -> floor is 1
            max_rec_depth = 1; // Ensure at least one level if N > base_case
        }
    }
    // Or, set a fixed reasonable depth:
    // let max_rec_depth = 3; // Example: apply up to 3 levels of recursion

    let base_case_threshold = 16; // Switch to GEMM if block size <= 16 (tune this)
                                  // For Strassen, often powers of 2 are good thresholds.
                                  // Or 1 if you want to recurse all the way down.

    let use_random_sp = true; // Always use random for this "approximate" interface
    let epsilon = 0.001 as Number; // Default epsilon for approximation

    let mut rng = rand::thread_rng();

    // --- Setup Y_factors (Strassen specific for now) ---
    let mut y_factors: Vec<Vec<Number>> = vec![Vec::new(), Vec::new(), Vec::new()];
    generate_strassen_y_factors(&mut y_factors, n_block_dim, r_rank);

    // --- Setup S_levels and P_levels ---
    let mut s_all_levels: Vec<Vec<i32>> = Vec::with_capacity(max_rec_depth);
    let mut p_all_levels: Vec<Vec<i32>> = Vec::with_capacity(max_rec_depth);
    let elements_per_sp_level_array = 3 * n_block_dim;

    for _level_idx in 0..max_rec_depth {
        let mut s_level_data = vec![0i32; elements_per_sp_level_array];
        let mut p_level_data = vec![0i32; elements_per_sp_level_array];

        if use_random_sp {
            for i in 0..elements_per_sp_level_array {
                s_level_data[i] = if rng.gen::<bool>() { 1 } else { -1 };
            }
            let mut p_temp_perm_row = vec![0i32; n_block_dim];
            for dim_type in 0..3 {
                generate_random_permutation_0_based(&mut p_temp_perm_row, n_block_dim, &mut rng);
                for block_idx in 0..n_block_dim {
                    p_level_data[dim_type + 3 * block_idx] = p_temp_perm_row[block_idx] + 1;
                }
            }
        } else {
            // Should not be reached if use_random_sp is always true for this func
            for i in 0..elements_per_sp_level_array {
                s_level_data[i] = 1;
            }
            for dim_type in 0..3 {
                for block_idx in 0..n_block_dim {
                    p_level_data[dim_type + 3 * block_idx] = (block_idx + 1) as i32;
                }
            }
        }
        s_all_levels.push(s_level_data);
        p_all_levels.push(p_level_data);
    }

    // --- Prepare for the call ---
    // Make copies of a_vec and b_vec because rand_mat_mult_rust_internal_interface expects Vecs
    // (though it only reads from them via slices in its current form)
    // If a_vec and b_vec are already Vecs passed by value, this is not needed.
    // If they are slices, we need to convert them to Vecs if the internal function expects owned Vecs.
    // For now, assuming rand_mat_mult_rust_internal_interface can take slices for A and B.
    // Let's adjust rand_mat_mult_rust_internal_interface to take slices for A & B input.

    let a_input_vec = a_vec.to_vec(); // If rand_mat_mult_rust_internal_interface needs owned Vecs
    let b_input_vec = b_vec.to_vec(); // If rand_mat_mult_rust_internal_interface needs owned Vecs
    let mut c_out: Vec<Number> = vec![0.0 as Number; len_a];

    rand_mat_mult_rust_interface(
        &mut c_out,
        &a_input_vec,
        &b_input_vec,
        n_overall,
        y_factors,
        s_all_levels,
        p_all_levels,
        epsilon,
        max_rec_depth,
        n_block_dim,
        r_rank,
        base_case_threshold,
    );

    Ok(c_out)
}

fn main() {
    let mut rng = rand::rng();

    let n_overall = 8;
    let n_block_dim = 2;
    let r_rank = 7;
    let max_rec_depth = 2;
    let base_case_threshold = 1;
    let use_random_sp = true;
    let mut epsilon = 0.001 as Number;

    if !use_random_sp {
        epsilon = 0.0;
        println!("INFO: DETERMINISTIC S/P, epsilon=0.");
    } else {
        println!("INFO: RANDOM S/P, epsilon={:.3e}.", epsilon);
    }

    let num_elements = n_overall * n_overall;
    let mut a: Vec<Number> = vec![0.0; num_elements];
    let mut b: Vec<Number> = vec![0.0; num_elements];
    let mut c_approx: Vec<Number> = vec![0.0; num_elements];
    let mut c_exact: Vec<Number> = vec![0.0; num_elements];

    println!(
        "Initializing matrices A and B (size {}x{})...",
        n_overall, n_overall
    );
    for i in 0..num_elements {
        a[i] = rng.random_range(-10.0..10.0);
        b[i] = rng.random_range(-10.0..10.0);
    }

    let mut y_factors: Vec<Vec<Number>> = vec![Vec::new(), Vec::new(), Vec::new()]; // Initialize with 3 empty Vecs
    println!("Generating Y factors (Strassen's U,V,W)...");
    generate_strassen_y_factors(&mut y_factors, n_block_dim, r_rank);

    let mut s_all_levels: Vec<Vec<i32>> = Vec::with_capacity(max_rec_depth);
    let mut p_all_levels: Vec<Vec<i32>> = Vec::with_capacity(max_rec_depth);
    let elements_per_sp_level_array = 3 * n_block_dim;

    println!("Generating S & P for {} levels...", max_rec_depth);
    for _level_idx in 0..max_rec_depth {
        let mut s_level_data = vec![0i32; elements_per_sp_level_array];
        let mut p_level_data = vec![0i32; elements_per_sp_level_array];
        if use_random_sp {
            for i in 0..elements_per_sp_level_array {
                s_level_data[i] = if rng.gen::<bool>() { 1 } else { -1 };
            }
            let mut p_temp_perm_row = vec![0i32; n_block_dim];
            for dim_type in 0..3 {
                generate_random_permutation_0_based(&mut p_temp_perm_row, n_block_dim, &mut rng);
                for block_idx in 0..n_block_dim {
                    p_level_data[dim_type + 3 * block_idx] = p_temp_perm_row[block_idx] + 1;
                }
            }
        } else {
            for i in 0..elements_per_sp_level_array {
                s_level_data[i] = 1;
            }
            for dim_type in 0..3 {
                for block_idx in 0..n_block_dim {
                    p_level_data[dim_type + 3 * block_idx] = (block_idx + 1) as i32;
                }
            }
        }
        s_all_levels.push(s_level_data);
        p_all_levels.push(p_level_data);
    }

    println!("Running matrix multiplication...");
    rand_mat_mult_rust_interface(
        &mut c_approx,
        &a,
        &b,
        n_overall,
        y_factors,
        s_all_levels,
        p_all_levels, // These are moved here
        epsilon,
        max_rec_depth,
        n_block_dim,
        r_rank,
        base_case_threshold,
    );

    if n_overall <= 8 {
        print_matrix(
            "C_approx (Algorithm result)",
            &c_approx,
            n_overall,
            n_overall,
        );
    }
    println!("Running exact GEMM_naive for comparison...");
    gemm_naive(
        'N',
        'N',
        n_overall,
        n_overall,
        n_overall,
        1.0,
        &a,
        n_overall,
        &b,
        n_overall,
        0.0,
        &mut c_exact,
        n_overall,
    );
    if n_overall <= 8 {
        print_matrix("C_exact (GEMM_naive)", &c_exact, n_overall, n_overall);
    }

    let mut frobenius_norm_diff_sq = 0.0;
    let mut frobenius_norm_exact_sq = 0.0;
    let mut max_abs_diff = 0.0;
    for i in 0..num_elements {
        let diff = c_exact[i] - c_approx[i];
        frobenius_norm_diff_sq += diff * diff;
        frobenius_norm_exact_sq += c_exact[i] * c_exact[i];
        if diff.abs() > max_abs_diff {
            max_abs_diff = diff.abs();
        }
    }
    let frobenius_norm_diff = frobenius_norm_diff_sq.sqrt();
    let frobenius_norm_exact = if frobenius_norm_exact_sq > 1e-18 {
        frobenius_norm_exact_sq.sqrt()
    } else {
        0.0
    };
    let relative_error = if frobenius_norm_exact > 1e-9 {
        frobenius_norm_diff / frobenius_norm_exact
    } else {
        frobenius_norm_diff
    };

    println!("\n--- Error Metrics ---");
    println!(
        "Frobenius norm of (C_exact - C_approx): {:.6e}",
        frobenius_norm_diff
    );
    println!(
        "Frobenius norm of (C_exact):            {:.6e}",
        frobenius_norm_exact
    );
    println!(
        "Relative Frobenius error:               {:.6e}",
        relative_error
    );
    println!(
        "Maximum absolute difference:            {:.6e}",
        max_abs_diff
    );
    println!("\nDone.");
}
