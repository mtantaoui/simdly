// Assuming qr_solve is in a module like my_project_name::linalg::qr_solver
// Adjust the path according to your project structure.
// If qr_solve is in src/linalg/qr_solver.rs and lib.rs has `pub mod linalg;`
// and src/linalg/mod.rs has `pub mod qr_solver;`
use simdly::linalg::blas::solver::qr_solve; // Replace `your_project_name`

// You might also need DqrlsError if you plan to inspect it specifically,
// but QrSolveError (which wraps DqrlsError) is returned by qr_solve.
// use your_project_name::linalg::blas::qrls::DqrlsError;

// Helper function to compare float vectors
fn compare_vecs(v1: &[f64], v2: &[f64], tol: f64) -> bool {
    if v1.len() != v2.len() {
        return false;
    }
    for (a, b) in v1.iter().zip(v2.iter()) {
        if (a - b).abs() > tol {
            return false;
        }
    }
    true
}

// Helper to print a matrix (assuming column major)
fn print_matrix_col_major(m: usize, n: usize, a: &[f64], name: &str) {
    println!("{name} ({m}x{n}, column-major):");
    if m == 0 || n == 0 {
        println!("  []");
        return;
    }
    let lda = m; // Assuming packed storage where lda = m_rows
    for i in 0..m {
        print!("  [");
        for j in 0..n {
            let val = a.get(i + j * lda).copied().unwrap_or(f64::NAN);
            print!("{val:.4}");
            if j < n - 1 {
                print!(", ");
            }
        }
        println!("]");
    }
}

fn print_vector(v: &[f64], name: &str) {
    println!("{name} ({}):", v.len());
    print!("  [");
    for (i, val) in v.iter().enumerate() {
        print!("{val:.4}");
        if i < v.len() - 1 {
            print!(", ");
        }
    }
    println!("]");
}

fn run_test_case(
    m_rows: usize,
    n_cols: usize,
    a_in: &[f64],
    b_rhs: &[f64],
    expected_x_sol: Option<&[f64]>, // Option allows testing error cases where no solution is expected
    test_name: &str,
) {
    println!("\n--- Running Test Case: {test_name} ---");
    print_matrix_col_major(m_rows, n_cols, a_in, "Matrix A");
    print_vector(b_rhs, "Vector b");
    if let Some(expected) = expected_x_sol {
        print_vector(expected, "Expected x");
    }

    match qr_solve(m_rows, n_cols, a_in, b_rhs) {
        Ok(x_sol) => {
            print_vector(&x_sol, "Computed x");
            if let Some(expected) = expected_x_sol {
                let tolerance = 1e-9; // Adjust tolerance as needed
                if compare_vecs(&x_sol, expected, tolerance) {
                    println!(
                        "SUCCESS: Computed solution matches expected solution within tolerance {tolerance}."
                    );
                } else {
                    println!("FAILURE: Computed solution does NOT match expected solution.");
                    println!("  Expected: {expected:?}");
                    println!("  Computed: {x_sol:?}");
                }
            } else {
                println!("INFO: qr_solve returned Ok, but no expected solution was provided for comparison.");
            }
        }
        Err(e) => {
            eprintln!("FAILURE: qr_solve returned an error: {e}");
            if expected_x_sol.is_some() {
                eprintln!("  An Ok result was expected for this test case.");
            } else {
                println!("INFO: qr_solve returned an error, and no Ok result was expected (this might be a correct error test).");
            }
        }
    }
}

fn main() {
    println!("Starting QR Solve Tests...");

    // --- Test Case 1: Square, Well-conditioned System ---
    // A = [[1, 1],
    //      [1, 2]]
    // b = [[3],
    //      [5]]
    // Expected x = [[1], [2]]
    let m1 = 2;
    let n1 = 2;
    let a1 = vec![
        1.0, 1.0, // Col 0
        1.0, 2.0, // Col 1
    ];
    let b1 = vec![3.0, 5.0];
    let expected_x1 = vec![1.0, 2.0];
    run_test_case(
        m1,
        n1,
        &a1,
        &b1,
        Some(&expected_x1),
        "Square, Well-conditioned",
    );

    // --- Test Case 2: Overdetermined System (M > N) ---
    // A = [[1, 0],
    //      [0, 1],
    //      [1, 1]]
    // b = [[1],
    //      [2],
    //      [4]]
    // Solved using Python numpy.linalg.lstsq: x approx [1.3333, 2.3333]
    // A^T A = [[2, 1], [1, 2]]
    // A^T b = [[5], [6]]
    // [[2, 1], [1, 2]] [x1, x2]^T = [[5], [6]]^T
    // 2x1 + x2 = 5
    // x1 + 2x2 = 6  => x1 = 6 - 2x2
    // 2(6 - 2x2) + x2 = 5
    // 12 - 4x2 + x2 = 5
    // 12 - 3x2 = 5
    // 3x2 = 7 => x2 = 7/3
    // x1 = 6 - 2(7/3) = 6 - 14/3 = (18-14)/3 = 4/3
    // Expected x = [4/3, 7/3] = [1.333..., 2.333...]
    let m2 = 3;
    let n2 = 2;
    let a2 = vec![
        1.0, 0.0, 1.0, // Col 0
        0.0, 1.0, 1.0, // Col 1
    ];
    let b2 = vec![1.0, 2.0, 4.0];
    let expected_x2 = vec![4.0 / 3.0, 7.0 / 3.0];
    run_test_case(
        m2,
        n2,
        &a2,
        &b2,
        Some(&expected_x2),
        "Overdetermined (M > N)",
    );

    // --- Test Case 3: Underdetermined System (M < N) ---
    // A = [[1, 1, 1]]
    // b = [[3]]
    // Minimum norm solution x = [1, 1, 1]
    let m3 = 1;
    let n3 = 3;
    let a3 = vec![
        1.0, // Col 0
        1.0, // Col 1
        1.0, // Col 2
    ];
    let b3 = vec![3.0];
    let expected_x3 = vec![1.0, 1.0, 1.0];
    run_test_case(
        m3,
        n3,
        &a3,
        &b3,
        Some(&expected_x3),
        "Underdetermined (M < N)",
    );

    // --- Test Case 4: N_cols = 0 ---
    // Should return Ok(empty_vec)
    let m4 = 2;
    let n4 = 0;
    let a4 = vec![]; // A is M x 0
    let b4 = vec![1.0, 2.0];
    let expected_x4 = vec![];
    run_test_case(m4, n4, &a4, &b4, Some(&expected_x4), "N_cols = 0");

    // --- Test Case 5: M_rows = 0, N_cols > 0 ---
    // A is 0xN. x should be N zeros. kr should be 0.
    // (Assuming dqrls handles this by returning zeros for x)
    let m5 = 0;
    let n5 = 2;
    let a5 = vec![]; // A is 0 x 2
    let b5 = vec![]; // b is 0-vector
    let expected_x5 = vec![0.0, 0.0];
    run_test_case(
        m5,
        n5,
        &a5,
        &b5,
        Some(&expected_x5),
        "M_rows = 0, N_cols > 0",
    );

    // --- Test Case 6: Error case - Invalid N_COLS in dqrls (N=0 for dqrls) ---
    // This is now handled by qr_solve itself before calling dqrls.
    // To test DqrlsError::InvalidNCols directly, one would call dqrls.
    // qr_solve handles N=0 by returning Ok(vec![]).

    // --- Test Case 7: Error case - Empty 'a_in' when m_rows > 0, n_cols > 0 ---
    let m7 = 1;
    let n7 = 1;
    let a7 = vec![];
    let b7 = vec![1.0];
    run_test_case(m7, n7, &a7, &b7, None, "Error: Empty 'a_in' with M,N > 0");

    // --- Test Case 8: Matrix with a zero column (might affect rank estimation) ---
    // A = [[1, 0],
    //      [1, 0]]
    // b = [[2],
    //      [2]]
    // Rank should be 1. x2 can be anything. For min norm, x2=0. x1=2.
    // So, x = [2, 0]
    // The current dqrls stub for rank estimation is very simple.
    // A proper DQRLS with rank determination is needed for this.
    // The simplified rank estimation in the provided dqrls stub might find rank 1 or 2
    // depending on 'tol' and R(1,1). If R(1,1) is exactly zero due to the zero column,
    // rank might be 1.
    let m8 = 2;
    let n8 = 2;
    let a8 = vec![
        1.0, 1.0, // Col 0
        0.0, 0.0, // Col 1
    ];
    let b8 = vec![2.0, 2.0];
    let expected_x8 = vec![2.0, 0.0]; // This is the min norm solution
    run_test_case(
        m8,
        n8,
        &a8,
        &b8,
        Some(&expected_x8),
        "Rank Deficient (Zero Column)",
    );

    println!("\nQR Solve Tests Finished.");
}
