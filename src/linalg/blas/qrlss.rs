use crate::linalg::blas::qrsl::{dqrsl, mat_idx}; // Assuming mat_idx is defined here as in the problem

/// Solves a linear system `A*X = B` in a least squares sense,
/// where `A` is an M-by-N matrix of rank `kr`.
///
/// This function must be preceded by a call to a `dqrdc`-like routine
/// which computes the QR factorization of `A`.
///
/// A solution `X` with at most `kr` non-zero components is found, minimizing
/// the 2-norm of the residual `(A*X - B)`.
///
/// # Arguments
///
/// * `a`: The QR factorization information from `dqrdc`. Contains `R` in its
///   upper triangle and Householder reflector information below the diagonal.
/// * `lda`: Leading dimension of `a`. Must be at least `m_rows`.
/// * `m_rows`: Number of rows of matrix `A`.
/// * `n_cols`: Number of columns of matrix `A`.
/// * `kr_rank`: Rank of matrix `A`, as determined by `dqrdc`.
/// * `b_rhs`: The right-hand side M-vector `B` of the linear system.
/// * `x_sol`: Output N-vector for the least squares solution `X`.
/// * `rsd`: Output M-vector for the residual `B - A*X`.
/// * `jpvt`: Pivot information from `dqrdc` (0-based indices).
///   `jpvt[k]` is the 0-based original column index at permuted position `k`.
/// * `qraux`: Auxiliary QR factorization information from `dqrdc`.
///
/// # Panics
/// Panics if slice dimensions are inconsistent or insufficient.
#[allow(clippy::too_many_arguments)]
pub fn dqrlss(
    a: &mut [f64],     // QR factorization of A (R in upper triangle, H-vectors below)
    lda: usize,        // Leading dimension of A
    m_rows: usize,     // Number of rows of A
    n_cols: usize,     // Number of columns of A
    kr_rank: usize,    // Rank of A
    b_rhs: &[f64],     // Right-hand side vector B (length m_rows)
    x_sol: &mut [f64], // Solution vector X (length n_cols)
    rsd: &mut [f64],   // Residual vector B - A*X (length m_rows)
    jpvt: &[isize],    // Pivot information (0-based) from dqrdc (length n_cols)
    qraux: &[f64],     // Auxiliary QR info from dqrdc (relevant length kr_rank)
) {
    // --- Argument Validation ---
    if m_rows > 0 && lda < m_rows {
        panic!("dqrlss: lda ({lda}) must be >= m_rows ({m_rows}) when m_rows > 0.");
    }

    if kr_rank > n_cols || kr_rank > m_rows {
        // Rank cannot exceed matrix dimensions
        panic!(
            "dqrlss: kr_rank ({kr_rank}) is invalid for m_rows ({m_rows}) and n_cols ({n_cols}). Must be <= m_rows and <= n_cols."
        );
    }

    // Check 'a' based on its usage by dqrsl (accessing R up to kr_rank columns and m_rows rows)
    if kr_rank > 0 && m_rows > 0 {
        // max index accessed in 'a' for R and H-vectors by dqrsl is typically related to mat_idx(m_rows-1, kr_rank-1, lda)
        if mat_idx(m_rows - 1, kr_rank.saturating_sub(1), lda) >= a.len() {
            // Use saturating_sub for kr_rank=0 case safety although guarded
            panic!(
                "dqrlss: a.len() ({}) insufficient for lda ({}), m_rows ({}), kr_rank ({}). Needs at least {}.",
                a.len(), lda, m_rows, kr_rank, mat_idx(m_rows - 1, kr_rank.saturating_sub(1), lda) + 1
            );
        }
    }

    if qraux.len() < kr_rank {
        // dqrsl uses qraux[..kr_rank]
        panic!(
            "dqrlss: qraux.len() ({}) must be >= kr_rank ({}).",
            qraux.len(),
            kr_rank
        );
    }
    if b_rhs.len() < m_rows {
        panic!(
            "dqrlss: b_rhs.len() ({}) must be >= m_rows ({}).",
            b_rhs.len(),
            m_rows
        );
    }
    if x_sol.len() < n_cols {
        panic!(
            "dqrlss: x_sol.len() ({}) must be >= n_cols ({}).",
            x_sol.len(),
            n_cols
        );
    }
    if rsd.len() < m_rows {
        panic!(
            "dqrlss: rsd.len() ({}) must be >= m_rows ({}).",
            rsd.len(),
            m_rows
        );
    }
    if jpvt.len() < n_cols {
        panic!(
            "dqrlss: jpvt.len() ({}) must be >= n_cols ({}).",
            jpvt.len(),
            n_cols
        );
    }

    // --- Handle kr_rank = 0 (or singular matrix from dqrsl info) ---
    // This combined handling matches original SLATEC DQRLSS.F behavior.
    if kr_rank == 0 {
        x_sol[..n_cols].fill(0.0);
        if m_rows > 0 {
            // Ensure slices are not empty if m_rows = 0
            rsd[..m_rows].copy_from_slice(&b_rhs[..m_rows]);
        }
        return;
    }

    // --- Prepare for dqrsl call ---
    // dqrsl needs various arrays. For job=110:
    // - y_in: b_rhs
    // - qy_out: not computed meaningfully for job=110 (job/1000 % 10 == 0). Dummy.
    // - qty_out: Q'*Y computed (job % 10 == 0).
    // - x_solution_k_out: solution to R*X_k = QTY_k computed (job/10 % 10 == 1).
    // - rsd_out: residual Y - A*X_k computed (job/100 % 10 == 1).
    // - ab_out: not computed meaningfully for job=110 (job/10000 % 10 == 0). Dummy.

    // LINPACK dqrsl qty argument is documented as length N (which is m_rows here)
    let mut temp_qty = vec![0.0; m_rows];
    // Dummy buffers for outputs not requested or overwritten. Length m_rows is safe.
    let mut temp_qy_dummy = vec![0.0; m_rows];
    let mut temp_ab_dummy = vec![0.0; m_rows];

    // Call dqrsl to compute QTY, solution for first kr_rank components, and residual
    // job = 110:
    //   Bit 0 (1s place, job % 10 == 0): Compute QTY (Q_transpose * Y)
    //   Bit 1 (10s place, job / 10 % 10 == 1): Compute X_k (solution for R*X_k = QTY_k)
    //   Bit 2 (100s place, job / 100 % 10 == 1): Compute RSD (Residual Y - A*X_k)
    //   Bit 3 (1000s place, job / 1000 % 10 == 0): Do not compute QY (Q*Y)
    //   Bit 4 (10000s place, job / 10000 % 10 == 0): Do not compute AB (A*X_k directly)
    let info = dqrsl(
        a,                     // QR factorization (input/output, modified by dqrsl if H-vectors are used)
        lda,                   // Leading dimension of a
        m_rows,                // Number of rows in original A
        kr_rank,               // Rank / number of columns of R to use
        &qraux[..kr_rank],     // Householder scalars for first kr_rank columns
        b_rhs,                 // Input Y (the RHS vector b)
        &mut temp_qy_dummy,    // Output QY (not computed for job=110)
        &mut temp_qty,         // Output QTY = Q_transpose * Y
        &mut x_sol[..kr_rank], // Output X_k (solution for first kr_rank permuted variables)
        rsd,                   // Output RSD (residual Y - A*X_k)
        &mut temp_ab_dummy,    // Output AB (A*X_k, not computed for job=110)
        110,                   // Job control integer
    );

    // --- Check for singularity indicated by dqrsl ---
    if info != 0 {
        // R is singular (R(info-1, info-1) was zero).
        // Set X to 0 and RSD to B, as per original SLATEC DQRLSS.
        // eprintln!("Warning: dqrsl returned info = {} (singular R); setting X=0, RSD=B.", info);
        x_sol[..n_cols].fill(0.0);
        if m_rows > 0 {
            // Ensure slices are not empty if m_rows = 0
            rsd[..m_rows].copy_from_slice(&b_rhs[..m_rows]);
        }
        return;
    }

    // --- Finalize solution vector X ---
    // Zero out components of X corresponding to dependent columns
    if kr_rank < n_cols {
        x_sol[kr_rank..n_cols].fill(0.0);
    }

    // Permute X back to original column order using jpvt
    // x_sol currently holds the solution for permuted columns.
    // jpvt[k] stores the 0-based original index of the column now at permuted position k.
    // We want final_x_sol[original_idx] = x_sol[k_permuted_idx].
    if n_cols > 0 {
        // Avoid allocation if n_cols is 0
        let mut temp_x_permuted_storage = vec![0.0; n_cols];
        for k_permuted_idx in 0..n_cols {
            let original_idx_isize = jpvt[k_permuted_idx];

            // Validate jpvt content (original index must be within bounds)
            if original_idx_isize < 0 || (original_idx_isize as usize) >= n_cols {
                panic!(
                    "dqrlss: jpvt[{k_permuted_idx}] contains invalid original index {original_idx_isize} (must be 0 <= idx < {n_cols})."
                );
            }
            let original_idx = original_idx_isize as usize;
            temp_x_permuted_storage[original_idx] = x_sol[k_permuted_idx];
        }
        x_sol[..n_cols].copy_from_slice(&temp_x_permuted_storage[..n_cols]);
    }
}
