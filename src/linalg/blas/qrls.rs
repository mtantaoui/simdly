use crate::linalg::blas::{qrdc::dqrdc, qrlss::dqrlss, qrsl::mat_idx};

/// Custom error type for `dqrls`.
#[derive(Debug, PartialEq, Eq)]
pub enum DqrlsError {
    InvalidLda,                   // LDA < M when M > 0
    InvalidNCols,                 // N <= 0 (in C, so N=0 for Rust usize)
    InvalidItask,                 // ITASK not 1 or 2
    FactorizationSingular(usize), // From dqrsl if R is singular (info from dqrsl)
    // NOTE: This error can only be emitted if the Rust `dqrlss` function
    // is capable of returning an error indicating singularity.
    // The `dqrlss` function signature used here returns `()`.
    RankEstimationFailed, // If rank estimation process itself fails.
                          // NOTE: This error is not currently emitted by the simplified rank
                          // estimation logic in this function. It would require `dqrdc` to
                          // return an error or more complex rank estimation logic with failure modes (e.g. invalid tol).
}

// Implement std::fmt::Display for DqrlsError for better error messages
impl std::fmt::Display for DqrlsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DqrlsError::InvalidLda => write!(
                f,
                "LDA must be greater than or equal to M (number of rows) when M > 0."
            ),
            DqrlsError::InvalidNCols => write!(f, "N (number of columns) must be positive."),
            DqrlsError::InvalidItask => write!(f, "ITASK must be 1 or 2."),
            DqrlsError::FactorizationSingular(idx) => {
                write!(f, "Matrix R is singular at column {idx} during solve stage. (Potentially from dqrsl via dqrlss if dqrlss could report it)")
            }
            DqrlsError::RankEstimationFailed => {
                write!(f, "Rank estimation during QR factorization failed (e.g., due to an issue in dqrdc or invalid tolerance).")
            }
        }
    }
}
// Implement std::error::Error for DqrlsError
impl std::error::Error for DqrlsError {}

/// Factors a linear system `A*X = B` and solves it in a least squares sense.
///
/// The system may be overdetermined, underdetermined, or singular. The solution
/// is obtained using a QR factorization of the coefficient matrix `A`.
/// This function can be used efficiently for multiple right-hand sides `B`
/// with the same matrix `A` by using the `itask` parameter.
///
/// This function closely mirrors the LINPACK `DQRLS` routine. Parallelism is primarily
/// achieved if the underlying factorization (`dqrdc`) and solve
/// (`dqrlss`) routines (and their BLAS L1 dependencies) are parallelized.
///
/// # Arguments
///
/// * `a`: Matrix `A` (M rows, N columns). Modified in place if `itask == 1`. Column-major.
/// * `lda`: Leading dimension of `a`. Must be `>= M` if `M > 0`.
/// * `m_rows`: Number of rows of `A`.
/// * `n_cols`: Number of columns of `A`.
/// * `tol`: Relative tolerance for rank determination during factorization (if `itask == 1`).
/// * `kr`:
///     - If `itask == 1` (factorization): Output parameter for the determined numerical rank of `A`.
///     - If `itask == 2` (solve only): Input parameter, the rank of `A` from a previous call.
/// * `b_rhs`: The right-hand side M-vector `B`.
/// * `x_sol`: Output N-vector for the least squares solution `X`.
/// * `rsd`: Output M-vector for the residual `B - A*X`.
/// * `jpvt`: Workspace/Output for pivot information (N-vector).
///     - If `itask == 1`: Used by factorization (`dqrdc`), and stores final permutation.
///     - If `itask == 2`: Input, the permutation from the factorization step.
/// * `qraux`: Workspace/Output for auxiliary QR factorization data (N-vector).
///     - If `itask == 1`: Used by factorization (`dqrdc`), stores reflector data.
///     - If `itask == 2`: Input, the reflector data from the factorization step.
/// * `itask`:
///     - `1`: Factor matrix `A` (calling `dqrdc` and performing rank estimation) and solve.
///       `kr` is an output. `jpvt` and `qraux` are outputs from factorization.
///     - `2`: Assume `A`, `qraux`, `jpvt` contain valid factorization data from a previous
///       call with `itask == 1`. `kr` is an input. Only solves the system.
///
/// # Returns
///
/// `Ok(())` on success.
/// `Err(DqrlsError)` on failure, indicating the type of error.
#[allow(clippy::too_many_arguments)]
pub fn dqrls(
    a: &mut [f64],
    lda: usize,
    m_rows: usize,
    n_cols: usize,
    tol: f64,
    kr: &mut usize,
    b_rhs: &[f64],
    x_sol: &mut [f64],
    rsd: &mut [f64],
    jpvt: &mut [isize],
    qraux: &mut [f64],
    itask: i32,
) -> Result<(), DqrlsError> {
    // --- Initial Parameter Validation (mimicking C error codes/conditions) ---
    if m_rows > 0 && lda < m_rows {
        return Err(DqrlsError::InvalidLda);
    }
    // C code: if ( n <= 0 ). For usize n_cols, this means n_cols == 0.
    if n_cols == 0 {
        return Err(DqrlsError::InvalidNCols);
    }
    // C code: if ( itask < 1 ). Common practice is to also check upper bound.
    if !(1..=2).contains(&itask) {
        return Err(DqrlsError::InvalidItask);
    }

    // --- Handle M=0 case (0 rows) ---
    // If m_rows is 0, A is 0xN. Rank is 0. X is N zeros. B is empty. RSD is empty.
    if m_rows == 0 {
        *kr = 0;
        // n_cols > 0 is guaranteed by the check above.
        // x_sol must have length n_cols. This check must come before use.
        if x_sol.len() < n_cols {
            panic!(
                "dqrls: x_sol.len() insufficient for M=0 case. Expected len >= n_cols ({}), got {}.",
                n_cols,
                x_sol.len()
            );
        }
        x_sol[..n_cols].fill(0.0);
        // rsd has length m_rows (0), so nothing to fill.
        // b_rhs has length m_rows (0).
        // jpvt, qraux are not used or modified if m_rows = 0 (dqrdc handles M=0).
        return Ok(());
    }

    // --- Slice Length Validations (Rust specific) ---
    // At this point: m_rows > 0 and n_cols > 0.
    // Check 'a'
    // The last accessible index in 'a' is mat_idx(m_rows - 1, n_cols - 1, lda).
    // So, a.len() must be at least mat_idx(m_rows - 1, n_cols - 1, lda) + 1.
    let required_a_len = mat_idx(m_rows - 1, n_cols - 1, lda) + 1;
    if a.len() < required_a_len {
        panic!(
            "dqrls: a.len() insufficient. For m_rows={}, n_cols={}, lda={}, expected len >= {}, got {}.",
            m_rows, n_cols, lda, required_a_len, a.len()
        );
    }
    // Check 'qraux' (DQRDC documentation typically specifies QRAUX(N))
    if qraux.len() < n_cols {
        panic!(
            "dqrls: qraux.len() insufficient. Expected len >= n_cols ({}), got {}.",
            n_cols,
            qraux.len()
        );
    }
    // Check 'jpvt' (DQRDC documentation typically specifies JPVT(N))
    if jpvt.len() < n_cols {
        panic!(
            "dqrls: jpvt.len() insufficient. Expected len >= n_cols ({}), got {}.",
            n_cols,
            jpvt.len()
        );
    }
    // Check 'b_rhs'
    if b_rhs.len() < m_rows {
        panic!(
            "dqrls: b_rhs.len() insufficient. Expected len >= m_rows ({}), got {}.",
            m_rows,
            b_rhs.len()
        );
    }
    // Check 'x_sol'
    if x_sol.len() < n_cols {
        panic!(
            "dqrls: x_sol.len() insufficient. Expected len >= n_cols ({}), got {}.",
            n_cols,
            x_sol.len()
        );
    }
    // Check 'rsd'
    if rsd.len() < m_rows {
        panic!(
            "dqrls: rsd.len() insufficient. Expected len >= m_rows ({}), got {}.",
            m_rows,
            rsd.len()
        );
    }

    // --- Factor the matrix if itask == 1 ---
    if itask == 1 {
        // Workspace for dqrdc (WORK array for JOB=1 for pivoting)
        // n_cols > 0 is guaranteed here.
        let mut work_dqrdc = vec![0.0; n_cols];
        let job_pivoting_dqrdc = 1; // Enable column pivoting in dqrdc

        // Call dqrdc for QR factorization with pivoting.
        // Assuming dqrdc is infallible or panics on its own internal errors.
        // If dqrdc could return an error, it might be mapped to DqrlsError::RankEstimationFailed.
        dqrdc(
            a,
            lda,
            m_rows,
            n_cols,
            qraux, // Output: Householder reflector data
            jpvt,  // Output: Permutation vector (0-based indices)
            &mut work_dqrdc,
            job_pivoting_dqrdc,
        );

        // Rank (kr) determination based on diagonal elements of R (in 'a') and 'tol'.
        // R is stored in the upper triangle of the permuted matrix 'a'.
        // m_rows > 0 and n_cols > 0 is true at this point.
        let mut estimated_rank = 0;
        // R(0,0) of the permuted matrix A*P. This is a[mat_idx(0,0,lda)] after dqrdc.
        let r00_abs = a[mat_idx(0, 0, lda)].abs();

        // A negative or NaN 'tol' could lead to unexpected behavior.
        // The C reference doesn't specify behavior for invalid 'tol'.
        // We assume 'tol' is a sensible (small, positive) value.
        // If tol is NaN, comparisons will be false, rank likely 0.
        // If tol is negative, abs(diag) > negative_val might make everything "significant".
        if tol < 0.0 {
            // This could be an error condition, e.g., DqrlsError::RankEstimationFailed
            // or a new DqrlsError::InvalidTolerance.
            // For now, mirroring C's lack of explicit check on tol's sign.
            // However, typical usage implies tol >= 0.
        }

        // Only proceed with rank counting if R(0,0) is significantly non-zero.
        // This avoids issues if r00_abs is extremely small, making tol * r00_abs also tiny or zero.
        if r00_abs > f64::EPSILON {
            // Check if R(0,0) is numerically distinguishable from zero.
            for k_idx in 0..n_cols.min(m_rows) {
                // Diagonal element R(k,k) is a[mat_idx(k_idx, k_idx, lda)]
                if a[mat_idx(k_idx, k_idx, lda)].abs() > tol * r00_abs {
                    estimated_rank = k_idx + 1;
                } else {
                    // Subsequent diagonal elements are considered too small relative to R(0,0).
                    break;
                }
            }
        }
        // If r00_abs is close to zero (or if tol causes all diags to fail the test), estimated_rank remains 0.
        *kr = estimated_rank;

        // DqrlsError::RankEstimationFailed is not triggered by this specific logic.
        // It would be relevant if `dqrdc` itself failed and returned an error,
        // or if `tol` was invalid (e.g., NaN) and this was explicitly checked to fail.
    }
    // If itask == 2, `*kr` (the value passed in) is used as input.
    // The caller is responsible for ensuring `a`, `qraux`, `jpvt`, and `*kr`
    // are valid from a previous `itask == 1` call.

    // --- Solve the least-squares problem using the factorization ---
    // The `dqrlss` function is assumed to be the one from `crate::linalg::blas::qrlss::dqrlss`.
    // In typical direct translations of LINPACK/BLAS routines, `dqrlss` would return `()`.
    // If the Rust version of `dqrlss` were enhanced to return a `Result`
    // (e.g., `Result<(), QrlssError>`) to signal singularity detected by its internal `dqrsl` call,
    // then that error could be mapped to `DqrlsError::FactorizationSingular`.
    // Example:
    // match dqrlss(...) {
    //     Ok(()) => Ok(()),
    //     Err(qrlss_error) => match qrlss_error {
    //         linalg::blas::qrlss::QrlssError::SingularR(idx) => Err(DqrlsError::FactorizationSingular(idx)),
    //         // other errors from qrlss
    //     }
    // }
    // However, using the signature implied by `use crate::linalg::blas::qrlss::dqrlss;` (likely `fn(...) -> ()`),
    // such error propagation is not possible without changing `dqrlss`.
    dqrlss(
        a, lda, m_rows, n_cols, *kr, // Use the determined (itask=1) or input (itask=2) rank
        b_rhs, x_sol, rsd, jpvt, qraux,
    );
    // As the assumed `dqrlss` signature returns `()`, it cannot directly signal
    // `DqrlsError::FactorizationSingular`. This error variant in `DqrlsError`
    // suggests an ideal scenario where `dqrlss` *could* report such conditions.

    Ok(())
}
