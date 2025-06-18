use crate::linalg::blas::qrls::{dqrls, DqrlsError}; // Assuming this is your dqrls module

/// Computes the maximum absolute value in a matrix represented as a flat slice.
///
/// Assumes column-major layout.
///
/// * `m_rows`: Number of rows in the matrix.
/// * `n_cols`: Number of columns in the matrix.
/// * `a`: Slice containing the matrix data.
/// * `lda`: Leading dimension of `a` (stride between columns, typically `m_rows` for packed storage).
///
/// # Panics
/// This function might panic if `lda` is too small for `m_rows` or if `a` is too short
/// for `m_rows`, `n_cols`, and `lda`, leading to out-of-bounds access. However,
/// `qr_solve` provides correctly sized inputs.
fn mat_abs_max(m_rows: usize, n_cols: usize, a: &[f64], lda: usize) -> f64 {
    if m_rows == 0 || n_cols == 0 {
        return 0.0;
    }
    // If m_rows > 0 and n_cols > 0, but 'a' is empty, this indicates an upstream issue.
    // qr_solve ensures 'a' (via a_qr) is not empty if m_rows > 0.
    if a.is_empty() {
        // Should not happen if m_rows > 0 due to qr_solve checks
        return 0.0;
    }

    let mut max_abs_val = 0.0;
    for j in 0..n_cols {
        // For each column
        for i in 0..m_rows {
            // For each row in that column
            let idx = i + j * lda;
            // The following check is a safeguard. Given how qr_solve prepares
            // a_qr and lda, idx should always be within bounds.
            if idx < a.len() {
                let val_abs = a[idx].abs();
                if val_abs > max_abs_val {
                    // Using > is fine; f64::max handles NaNs differently.
                    max_abs_val = val_abs;
                }
            } else {
                // This indicates a mismatch between dimensions and slice length.
                // Should not occur with inputs from qr_solve.
                // For a general utility, this might warrant a panic or error.
                // Breaking here as further access in this column would be invalid.
                break;
            }
        }
    }
    max_abs_val
}

/// Custom error type for `qr_solve`.
#[derive(Debug)]
pub enum QrSolveError {
    DqrlsError(DqrlsError), // Propagate errors from dqrls
    InputMatrixEmpty,       // Input matrix 'a' is empty when M > 0 and N > 0
                            // InvalidDimensions is removed as specific cases are handled or passed to dqrls.
}

impl std::fmt::Display for QrSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QrSolveError::DqrlsError(e) => write!(f, "Error from dqrls: {e}"),
            QrSolveError::InputMatrixEmpty => {
                write!(
                    f,
                    "Input matrix 'a_in' is empty but m_rows and n_cols are positive."
                )
            }
        }
    }
}

impl std::error::Error for QrSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            QrSolveError::DqrlsError(e) => Some(e),
            _ => None,
        }
    }
}

// Convert DqrlsError to QrSolveError
impl From<DqrlsError> for QrSolveError {
    fn from(err: DqrlsError) -> Self {
        QrSolveError::DqrlsError(err)
    }
}

/// Solves a linear system `A*X = B` in the least squares sense using QR factorization.
///
/// This function is a high-level wrapper that handles memory allocation for workspaces,
/// copies the input matrix `A` (as it's modified by the factorization),
/// calculates a default tolerance, and calls `dqrls` to perform the factorization
/// and solve.
///
/// # Arguments
///
/// * `m_rows`: Number of rows of matrix `A`.
/// * `n_cols`: Number of columns of matrix `A`.
/// * `a_in`: Input M-by-N matrix `A`, as a flat slice (column-major order assumed).
/// * `b_rhs`: Input M-vector `B`, the right-hand side.
///
/// # Returns
///
/// `Ok(Vec<f64>)` containing the N-vector solution `X` on success.
/// `Err(QrSolveError)` on failure.
///
/// # Panics
/// Panics if `a_in` length is inconsistent with `m_rows * n_cols` (when `m_rows > 0`),
/// or if `b_rhs` length is less than `m_rows`.
/// The underlying `dqrls` might panic for other invalid inputs not caught here if its
/// own slice checks fail (though `qr_solve` aims to prepare valid slices).
pub fn qr_solve(
    m_rows: usize,
    n_cols: usize,
    a_in: &[f64],  // Assumed column-major M x N
    b_rhs: &[f64], // M-vector
) -> Result<Vec<f64>, QrSolveError> {
    // Handle N=0 case: solution vector is empty.
    // dqrls would error for n_cols = 0, so we handle this common case upfront.
    if n_cols == 0 {
        return Ok(Vec::new());
    }

    // At this point, n_cols > 0.

    // Validate a_in based on m_rows.
    if m_rows > 0 {
        // If m_rows > 0, a_in cannot be empty.
        if a_in.is_empty() {
            return Err(QrSolveError::InputMatrixEmpty);
        }
        // Check if a_in is large enough for a packed m_rows x n_cols matrix.
        // This assumes a_in is column-major and lda will be m_rows.
        if a_in.len() < m_rows * n_cols {
            // This is a precondition violation for packed data.
            panic!(
                "qr_solve: a_in.len() ({}) is less than m_rows*n_cols ({}*{}) when m_rows > 0.",
                a_in.len(),
                m_rows,
                n_cols
            );
        }
    }
    // If m_rows == 0:
    // a_in can be empty (it represents a 0xN matrix). The check `a_in.len() < m_rows * n_cols`
    // becomes `a_in.len() < 0`, which is false, so it doesn't panic. This is fine.

    // Validate b_rhs length against m_rows.
    if b_rhs.len() < m_rows {
        panic!(
            "qr_solve: b_rhs.len() ({}) is less than m_rows ({})",
            b_rhs.len(),
            m_rows
        );
    }

    // Copy 'a_in' as dqrls modifies it.
    // If m_rows is 0, a_in might be empty. `a_in.to_vec()` handles empty slices correctly.
    let mut a_qr = a_in.to_vec();

    // LDA for a column-major matrix is typically the number of rows.
    // If m_rows is 0, lda will be 0. The `dqrls` function should handle lda=0 if m_rows=0.
    let lda = m_rows;

    // Calculate tolerance.
    // mat_abs_max handles m_rows=0 or n_cols=0 correctly by returning 0.0.
    let a_max_abs = mat_abs_max(m_rows, n_cols, &a_qr, lda);

    let tol = if a_max_abs < f64::EPSILON {
        // a_max_abs is non-negative
        // If matrix A is effectively zero or for robustness against very small a_max_abs.
        f64::EPSILON // A small positive tolerance
    } else {
        // Standard tolerance calculation.
        // Ensure tol remains positive and non-zero, even if a_max_abs is huge.
        (f64::EPSILON / a_max_abs).max(f64::MIN_POSITIVE)
    };

    // Allocate workspace arrays.
    // x_sol length is n_cols (guaranteed > 0 at this point).
    let mut x_sol = vec![0.0; n_cols];
    // jpvt and qraux are N-vectors (n_cols).
    let mut jpvt = vec![0_isize; n_cols];
    let mut qraux = vec![0.0; n_cols];
    // rsd is an M-vector (m_rows). If m_rows is 0, rsd is empty.
    let mut rsd = vec![0.0; m_rows];

    let mut kr: usize = 0; // Rank, will be set by dqrls.
    let itask = 1; // Instructs dqrls to factor and solve.

    // Call the underlying dqrls routine.
    // The `dqrls` implementation is assumed to handle the m_rows = 0, n_cols > 0 case
    // (e.g., by setting x_sol to zeros and kr to 0), which aligns with typical
    // least squares behavior for such systems.
    dqrls(
        &mut a_qr, lda, m_rows, n_cols, tol, &mut kr, b_rhs, &mut x_sol, &mut rsd, &mut jpvt,
        &mut qraux, itask,
    )?; // Propagates DqrlsError, converted to QrSolveError via From trait.

    Ok(x_sol)
}
