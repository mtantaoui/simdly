use crate::linalg::blas::{axpy::daxpy, dot::ddot};

#[inline(always)]
pub fn mat_idx(row: usize, col: usize, lda: usize) -> usize {
    row + col * lda // Column-major
}

/// Computes transformations, projections, and least squares solutions
/// using the QR factorization from a prior call to a `dqrdc`-like routine.
///
/// This function mirrors the functionality of the LINPACK `DQRSL` routine.
///
/// # Arguments
/// (Arguments match C, with slices instead of pointers)
/// * `a`: Matrix `A` after QR decomposition (output of `dqrdc`). Modified temporarily.
/// * `lda`: Leading dimension of `a`.
/// * `n`: Number of rows of `A`.
/// * `k`: Number of columns of `A` used in factorization (rank).
/// * `qraux`: Auxiliary reflector information from `dqrdc`.
/// * `y`: Input vector.
/// * `qy`: Output for `Q * y`.
/// * `qty`: Output for `Q' * y`.
/// * `b`: Output for least squares solution `B`.
/// * `rsd`: Output for residual `y - A*B`.
/// * `ab`: Output for approximation `A*B`.
/// * `job`: Integer specifying computations (see LINPACK docs or C comments).
///
/// # Returns
/// `info`: `0` for success. If `b` computation requested and `R` is singular,
/// `info` is the 1-based index of the first zero diagonal of `R`.
///
/// # Panics
/// Panics if input slice dimensions are inconsistent or insufficient for the
/// requested operations based on `n`, `k`, `lda`, and `job`.
#[allow(
    clippy::too_many_arguments,
    clippy::identity_op,
    clippy::cognitive_complexity
)]
pub fn dqrsl(
    a: &mut [f64],
    lda: usize,
    n: usize,
    k: usize,
    qraux: &[f64],
    y: &[f64],
    qy: &mut [f64],
    qty: &mut [f64],
    b: &mut [f64],
    rsd: &mut [f64],
    ab: &mut [f64],
    job: i32,
) -> usize {
    // --- Argument Validation ---
    if k > 0 && n > 0 && lda < n {
        // LDA must be at least N if there are rows and columns to process from A
        panic!("dqrsl: lda ({lda}) must be >= n ({n}) when k > 0 and n > 0.");
    }

    if k > 0 {
        // Minimum length for 'a' to access a[n-1, k-1] (0-indexed)
        // is mat_idx(n-1, k-1, lda) + 1.
        // If n=0, no elements of 'a' are accessed through mat_idx in main loops.
        // The ju=0 case for n=0 correctly handles this.
        // If n>0, k>0: min_a_len = (n-1) + (k-1)*lda + 1 = n + (k-1)*lda.
        let min_a_len = if n == 0 { 0 } else { n + (k - 1) * lda };
        if a.len() < min_a_len {
            panic!(
                "dqrsl: a.len() ({}) insufficient for lda ({}), n ({}), and k ({}). Needs at least {}.",
                a.len(), lda, n, k, min_a_len
            );
        }
        if qraux.len() < k {
            panic!("dqrsl: qraux.len() ({}) must be >= k ({}).", qraux.len(), k);
        }
    }

    if y.len() < n {
        // y must have at least n elements
        panic!("dqrsl: y.len() ({}) must be >= n ({}).", y.len(), n);
    }

    // Determine what is to be computed
    let job_qy = job / 10000 != 0;
    let job_cqty_eff = (job % 10000) != 0; // True if QTY, B, RSD, or AB needed
    let job_cb = (job % 1000) / 100 != 0;
    let job_cr = (job % 100) / 10 != 0;
    let job_cab = (job % 10) != 0;

    // Validate output slice lengths based on job flags
    if job_qy && qy.len() < n {
        panic!("dqrsl: qy.len() ({}) insufficient for n ({}).", qy.len(), n);
    }
    if job_cqty_eff && qty.len() < n {
        // If any of QTY, B, RSD, AB is needed, QTY (length n) is intermediate.
        panic!(
            "dqrsl: qty.len() ({}) insufficient for n ({}).",
            qty.len(),
            n
        );
    }
    if job_cb && b.len() < k {
        // b is solution, length k
        panic!("dqrsl: b.len() ({}) insufficient for k ({}).", b.len(), k);
    }
    if job_cr && rsd.len() < n {
        // rsd is residual, length n
        panic!(
            "dqrsl: rsd.len() ({}) insufficient for n ({}).",
            rsd.len(),
            n
        );
    }
    if job_cab && ab.len() < n {
        // ab is approximation A*B, length n
        panic!("dqrsl: ab.len() ({}) insufficient for n ({}).", ab.len(), n);
    }

    let mut info: usize = 0;

    // ju is the number of Householder transformations to apply.
    // k.min(n.saturating_sub(1)) handles n=0, n=1 cases correctly for ju.
    // If n=0 or n=1, n.saturating_sub(1) is 0, so ju=0.
    // Otherwise, ju = k.min(n-1).
    let ju = k.min(n.saturating_sub(1));

    // Special action when ju = 0.
    // This occurs if k=0 (no columns from A) or if n=0 or n=1.
    if ju == 0 {
        if n > 0 {
            // Only operate if there are elements (i.e., n=1 case if ju=0 and n>0)
            if job_qy {
                qy[..n].copy_from_slice(&y[..n]); // QY = Y (Q is I)
            }
            if job_cqty_eff {
                qty[..n].copy_from_slice(&y[..n]); // QTY = Y (Q' is I)
            }

            if job_cab {
                // Compute AB = AK*B
                if k == 0 {
                    // AK is N x 0 matrix, AK*B = 0 vector
                    ab[..n].fill(0.0);
                } else {
                    // implies n=1, k>=1. AK is a[0,0]. B = Y[0]/a[0,0]. AK*B = Y[0].
                    ab[0] = y[0];
                    // If n > 1 was possible here (it's not if ju=0, k>0), rest would be 0.
                }
            }
            if job_cb {
                // Compute B
                if k == 0 { // AK is N x 0. B is 0 x 1 (empty). Problem might be considered singular.
                     // b slice has length k=0, so no elements to set. info for singularity.
                     // No explicit info set here, relies on caller interpreting k=0 with job_cb.
                     // LINPACK DQRLS would set info via DQRSL. If DQRSL is called with k=0,
                     // b is not computed. Let's align by setting info if b is requested and k=0.
                     // However, original C code's special case for ju=0 does not set info if k=0.
                     // It only sets info if a[0,0] is zero (implying k>=1, n=1).
                     // For consistency with C behavior for ju=0:
                     // if k>0 (which implies n=1 here) and a[0,0] is singular, then info=1.
                     // If k=0, info remains 0 from this block.
                }
                // This 'else' covers k > 0. Since ju=0, this means n=1.
                if k > 0 {
                    // This is for the n=1, k>=1 case.
                    if a[mat_idx(0, 0, lda)].abs() < f64::EPSILON {
                        info = 1; // 1-based index for singular R(0,0)
                    } else if !b.is_empty() {
                        // Ensure b is writeable, b.len() must be >= k
                        b[0] = y[0] / a[mat_idx(0, 0, lda)];
                    }
                }
            }
            if job_cr {
                // Compute RSD = Y - AK*B
                if k == 0 {
                    // AK*B = 0. RSD = Y.
                    rsd[..n].copy_from_slice(&y[..n]);
                } else {
                    // implies n=1, k>=1. AK*B=Y[0]. RSD = Y[0]-Y[0] = 0.
                    rsd[0] = 0.0;
                    // If n > 1 was possible, rsd[1..n] would be y[1..n] as Q diagonal from 1..
                }
            }
        }
        // If n=0, all ..n slices are empty, no work done. info=0.
        return info;
    }

    // --- Main computation for ju > 0 ---

    if job_qy {
        qy[..n].copy_from_slice(&y[..n]);
    }
    if job_cqty_eff {
        qty[..n].copy_from_slice(&y[..n]);
    }

    // Compute QY: Apply Q (from right to left using reflectors: H_ju ... H_1)
    if job_qy {
        for j_idx in (0..ju).rev() {
            // j_idx from ju-1 down to 0
            if qraux[j_idx].abs() > f64::EPSILON {
                let temp_diag_a = a[mat_idx(j_idx, j_idx, lda)];
                a[mat_idx(j_idx, j_idx, lda)] = qraux[j_idx]; // Store v_0 related term

                let segment_len = n - j_idx;
                let a_col_segment_start = mat_idx(j_idx, j_idx, lda);
                let h_vec_slice = &a[a_col_segment_start..a_col_segment_start + segment_len];
                let qy_target_slice = &mut qy[j_idx..j_idx + segment_len];

                let t = -ddot(segment_len, h_vec_slice, 1, qy_target_slice, 1)
                    / a[mat_idx(j_idx, j_idx, lda)];
                daxpy(segment_len, t, h_vec_slice, 1, qy_target_slice, 1);

                a[mat_idx(j_idx, j_idx, lda)] = temp_diag_a; // Restore R(j,j)
            }
        }
    }

    // Compute Q'*Y (QTY): Apply Q' (from left to right: H_1 ... H_ju)
    if job_cqty_eff {
        for j_idx in 0..ju {
            // j_idx from 0 up to ju-1
            if qraux[j_idx].abs() > f64::EPSILON {
                let temp_diag_a = a[mat_idx(j_idx, j_idx, lda)];
                a[mat_idx(j_idx, j_idx, lda)] = qraux[j_idx];

                let segment_len = n - j_idx;
                let a_col_segment_start = mat_idx(j_idx, j_idx, lda);
                let h_vec_slice = &a[a_col_segment_start..a_col_segment_start + segment_len];
                let qty_target_slice = &mut qty[j_idx..j_idx + segment_len];

                let t = -ddot(segment_len, h_vec_slice, 1, qty_target_slice, 1)
                    / a[mat_idx(j_idx, j_idx, lda)];
                daxpy(segment_len, t, h_vec_slice, 1, qty_target_slice, 1);

                a[mat_idx(j_idx, j_idx, lda)] = temp_diag_a;
            }
        }
    }

    if job_cb && k > 0 {
        // Ensure b is not empty and qty has k elements.
        b[0..k].copy_from_slice(&qty[0..k]);
    }

    if job_cab {
        if k > 0 {
            ab[0..k].copy_from_slice(&qty[0..k]);
        }
        if k < n {
            ab[k..n].fill(0.0);
        } else if k == 0 && n > 0 {
            // if k=0, ab is all zeros
            ab[..n].fill(0.0);
        }
    }
    if job_cr {
        rsd[0..k].fill(0.0);
        if k < n {
            rsd[k..n].copy_from_slice(&qty[k..n]);
        } else if k == 0 && n > 0 {
            // if k=0, rsd is qty (which is y)
            rsd[..n].copy_from_slice(&qty[..n]);
        }
    }

    // Compute B (least squares solution by back-substitution R*B = QTY_first_k)
    if job_cb {
        for j_idx in (0..k).rev() {
            // j_idx from k-1 down to 0
            if a[mat_idx(j_idx, j_idx, lda)].abs() < f64::EPSILON {
                info = j_idx + 1; // 1-based singular index
                return info;
            }
            b[j_idx] /= a[mat_idx(j_idx, j_idx, lda)];

            if j_idx > 0 {
                let t = -b[j_idx];
                let r_col_segment_start = mat_idx(0, j_idx, lda);
                let r_col_slice = &a[r_col_segment_start..r_col_segment_start + j_idx];
                daxpy(j_idx, t, r_col_slice, 1, &mut b[0..j_idx], 1);
            }
        }
    }

    if job_cr || job_cab {
        for j_idx in (0..ju).rev() {
            // j_idx from ju-1 down to 0
            if qraux[j_idx].abs() > f64::EPSILON {
                let temp_diag_a = a[mat_idx(j_idx, j_idx, lda)];
                a[mat_idx(j_idx, j_idx, lda)] = qraux[j_idx];

                let segment_len = n - j_idx;
                let a_col_segment_start = mat_idx(j_idx, j_idx, lda);
                let h_vec_slice = &a[a_col_segment_start..a_col_segment_start + segment_len];

                if job_cr {
                    let rsd_target_slice = &mut rsd[j_idx..j_idx + segment_len];
                    let t = -ddot(segment_len, h_vec_slice, 1, rsd_target_slice, 1)
                        / a[mat_idx(j_idx, j_idx, lda)];
                    daxpy(segment_len, t, h_vec_slice, 1, rsd_target_slice, 1);
                }
                if job_cab {
                    let ab_target_slice = &mut ab[j_idx..j_idx + segment_len];
                    let t = -ddot(segment_len, h_vec_slice, 1, ab_target_slice, 1)
                        / a[mat_idx(j_idx, j_idx, lda)];
                    daxpy(segment_len, t, h_vec_slice, 1, ab_target_slice, 1);
                }
                a[mat_idx(j_idx, j_idx, lda)] = temp_diag_a;
            }
        }
    }

    info
}
