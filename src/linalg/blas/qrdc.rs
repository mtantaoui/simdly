use crate::linalg::blas::{axpy::daxpy, dot::ddot, nrm2::dnrm2, scal::dscal};

// Assume swap_matrix_columns is provided as in the problem description
fn swap_matrix_columns(a: &mut [f64], lda: usize, n_rows: usize, col1_idx: usize, col2_idx: usize) {
    if col1_idx == col2_idx || n_rows == 0 {
        return;
    }
    for r_idx in 0..n_rows {
        // Assuming mat_idx is available or defined as below for column-major
        let idx1 = r_idx + col1_idx * lda; // mat_idx(r_idx, col1_idx, lda)
        let idx2 = r_idx + col2_idx * lda; // mat_idx(r_idx, col2_idx, lda)
        a.swap(idx1, idx2);
    }
}

/// Computes the QR factorization of a real N-by-P matrix A using Householder transformations.
/// A is assumed to be in column-major order.
#[allow(
    clippy::too_many_arguments,
    clippy::cognitive_complexity,
    clippy::identity_op,
    clippy::needless_range_loop,
    clippy::comparison_chain
)]
pub fn dqrdc(
    a: &mut [f64],      // Matrix data (column-major)
    lda: usize,         // Leading dimension of A (lda >= n, effectively stride between columns)
    n: usize,           // Number of rows of A
    p: usize,           // Number of columns of A
    qraux: &mut [f64],  // Output: Further info for Q (Householder scalars)
    jpvt: &mut [isize], // Input/Output: Pivoting info (0-based column indices)
    work: &mut [f64],   // Workspace (stores original norms)
    job_pivoting: i32,  // Control for pivoting (0 for no pivot, !=0 for pivot)
) {
    // Helper for column-major indexing
    #[inline]
    fn mat_idx(row: usize, col: usize, lda: usize) -> usize {
        row + col * lda
    }

    // --- Initial checks ---
    if n > 0 && lda < n {
        panic!("dqrdc: lda ({lda}) must be >= n ({n}).");
    }
    if p > 0 {
        if n > 0 && (mat_idx(n - 1, p - 1, lda) >= a.len()) {
            panic!(
                "dqrdc: a.len() ({}) insufficient for lda ({}), p ({}), n ({}). Last element index: {}",
                a.len(), lda, p, n, mat_idx(n - 1, p - 1, lda)
            );
        }
        if qraux.len() < p {
            panic!("dqrdc: qraux.len() ({}) must be >= p ({}).", qraux.len(), p);
        }
        if jpvt.len() < p {
            panic!("dqrdc: jpvt.len() ({}) must be >= p ({}).", jpvt.len(), p);
        }
        // Based on C code behavior, 'work' is used for initial norms of free columns
        // regardless of job_pivoting value if p > 0.
        if work.len() < p {
            panic!(
                "dqrdc: work.len() ({}) must be >= p ({}) as it's used for norms.",
                work.len(),
                p
            );
        }
    }

    if p == 0 || n == 0 {
        return;
    }

    let pivoting_enabled = job_pivoting != 0;
    let mut pl_idx: usize = 0; // 0-based index: first "free" column
    let mut pu_idx: usize = p; // 0-based index: one past the last "free" column

    // --- Pivoting setup ---
    if pivoting_enabled {
        let mut temp_jpvt_input_flags: Vec<i32> = vec![0; p];
        for i in 0..p {
            temp_jpvt_input_flags[i] = jpvt[i] as i32;
        }
        for i in 0..p {
            jpvt[i] = i as isize; // jpvt[k] stores original 0-based index of column now at pos k
        }

        // Move initial columns to the left
        for j_current_pos in 0..p {
            let original_col_idx = jpvt[j_current_pos] as usize;
            if temp_jpvt_input_flags[original_col_idx] > 0 {
                // If original column was 'initial'
                if j_current_pos != pl_idx {
                    swap_matrix_columns(a, lda, n, pl_idx, j_current_pos);
                    jpvt.swap(pl_idx, j_current_pos);
                }
                pl_idx += 1;
            }
        }
        // Move final columns to the right (from remaining columns)
        for j_current_pos in (pl_idx..p).rev() {
            let original_col_idx = jpvt[j_current_pos] as usize;
            if temp_jpvt_input_flags[original_col_idx] < 0 {
                // If original column was 'final'
                pu_idx -= 1;
                if j_current_pos != pu_idx {
                    swap_matrix_columns(a, lda, n, pu_idx, j_current_pos);
                    jpvt.swap(pu_idx, j_current_pos);
                }
            }
        }
    } else {
        jpvt.iter_mut()
            .enumerate()
            .for_each(|(i, val)| *val = i as isize);
        // pl_idx remains 0, pu_idx remains p (all columns are "free")
    }

    // --- Calculate initial column norms for "free" columns ---
    // Store in qraux (current norms) and work (original norms cache)
    for j_idx in pl_idx..pu_idx {
        let col_data_start_idx = mat_idx(0, j_idx, lda);
        let col_j_slice = &a[col_data_start_idx..col_data_start_idx + n];
        let norm_val = dnrm2(n, col_j_slice, 1);
        qraux[j_idx] = norm_val;
        work[j_idx] = norm_val;
    }

    let lup = n.min(p); // Max number of Householder transformations

    // --- Main loop for Householder transformations ---
    for l_idx in 0..lup {
        // l_idx is current column/row for transformation (0-based)
        if pivoting_enabled && l_idx >= pl_idx && l_idx < pu_idx {
            let mut maxnrm = 0.0;
            let mut maxj_idx = l_idx;
            for j_check_idx in l_idx..pu_idx {
                // Search in free columns from l_idx to pu_idx-1
                if qraux[j_check_idx] > maxnrm {
                    maxnrm = qraux[j_check_idx];
                    maxj_idx = j_check_idx;
                }
            }
            if maxj_idx != l_idx {
                swap_matrix_columns(a, lda, n, l_idx, maxj_idx);
                qraux.swap(l_idx, maxj_idx);
                work.swap(l_idx, maxj_idx);
                jpvt.swap(l_idx, maxj_idx);
            }
        }

        qraux[l_idx] = 0.0; // Initialize qraux for current column

        let segment_len = n - l_idx; // Length of vector segment for Householder transform
                                     // (corresponds to Fortran N-L+1 for 0-indexed L)
        if segment_len == 0 {
            // Should only happen if n=0, but then lup=0, loop doesn't run.
            continue; // Or if l_idx = n, but l_idx < n.min(p) <= n.
        }

        let col_l_diag_segment_start_idx = mat_idx(l_idx, l_idx, lda);

        let nrmxl = {
            let col_segment_for_norm =
                &a[col_l_diag_segment_start_idx..(col_l_diag_segment_start_idx + segment_len)];
            dnrm2(segment_len, col_segment_for_norm, 1)
        };

        if nrmxl.abs() > f64::EPSILON {
            let current_a_ll = a[col_l_diag_segment_start_idx];
            let nrmxl_signed = if current_a_ll.abs() > f64::EPSILON {
                nrmxl.copysign(current_a_ll)
            } else {
                nrmxl // If a[l,l] is zero, use positive norm
            };

            if l_idx == n - 1 {
                // Special case: segment_len == 1 (Fortran L == N)
                qraux[l_idx] = current_a_ll; // Store original a[l,l]
                a[col_l_diag_segment_start_idx] = -nrmxl_signed; // Store -sigma
            } else {
                // General case: segment_len > 1 (l_idx < n - 1)
                // Form Householder vector v in a[l_idx.., l_idx]
                let col_l_segment_mut = &mut a
                    [col_l_diag_segment_start_idx..(col_l_diag_segment_start_idx + segment_len)];
                dscal(segment_len, 1.0 / nrmxl_signed, col_l_segment_mut, 1);
                a[col_l_diag_segment_start_idx] += 1.0; // v_0 = 1 + scaled_a_ll

                // Apply transformation to remaining columns
                for j_target_col_idx in (l_idx + 1)..p {
                    let target_col_segment_start_idx = mat_idx(l_idx, j_target_col_idx, lda);
                    let v_0_val = a[col_l_diag_segment_start_idx]; // Householder vector's first element

                    if v_0_val.abs() > f64::EPSILON {
                        let t_val: f64;
                        {
                            // Scope for ddot (immutable borrows)
                            let h_vec_for_ddot = &a[col_l_diag_segment_start_idx
                                ..(col_l_diag_segment_start_idx + segment_len)];
                            let target_col_for_ddot = &a[target_col_segment_start_idx
                                ..(target_col_segment_start_idx + segment_len)];
                            t_val = -ddot(segment_len, h_vec_for_ddot, 1, target_col_for_ddot, 1)
                                / v_0_val;
                        }

                        {
                            // Scope for daxpy (mutable borrow of target column)
                            let (slice_before_target, slice_target_and_after) =
                                a.split_at_mut(target_col_segment_start_idx);
                            let h_vec_for_daxpy = &slice_before_target[col_l_diag_segment_start_idx
                                ..(col_l_diag_segment_start_idx + segment_len)];
                            let target_col_mut_for_daxpy =
                                &mut slice_target_and_after[0..segment_len];
                            daxpy(
                                segment_len,
                                t_val,
                                h_vec_for_daxpy,
                                1,
                                target_col_mut_for_daxpy,
                                1,
                            );
                        }
                    }

                    // Update norm of column j_target_col_idx if it's "free"
                    if j_target_col_idx >= pl_idx
                        && j_target_col_idx < pu_idx
                        && qraux[j_target_col_idx].abs() > f64::EPSILON
                    {
                        let val_a_lj_updated = a[mat_idx(l_idx, j_target_col_idx, lda)]; // a[l,j] after transformation
                        let mut tt_sin_sq =
                            1.0 - (val_a_lj_updated.abs() / qraux[j_target_col_idx]).powi(2);
                        tt_sin_sq = tt_sin_sq.max(0.0);

                        let work_val_j = work[j_target_col_idx]; // Original norm of col j
                        let tt_decision_factor = if work_val_j.abs() < f64::EPSILON {
                            1.0
                        } else {
                            1.0 + 0.05 * tt_sin_sq * (qraux[j_target_col_idx] / work_val_j).powi(2)
                        };

                        if (tt_decision_factor - 1.0).abs() > f64::EPSILON {
                            qraux[j_target_col_idx] *= tt_sin_sq.sqrt();
                        } else {
                            let sub_col_start_row = l_idx + 1;
                            if sub_col_start_row < n {
                                let sub_col_len = n - sub_col_start_row;
                                let sub_col_data_start_idx =
                                    mat_idx(sub_col_start_row, j_target_col_idx, lda);
                                let sub_col_slice_for_norm = &a[sub_col_data_start_idx
                                    ..(sub_col_data_start_idx + sub_col_len)];
                                let new_norm = dnrm2(sub_col_len, sub_col_slice_for_norm, 1);
                                qraux[j_target_col_idx] = new_norm;
                                work[j_target_col_idx] = new_norm; // Update work cache
                            } else {
                                qraux[j_target_col_idx] = 0.0;
                                work[j_target_col_idx] = 0.0;
                            }
                        }
                    }
                } // End loop over j_target_col_idx

                // Save transformation details
                qraux[l_idx] = a[col_l_diag_segment_start_idx]; // Store v_0
                a[col_l_diag_segment_start_idx] = -nrmxl_signed; // Store -sigma
            }
        } // End if nrmxl non-zero
    } // End main loop over l_idx
}
