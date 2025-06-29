use std::{
    alloc::{alloc, handle_alloc_error, Layout},
    cmp::min,
    mem,
    ptr::copy_nonoverlapping,
};

use num::Float;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

const MR: usize = 8;
const NR: usize = 8;

const NC: usize = 1024;
const KC: usize = 256;
const MC: usize = 64;

const AVX_ALIGNMENT: usize = 32;

/// Packs a panel of A into a buffer in column-major micro-panel format.
/// The kernel expects this layout to stream columns of A.
/// Layout: For each k, store MR consecutive elements: [a(0,k), a(1,k), ..., a(MR-1,k)]
unsafe fn pack_a(
    mc: usize,
    kc: usize,
    packed_buffer: *mut f32,
    a: *const f32,
    rsa: isize,
    csa: isize,
) {
    let mut pack_ptr = packed_buffer;

    // For each micro-panel of MR rows
    for i_panel in (0..mc).step_by(MR) {
        let panel_size = min(MR, mc - i_panel);

        // For each column k in this micro-panel
        for k in 0..kc {
            // Pack MR elements from column k
            for i in 0..MR {
                if i < panel_size {
                    // Valid element: get a[i_panel + i, k]
                    let src_ptr = a.offset((i_panel + i) as isize * rsa + k as isize * csa);
                    *pack_ptr = *src_ptr;
                } else {
                    // Padding for incomplete micro-panels
                    *pack_ptr = 0.0;
                }
                pack_ptr = pack_ptr.add(1);
            }
        }
    }
}

/// Packs a panel of B into a buffer in row-major micro-panel format.
/// The kernel expects this layout to stream rows of B.
/// Layout: For each k, store NR consecutive elements: [b(k,0), b(k,1), ..., b(k,NR-1)]
unsafe fn pack_b(
    kc: usize,
    nc: usize,
    packed_buffer: *mut f32,
    b: *const f32,
    rsb: isize,
    csb: isize,
) {
    let mut pack_ptr = packed_buffer;

    // For each micro-panel of NR columns
    for j_panel in (0..nc).step_by(NR) {
        let panel_size = min(NR, nc - j_panel);

        // For each row k in this micro-panel
        for k in 0..kc {
            // Pack NR elements from row k
            for j in 0..NR {
                if j < panel_size {
                    // Valid element: get b[k, j_panel + j]
                    let src_ptr = b.offset(k as isize * rsb + (j_panel + j) as isize * csb);
                    *pack_ptr = *src_ptr;
                } else {
                    // Padding for incomplete micro-panels
                    *pack_ptr = 0.0;
                }
                pack_ptr = pack_ptr.add(1);
            }
        }
    }
}

unsafe fn my_sgemm(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    rsa: isize,
    csa: isize,
    b: &[f32],
    rsb: isize,
    csb: isize,
    beta: f32,
    c: &mut [f32],
    rsc: isize,
    csc: isize,
) {
    if m == 0 || n == 0 {
        return;
    }

    if k == 0 {
        if beta == 0.0 {
            for i in 0..m {
                for j in 0..n {
                    *c.as_mut_ptr().offset(i as isize * rsc + j as isize * csc) = 0.0;
                }
            }
        } else if beta != 1.0 {
            for i in 0..m {
                for j in 0..n {
                    let c_ptr = c.as_mut_ptr().offset(i as isize * rsc + j as isize * csc);
                    *c_ptr *= beta;
                }
            }
        }
        return;
    }

    let kernel_nc = NC;
    let kernel_kc = KC;
    let kernel_mc = MC;

    let packing_buffer: PackedBuffer<f32> = PackedBuffer::new(m, n, k);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    for jc in (0..n).step_by(kernel_nc) {
        let nc = min(kernel_nc, n - jc);
        let b_col_panel = b_ptr.offset(jc as isize * csb);
        let c_col_panel = c_ptr.offset(jc as isize * csc);

        for pc in (0..k).step_by(kernel_kc) {
            let kc = min(kernel_kc, k - pc);
            let b_panel = b_col_panel.offset(pc as isize * rsb);
            let a_panel = a_ptr.offset(pc as isize * csa);

            pack_b(kc, nc, packing_buffer.packed_b_ptr, b_panel, rsb, csb);

            let betap = if pc == 0 { beta } else { 1.0 };

            for ic in (0..m).step_by(kernel_mc) {
                let mc = min(kernel_mc, m - ic);
                let a_micropanel = a_panel.offset(ic as isize * rsa);
                let c_micropanel = c_col_panel.offset(ic as isize * rsc);

                pack_a(mc, kc, packing_buffer.packed_a_ptr, a_micropanel, rsa, csa);

                sgemm_packed(
                    nc,
                    kc,
                    mc,
                    alpha,
                    packing_buffer.packed_a_ptr,
                    packing_buffer.packed_b_ptr,
                    betap,
                    c_micropanel,
                    rsc,
                    csc,
                );
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

pub unsafe fn kernel_8x8(
    k: usize,
    alpha: f32,
    a: *const f32,
    b: *const f32,
    beta: f32,
    c: *mut f32,
    rsc: isize,
    csc: isize,
) {
    const MR: usize = 8; // Number of rows
    const NR: usize = 8; // Number of columns

    debug_assert_ne!(k, 0);

    let mut ab = [_mm256_setzero_ps(); MR];

    // This kernel can operate in either transposition (C = A B or C^T = B^T A^T)
    let prefer_row_major_c = rsc != 1;

    let (mut a, mut b) = if prefer_row_major_c { (a, b) } else { (b, a) };
    let (rsc, csc) = if prefer_row_major_c {
        (rsc, csc)
    } else {
        (csc, rsc)
    };

    // Shuffle and permute mask constants
    const PERM32_2301: i32 = (1 << 6) | (0 << 4) | (3 << 2) | 2; // 0x4E
    const PERM128_30: i32 = (0 << 4) | 3; // 0x03
    const SHUF_0123: i32 = (3 << 6) | (2 << 4) | (1 << 2) | 0; // 0xE4
    const PERM128_02: i32 = (2 << 4) | 0; // 0x20
    const PERM128_31: i32 = (1 << 4) | 3; // 0x13
    const PERM32_ROTATE: i32 = (0 << 6) | (3 << 4) | (2 << 2) | 1; // 0x39

    // Start data load before each iteration
    let mut av = _mm256_load_ps(a);
    let mut bv = _mm256_load_ps(b);

    // Main computation loop - unrolled by 4
    let mut remaining_k = k;

    // Process in chunks of 4
    while remaining_k >= 4 {
        for _ in 0..4 {
            // Load and duplicate even elements: a0 a0 a2 a2 a4 a4 a6 a6
            let a0246 = _mm256_moveldup_ps(av);
            let a2064 = _mm256_permute_ps(a0246, PERM32_2301);

            // Load and duplicate odd elements: a1 a1 a3 a3 a5 a5 a7 a7
            let a1357 = _mm256_movehdup_ps(av);
            let a3175 = _mm256_permute_ps(a1357, PERM32_2301);

            let bv_lh = _mm256_permute2f128_ps(bv, bv, PERM128_30);

            // Compute partial products using FMA if available
            ab[0] = _mm256_fmadd_ps(a0246, bv, ab[0]);
            ab[1] = _mm256_fmadd_ps(a2064, bv, ab[1]);
            ab[2] = _mm256_fmadd_ps(a0246, bv_lh, ab[2]);
            ab[3] = _mm256_fmadd_ps(a2064, bv_lh, ab[3]);

            ab[4] = _mm256_fmadd_ps(a1357, bv, ab[4]);
            ab[5] = _mm256_fmadd_ps(a3175, bv, ab[5]);
            ab[6] = _mm256_fmadd_ps(a1357, bv_lh, ab[6]);
            ab[7] = _mm256_fmadd_ps(a3175, bv_lh, ab[7]);

            // Advance pointers for next iteration
            if remaining_k > 1 {
                a = a.add(MR);
                b = b.add(NR);
                bv = _mm256_load_ps(b);
                av = _mm256_load_ps(a);
            }

            remaining_k -= 1;
            if remaining_k == 0 {
                break;
            }
        }
    }

    // Handle remaining iterations (less than 4)
    while remaining_k > 0 {
        let a0246 = _mm256_moveldup_ps(av);
        let a2064 = _mm256_permute_ps(a0246, PERM32_2301);

        let a1357 = _mm256_movehdup_ps(av);
        let a3175 = _mm256_permute_ps(a1357, PERM32_2301);

        let bv_lh = _mm256_permute2f128_ps(bv, bv, PERM128_30);

        ab[0] = _mm256_fmadd_ps(a0246, bv, ab[0]);
        ab[1] = _mm256_fmadd_ps(a2064, bv, ab[1]);
        ab[2] = _mm256_fmadd_ps(a0246, bv_lh, ab[2]);
        ab[3] = _mm256_fmadd_ps(a2064, bv_lh, ab[3]);

        ab[4] = _mm256_fmadd_ps(a1357, bv, ab[4]);
        ab[5] = _mm256_fmadd_ps(a3175, bv, ab[5]);
        ab[6] = _mm256_fmadd_ps(a1357, bv_lh, ab[6]);
        ab[7] = _mm256_fmadd_ps(a3175, bv_lh, ab[7]);

        if remaining_k > 1 {
            a = a.add(MR);
            b = b.add(NR);
            bv = _mm256_load_ps(b);
            av = _mm256_load_ps(a);
        }

        remaining_k -= 1;
    }

    let alphav = _mm256_set1_ps(alpha);

    // Permute to put the abij elements in order
    let ab0246 = ab[0];
    let ab2064 = ab[1];
    let ab4602 = ab[2]; // reverse order
    let ab6420 = ab[3]; // reverse order

    let ab1357 = ab[4];
    let ab3175 = ab[5];
    let ab5713 = ab[6]; // reverse order
    let ab7531 = ab[7]; // reverse order

    // First level of shuffling
    let ab0044 = _mm256_shuffle_ps(ab0246, ab2064, SHUF_0123);
    let ab2266 = _mm256_shuffle_ps(ab2064, ab0246, SHUF_0123);

    let ab4400 = _mm256_shuffle_ps(ab4602, ab6420, SHUF_0123);
    let ab6622 = _mm256_shuffle_ps(ab6420, ab4602, SHUF_0123);

    let ab1155 = _mm256_shuffle_ps(ab1357, ab3175, SHUF_0123);
    let ab3377 = _mm256_shuffle_ps(ab3175, ab1357, SHUF_0123);

    let ab5511 = _mm256_shuffle_ps(ab5713, ab7531, SHUF_0123);
    let ab7733 = _mm256_shuffle_ps(ab7531, ab5713, SHUF_0123);

    // Second level of permutation
    let ab0000 = _mm256_permute2f128_ps(ab0044, ab4400, PERM128_02);
    let ab4444 = _mm256_permute2f128_ps(ab0044, ab4400, PERM128_31);

    let ab2222 = _mm256_permute2f128_ps(ab2266, ab6622, PERM128_02);
    let ab6666 = _mm256_permute2f128_ps(ab2266, ab6622, PERM128_31);

    let ab1111 = _mm256_permute2f128_ps(ab1155, ab5511, PERM128_02);
    let ab5555 = _mm256_permute2f128_ps(ab1155, ab5511, PERM128_31);

    let ab3333 = _mm256_permute2f128_ps(ab3377, ab7733, PERM128_02);
    let ab7777 = _mm256_permute2f128_ps(ab3377, ab7733, PERM128_31);

    ab[0] = ab0000;
    ab[1] = ab1111;
    ab[2] = ab2222;
    ab[3] = ab3333;
    ab[4] = ab4444;
    ab[5] = ab5555;
    ab[6] = ab6666;
    ab[7] = ab7777;

    // Helper function to compute C matrix offset
    let c_offset =
        |i: usize, j: usize| -> *mut f32 { c.offset(rsc * i as isize + csc * j as isize) };

    // C ← α A B + β C
    let mut cv = [_mm256_setzero_ps(); MR];

    if beta != 0.0 {
        let betav = _mm256_set1_ps(beta);

        // Read C
        if csc == 1 {
            // Contiguous columns - can load directly
            cv[0] = _mm256_loadu_ps(c_offset(0, 0));
            cv[1] = _mm256_loadu_ps(c_offset(1, 0));
            cv[2] = _mm256_loadu_ps(c_offset(2, 0));
            cv[3] = _mm256_loadu_ps(c_offset(3, 0));
            cv[4] = _mm256_loadu_ps(c_offset(4, 0));
            cv[5] = _mm256_loadu_ps(c_offset(5, 0));
            cv[6] = _mm256_loadu_ps(c_offset(6, 0));
            cv[7] = _mm256_loadu_ps(c_offset(7, 0));
        } else {
            // Strided columns - gather elements manually
            cv[0] = _mm256_setr_ps(
                *c_offset(0, 0),
                *c_offset(0, 1),
                *c_offset(0, 2),
                *c_offset(0, 3),
                *c_offset(0, 4),
                *c_offset(0, 5),
                *c_offset(0, 6),
                *c_offset(0, 7),
            );
            cv[1] = _mm256_setr_ps(
                *c_offset(1, 0),
                *c_offset(1, 1),
                *c_offset(1, 2),
                *c_offset(1, 3),
                *c_offset(1, 4),
                *c_offset(1, 5),
                *c_offset(1, 6),
                *c_offset(1, 7),
            );
            cv[2] = _mm256_setr_ps(
                *c_offset(2, 0),
                *c_offset(2, 1),
                *c_offset(2, 2),
                *c_offset(2, 3),
                *c_offset(2, 4),
                *c_offset(2, 5),
                *c_offset(2, 6),
                *c_offset(2, 7),
            );
            cv[3] = _mm256_setr_ps(
                *c_offset(3, 0),
                *c_offset(3, 1),
                *c_offset(3, 2),
                *c_offset(3, 3),
                *c_offset(3, 4),
                *c_offset(3, 5),
                *c_offset(3, 6),
                *c_offset(3, 7),
            );
            cv[4] = _mm256_setr_ps(
                *c_offset(4, 0),
                *c_offset(4, 1),
                *c_offset(4, 2),
                *c_offset(4, 3),
                *c_offset(4, 4),
                *c_offset(4, 5),
                *c_offset(4, 6),
                *c_offset(4, 7),
            );
            cv[5] = _mm256_setr_ps(
                *c_offset(5, 0),
                *c_offset(5, 1),
                *c_offset(5, 2),
                *c_offset(5, 3),
                *c_offset(5, 4),
                *c_offset(5, 5),
                *c_offset(5, 6),
                *c_offset(5, 7),
            );
            cv[6] = _mm256_setr_ps(
                *c_offset(6, 0),
                *c_offset(6, 1),
                *c_offset(6, 2),
                *c_offset(6, 3),
                *c_offset(6, 4),
                *c_offset(6, 5),
                *c_offset(6, 6),
                *c_offset(6, 7),
            );
            cv[7] = _mm256_setr_ps(
                *c_offset(7, 0),
                *c_offset(7, 1),
                *c_offset(7, 2),
                *c_offset(7, 3),
                *c_offset(7, 4),
                *c_offset(7, 5),
                *c_offset(7, 6),
                *c_offset(7, 7),
            );
        }

        // Scale by beta: βC
        cv[0] = _mm256_mul_ps(cv[0], betav);
        cv[1] = _mm256_mul_ps(cv[1], betav);
        cv[2] = _mm256_mul_ps(cv[2], betav);
        cv[3] = _mm256_mul_ps(cv[3], betav);
        cv[4] = _mm256_mul_ps(cv[4], betav);
        cv[5] = _mm256_mul_ps(cv[5], betav);
        cv[6] = _mm256_mul_ps(cv[6], betav);
        cv[7] = _mm256_mul_ps(cv[7], betav);
    }

    // Compute C = αAB + βC using FMA
    cv[0] = _mm256_fmadd_ps(alphav, ab[0], cv[0]);
    cv[1] = _mm256_fmadd_ps(alphav, ab[1], cv[1]);
    cv[2] = _mm256_fmadd_ps(alphav, ab[2], cv[2]);
    cv[3] = _mm256_fmadd_ps(alphav, ab[3], cv[3]);
    cv[4] = _mm256_fmadd_ps(alphav, ab[4], cv[4]);
    cv[5] = _mm256_fmadd_ps(alphav, ab[5], cv[5]);
    cv[6] = _mm256_fmadd_ps(alphav, ab[6], cv[6]);
    cv[7] = _mm256_fmadd_ps(alphav, ab[7], cv[7]);

    // Store C back to memory
    if csc == 1 {
        // Contiguous columns - can store directly
        _mm256_storeu_ps(c_offset(0, 0), cv[0]);
        _mm256_storeu_ps(c_offset(1, 0), cv[1]);
        _mm256_storeu_ps(c_offset(2, 0), cv[2]);
        _mm256_storeu_ps(c_offset(3, 0), cv[3]);
        _mm256_storeu_ps(c_offset(4, 0), cv[4]);
        _mm256_storeu_ps(c_offset(5, 0), cv[5]);
        _mm256_storeu_ps(c_offset(6, 0), cv[6]);
        _mm256_storeu_ps(c_offset(7, 0), cv[7]);
    } else {
        // Strided columns - scatter elements individually
        for i in 0..8 {
            let cvlo = _mm256_extractf128_ps(cv[i], 0);
            let cvhi = _mm256_extractf128_ps(cv[i], 1);

            // Store lower 128 bits (4 elements)
            _mm_store_ss(c_offset(i, 0), cvlo);
            let cperm = _mm_permute_ps(cvlo, PERM32_ROTATE);
            _mm_store_ss(c_offset(i, 1), cperm);
            let cperm = _mm_permute_ps(cperm, PERM32_ROTATE);
            _mm_store_ss(c_offset(i, 2), cperm);
            let cperm = _mm_permute_ps(cperm, PERM32_ROTATE);
            _mm_store_ss(c_offset(i, 3), cperm);

            // Store upper 128 bits (4 elements)
            _mm_store_ss(c_offset(i, 4), cvhi);
            let cperm = _mm_permute_ps(cvhi, PERM32_ROTATE);
            _mm_store_ss(c_offset(i, 5), cperm);
            let cperm = _mm_permute_ps(cperm, PERM32_ROTATE);
            _mm_store_ss(c_offset(i, 6), cperm);
            let cperm = _mm_permute_ps(cperm, PERM32_ROTATE);
            _mm_store_ss(c_offset(i, 7), cperm);
        }
    }
}
#[repr(align(32))]
struct PackedBuffer<F: Float> {
    ptr: *mut F,
    len: usize,
    packed_a_ptr: *mut F,
    packed_a_len: usize,
    packed_b_ptr: *mut F,
    packed_b_len: usize,
    alignment: usize,
}

impl<F: Float> PackedBuffer<F> {
    fn new(m: usize, n: usize, k: usize) -> Self {
        // Calculate buffer sizes with proper rounding up
        let packed_a_len = min(k, KC) * ((min(m, MC) + MR - 1) / MR) * MR;
        let packed_b_len = min(k, KC) * ((min(n, NC) + NR - 1) / NR) * NR;

        let buffer_len = packed_a_len + packed_b_len;

        let layout = unsafe {
            Layout::from_size_align_unchecked(mem::size_of::<F>() * buffer_len, AVX_ALIGNMENT)
        };

        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        let packed_a_ptr: *mut F = ptr as *mut F;
        let packed_b_ptr = unsafe { packed_a_ptr.add(packed_a_len) };

        Self {
            ptr: ptr as *mut F,
            len: buffer_len,
            packed_a_ptr,
            packed_a_len,
            packed_b_ptr,
            packed_b_len,
            alignment: AVX_ALIGNMENT,
        }
    }
}

/// Loops 1 and 2 around the µ-kernel
unsafe fn sgemm_packed(
    nc: usize,
    kc: usize,
    mc: usize,
    alpha: f32,
    a_packed_pointer: *const f32,
    b_packed_pointer: *const f32,
    beta: f32,
    c: *mut f32,
    rsc: isize,
    csc: isize,
) {
    let mr = MR;
    let nr = NR;

    // Calculate the stride for accessing packed data
    let num_mr_panels = (mc + mr - 1) / mr;
    let num_nr_panels = (nc + nr - 1) / nr;

    for jr in 0..num_nr_panels {
        // Each B panel has kc * nr elements
        let bpp = b_packed_pointer.add(jr * kc * nr);
        let c_col = c.offset((jr * nr) as isize * csc);

        for ir in 0..num_mr_panels {
            // Each A panel has kc * mr elements
            let app = a_packed_pointer.add(ir * kc * mr);
            let c_block = c_col.offset((ir * mr) as isize * rsc);

            kernel_8x8(kc, alpha, app, bpp, beta, c_block, rsc, csc);
        }
    }
}

// Drop implementation for PackedBuffer
impl<F: Float> Drop for PackedBuffer<F> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let layout = Layout::from_size_align_unchecked(
                    mem::size_of::<F>() * self.len,
                    self.alignment,
                );
                std::alloc::dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

// --- TEST SUITE ---
fn naive_sgemm(
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    rsa: isize,
    csa: isize,
    b: &[f32],
    rsb: isize,
    csb: isize,
    beta: f32,
    c: &mut [f32],
    rsc: isize,
    csc: isize,
) {
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[(i as isize * rsc + j as isize * csc) as usize] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let c_idx = (i as isize * rsc + j as isize * csc) as usize;
                c[c_idx] *= beta;
            }
        }
    }
    if k == 0 {
        return;
    }
    for i in 0..m {
        for j in 0..n {
            let mut dot_product = 0.0;
            for l in 0..k {
                let a_val = a[(i as isize * rsa + l as isize * csa) as usize];
                let b_val = b[(l as isize * rsb + j as isize * csb) as usize];
                dot_product += a_val * b_val;
            }
            c[(i as isize * rsc + j as isize * csc) as usize] += alpha * dot_product;
        }
    }
}

fn check_results(
    m: usize,
    n: usize,
    c_impl: &[f32],
    c_ref: &[f32],
    rsc: isize,
    csc: isize,
) -> bool {
    let epsilon = 1e-3;
    for i in 0..m {
        for j in 0..n {
            let idx = (i as isize * rsc + j as isize * csc) as usize;
            let ref_val = c_ref[idx];
            let impl_val = c_impl[idx];
            if (ref_val - impl_val).abs() > epsilon {
                println!(
                    "Mismatch at C({}, {}): reference = {}, implementation = {}",
                    i, j, ref_val, impl_val
                );
                return false;
            }
        }
    }
    true
}

fn main() {
    if !is_x86_feature_detected!("avx") || !is_x86_feature_detected!("fma") {
        println!("AVX or FMA not detected on this CPU. Cannot run the test.");
        return;
    }

    let m = 77;
    let k = 49;
    let n = 3;
    let alpha = 1.0;
    let beta = 1.0;

    let a: Vec<f32> = (0..(m * k)).map(|i| (i % 100) as f32 / 10.0).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i % 100) as f32 / 10.0).collect();
    let c_orig: Vec<f32> = (0..(m * n)).map(|i| (i % 100) as f32 / 10.0).collect();

    println!("--- Testing with row-major C matrix ---");
    let mut c_impl = c_orig.clone();
    let mut c_ref = c_orig.clone();
    let rsa = k as isize;
    let csa = 1;
    let rsb = n as isize;
    let csb = 1;
    let rsc = n as isize;
    let csc = 1;
    unsafe {
        my_sgemm(
            m,
            k,
            n,
            alpha,
            &a,
            rsa,
            csa,
            &b,
            rsb,
            csb,
            beta,
            &mut c_impl,
            rsc,
            csc,
        );
    }
    naive_sgemm(
        m, k, n, alpha, &a, rsa, csa, &b, rsb, csb, beta, &mut c_ref, rsc, csc,
    );
    if check_results(m, n, &c_impl, &c_ref, rsc, csc) {
        println!("Test PASSED for row-major layout.");
    } else {
        println!("Test FAILED for row-major layout.");
    }

    println!("\n--- Testing with column-major C matrix ---");
    let mut c_impl = c_orig.clone();
    let mut c_ref = c_orig.clone();
    let rsc_col = 1;
    let csc_col = m as isize;
    unsafe {
        my_sgemm(
            m,
            k,
            n,
            alpha,
            &a,
            rsa,
            csa,
            &b,
            rsb,
            csb,
            beta,
            &mut c_impl,
            rsc_col,
            csc_col,
        );
    }
    naive_sgemm(
        m, k, n, alpha, &a, rsa, csa, &b, rsb, csb, beta, &mut c_ref, rsc_col, csc_col,
    );
    if check_results(m, n, &c_impl, &c_ref, rsc_col, csc_col) {
        println!("Test PASSED for column-major C layout.");
    } else {
        println!("Test FAILED for column-major C layout.");
    }
}
