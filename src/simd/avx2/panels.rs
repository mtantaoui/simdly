use std::cmp::min;

use std::alloc::{self, Layout};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::slice;

use crate::simd::avx2::f32x8::AVX_ALIGNMENT;

pub(crate) const MR: usize = 8;
pub(crate) const NR: usize = 8;
pub(crate) const KC: usize = 256;

/// Calculates the 1D index for a 2D element in a column-major matrix.
///
/// # Arguments
/// * `i` - Row index.
/// * `j` - Column index.
/// * `ld` - Leading dimension (number of rows in the matrix).
#[inline(always)]
pub(crate) fn at(i: usize, j: usize, ld: usize) -> usize {
    (j * ld) + i
}

/// A packed panel of matrix B with dimensions KC×NR.
///
/// Data is stored in **row-major** format: `data[k][j]` contains B(k,j) where:
/// - k ranges from 0 to KC-1 (rows from the KC×NR block of B)
/// - j ranges from 0 to NR-1 (columns within this NR-wide panel)
///
/// This layout enables the microkernel to:
/// 1. Load a full row `data[k]` to get B(k, 0..NR-1)
/// 2. Broadcast individual elements B(k,j) efficiently
///
/// The `#[repr(C, align(32))]` ensures 32-byte alignment for AVX2 operations.
#[repr(C, align(32))]
pub struct BPanel<const KC: usize, const NR: usize> {
    pub data: [[f32; NR]; KC],
}

/// A heap-allocated block containing multiple B panels.
///
/// Manages memory for ceil(nc/NR) panels of KC×NR elements each, where nc is the
/// number of columns from the original B matrix block being packed.
///
/// Memory layout: [Panel₀][Panel₁]...[Panelₙ] with 32-byte alignment.
pub struct BBlock<const KC: usize, const NR: usize> {
    /// A raw pointer to the heap-allocated, aligned block of `BPanel`s.
    ptr: *mut BPanel<KC, NR>,
    /// The number of `BPanel`s allocated in the memory block.
    num_panels: usize,
    /// The memory `Layout` used for allocation. Storing this guarantees that
    /// deallocation is performed with the exact same layout, which is required for safety.
    layout: Layout,
    /// The original number of columns from matrix `B` packed into this block.
    pub nc: usize,
    /// `PhantomData` informs the Rust compiler that this struct "owns" the data
    /// pointed to by `ptr`, enabling proper borrow checking and drop semantics.
    _marker: PhantomData<BPanel<KC, NR>>,
}

impl<const KC: usize, const NR: usize> BBlock<KC, NR> {
    /// Allocates zero-initialized, aligned memory for packing nc columns.
    ///
    /// Creates ceil(nc/NR) panels to hold all nc columns. Panels beyond the first
    /// may be partially filled if nc is not a multiple of NR.
    ///
    /// # Arguments
    /// * `nc` - Number of columns from original B matrix to pack
    ///
    /// # Returns
    /// New BBlock or allocation error
    #[inline(always)]
    pub fn new(nc: usize) -> Result<Self, Layout> {
        // Calculate panels needed: ceil(nc / NR)
        let num_panels = nc.div_ceil(NR);

        // Define the memory layout for an array of `BPanel`s.
        let layout = Layout::array::<BPanel<KC, NR>>(num_panels).unwrap();

        // Ensure the layout meets our minimum alignment requirement.
        let layout = layout.align_to(AVX_ALIGNMENT).unwrap();

        let ptr = unsafe {
            // Zero-initialize since partial panels need zero-padding
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            raw_ptr.cast::<BPanel<KC, NR>>()
        };

        Ok(BBlock {
            ptr,
            num_panels,
            layout,
            nc,
            _marker: PhantomData,
        })
    }

    /// Returns an immutable slice view of the `BPanel`s.
    #[inline(always)]
    pub fn as_panels(&self) -> &[BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_panels) }
    }

    /// Returns a mutable slice view of the `BPanel`s.
    #[inline(always)]
    pub fn as_panels_mut(&mut self) -> &mut [BPanel<KC, NR>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_panels) }
    }
}

/// The `Drop` implementation ensures that the heap-allocated memory is safely
/// deallocated when a `BBlock` goes out of scope.
impl<const KC: usize, const NR: usize> Drop for BBlock<KC, NR> {
    #[inline(always)]
    fn drop(&mut self) {
        // Deallocating with a zero-sized layout is undefined behavior.
        // This check prevents UB if a BBlock was created with `nc = 0`.
        if self.layout.size() > 0 {
            unsafe {
                // Deallocate the memory using the exact layout that was stored
                // during allocation. This is the only safe way to deallocate
                // memory obtained from the global allocator.
                alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

// --- Convenience Indexing for BBlock ---

impl<const KC: usize, const NR: usize> Index<usize> for BBlock<KC, NR> {
    type Output = BPanel<KC, NR>;
    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_panels()[index]
    }
}

impl<const KC: usize, const NR: usize> IndexMut<usize> for BBlock<KC, NR> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_panels_mut()[index]
    }
}

// --- Packed Data Structures for Matrix A ---

/// A packed panel of matrix A with dimensions MR×KC.
///
/// Data is stored to optimize microkernel access: `data[k]` contains the k-th column
/// of an MR×KC block from matrix A, i.e., A(0..MR-1, k).
///
/// This layout enables the microkernel to:
/// 1. Load `data[k]` as a vector: A(0..MR-1, k)
/// 2. Process entire columns efficiently with SIMD
///
/// The `#[repr(C, align(32))]` ensures 32-byte alignment for AVX2 loads.
#[repr(C, align(32))]
pub struct APanel<const MR: usize, const KC: usize> {
    pub data: [[f32; MR]; KC],
}

/// A heap-allocated block containing multiple A panels.
///
/// Manages memory for ceil(mc/MR) panels of MR×KC elements each, where mc is the
/// number of rows from the original A matrix block being packed.
pub struct ABlock<const MR: usize, const KC: usize> {
    /// A raw pointer to the heap-allocated, aligned block of `APanel`s.
    ptr: *mut APanel<MR, KC>,
    /// The number of `APanel`s allocated in the memory block.
    num_panels: usize,
    /// The memory `Layout` used for allocation, required for safe deallocation.
    layout: Layout,
    /// The original number of rows from matrix `A` packed into this block.
    pub mc: usize,
    /// `PhantomData` for ownership and drop-check semantics.
    _marker: PhantomData<APanel<MR, KC>>,
}

impl<const MR: usize, const KC: usize> ABlock<MR, KC> {
    /// Allocates zero-initialized, aligned memory for packing mc rows.
    #[inline(always)]
    pub fn new(mc: usize) -> Result<Self, Layout> {
        let num_panels = mc.div_ceil(MR);

        let layout = Layout::array::<APanel<MR, KC>>(num_panels)
            .expect("Invalid layout for APanel")
            .align_to(AVX_ALIGNMENT)
            .unwrap();

        let ptr = unsafe {
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            raw_ptr.cast::<APanel<MR, KC>>()
        };

        Ok(ABlock {
            ptr,
            num_panels,
            layout,
            mc,
            _marker: PhantomData,
        })
    }

    /// Returns an immutable slice view of the `APanel`s.
    #[inline(always)]
    pub fn as_panels(&self) -> &[APanel<MR, KC>] {
        unsafe { slice::from_raw_parts(self.ptr, self.num_panels) }
    }

    /// Returns a mutable slice view of the `APanel`s.
    pub fn as_panels_mut(&mut self) -> &mut [APanel<MR, KC>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.num_panels) }
    }
}

/// Safe deallocation for `ABlock`'s memory.
impl<const MR: usize, const KC: usize> Drop for ABlock<MR, KC> {
    #[inline(always)]
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe {
                alloc::dealloc(self.ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

/// Convenience indexing for `ABlock`.
impl<const MR: usize, const KC: usize> Index<usize> for ABlock<MR, KC> {
    type Output = APanel<MR, KC>;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.as_panels()[index]
    }
}

impl<const MR: usize, const KC: usize> IndexMut<usize> for ABlock<MR, KC> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_panels_mut()[index]
    }
}

/// High-performance packing of an mc×kc block of matrix A using algorithmic optimizations.
///
/// Extracts A(ic:ic+mc-1, pc:pc+kc-1) and reorganizes it into MR-wide row panels,
/// where each panel stores KC columns in a layout optimized for microkernel access.
/// Uses cache-friendly memory access patterns and eliminates function call overhead.
///
/// # Arguments
/// * `a` - Source matrix A in column-major order
/// * `mc`, `kc` - Block dimensions to pack  
/// * `m` - Leading dimension (number of rows) of matrix A
/// * `ic`, `pc` - Top-left coordinates of block in A
#[inline]
pub fn pack_a<const MR: usize, const KC: usize>(
    a: &[f32],
    mc: usize,
    kc: usize,
    m: usize,
    ic: usize,
    pc: usize,
) -> ABlock<MR, KC> {
    let mut packed_block = ABlock::<MR, KC>::new(mc).expect("Memory allocation failed for ABlock");

    // Pre-calculate base addresses to eliminate redundant calculations
    let base_src_row = ic;
    let base_src_col_offset = pc * m;

    // Process mc rows in groups of MR (microkernel row dimension)
    for (panel_idx, i_panel_start) in (0..mc).step_by(MR).enumerate() {
        let dest_panel = &mut packed_block[panel_idx];
        let mr_in_panel = min(MR, mc - i_panel_start);
        
        // Calculate source row offset once per panel
        let panel_src_row_offset = base_src_row + i_panel_start;
        
        // Pack all KC columns of this row panel with optimized inner loop
        for p_col in 0..kc {
            // Inline index calculation - eliminates function call overhead
            let src_start = base_src_col_offset + p_col * m + panel_src_row_offset;
            
            let dest_col = &mut dest_panel.data[p_col];
            
            // Optimized copy with manual unrolling for common cases
            match mr_in_panel {
                8 => {
                    // Full panel - most common case, manually unrolled
                    dest_col[0] = a[src_start];
                    dest_col[1] = a[src_start + 1];
                    dest_col[2] = a[src_start + 2];
                    dest_col[3] = a[src_start + 3];
                    dest_col[4] = a[src_start + 4];
                    dest_col[5] = a[src_start + 5];
                    dest_col[6] = a[src_start + 6];
                    dest_col[7] = a[src_start + 7];
                },
                4 => {
                    // Half panel
                    dest_col[0] = a[src_start];
                    dest_col[1] = a[src_start + 1];
                    dest_col[2] = a[src_start + 2];
                    dest_col[3] = a[src_start + 3];
                },
                mr => {
                    // Partial panel - use slice copy for irregular sizes
                    let src_slice = &a[src_start..src_start + mr];
                    let dest_slice = &mut dest_col[0..mr];
                    dest_slice.copy_from_slice(src_slice);
                }
            }
        }
    }

    packed_block
}

/// High-performance packing of a kc×nc block of matrix B using algorithmic optimizations.
///
/// Extracts B(pc:pc+kc-1, jc:jc+nc-1) and reorganizes it into NR-wide column panels,
/// where each panel stores KC rows in row-major format for efficient broadcasting.
/// Eliminates nested loops, function call overhead, and optimizes memory access patterns.
///
/// # Arguments  
/// * `b` - Source matrix B in column-major order
/// * `nc`, `kc` - Block dimensions to pack
/// * `k` - Leading dimension (number of rows) of matrix B
/// * `pc`, `jc` - Top-left coordinates of block in B
#[inline]
pub fn pack_b<const KC: usize, const NR: usize>(
    b: &[f32],
    nc: usize,
    kc: usize,
    k: usize,
    pc: usize,
    jc: usize,
) -> BBlock<KC, NR> {
    let mut packed_block = BBlock::<KC, NR>::new(nc).expect("Memory allocation failed for BBlock");

    // Process nc columns in groups of NR (microkernel column dimension)
    for (panel_idx, j_panel_start) in (0..nc).step_by(NR).enumerate() {
        let dest_panel = &mut packed_block[panel_idx];
        let nr_in_panel = min(NR, nc - j_panel_start);

        // Pre-calculate column base addresses to eliminate function calls
        let col_base = jc + j_panel_start;
        
        // Pack KC rows of this column panel with optimized inner loops
        for p_row in 0..kc {
            let src_row = pc + p_row;
            let dest_row = &mut dest_panel.data[p_row];
            
            // Optimized packing with manual unrolling for common cases
            match nr_in_panel {
                8 => {
                    // Full panel - most common case, manually unrolled
                    // Inline index calculation: col * k + row
                    dest_row[0] = b[(col_base + 0) * k + src_row];
                    dest_row[1] = b[(col_base + 1) * k + src_row];
                    dest_row[2] = b[(col_base + 2) * k + src_row];
                    dest_row[3] = b[(col_base + 3) * k + src_row];
                    dest_row[4] = b[(col_base + 4) * k + src_row];
                    dest_row[5] = b[(col_base + 5) * k + src_row];
                    dest_row[6] = b[(col_base + 6) * k + src_row];
                    dest_row[7] = b[(col_base + 7) * k + src_row];
                },
                4 => {
                    // Half panel
                    dest_row[0] = b[(col_base + 0) * k + src_row];
                    dest_row[1] = b[(col_base + 1) * k + src_row];
                    dest_row[2] = b[(col_base + 2) * k + src_row];
                    dest_row[3] = b[(col_base + 3) * k + src_row];
                },
                6 => {
                    // 3/4 panel
                    dest_row[0] = b[(col_base + 0) * k + src_row];
                    dest_row[1] = b[(col_base + 1) * k + src_row];
                    dest_row[2] = b[(col_base + 2) * k + src_row];
                    dest_row[3] = b[(col_base + 3) * k + src_row];
                    dest_row[4] = b[(col_base + 4) * k + src_row];
                    dest_row[5] = b[(col_base + 5) * k + src_row];
                },
                2 => {
                    // Quarter panel
                    dest_row[0] = b[(col_base + 0) * k + src_row];
                    dest_row[1] = b[(col_base + 1) * k + src_row];
                },
                1 => {
                    // Single element
                    dest_row[0] = b[col_base * k + src_row];
                },
                nr => {
                    // General case for other sizes - still optimized with eliminated function calls
                    for j_col_in_panel in 0..nr {
                        let src_col = col_base + j_col_in_panel;
                        dest_row[j_col_in_panel] = b[src_col * k + src_row];
                    }
                }
            }
        }
    }

    packed_block
}
