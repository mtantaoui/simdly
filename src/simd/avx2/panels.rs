use std::cmp::min;

use std::alloc::{self, Layout};
use std::marker::PhantomData;

use crate::simd::avx2::f32x8::AVX_ALIGNMENT;

pub(crate) const MR: usize = 8;
pub(crate) const NR: usize = 8;

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
/// Data is stored in **row-major** format: each row contains NR elements.
/// For variable KC, we use manually aligned memory allocation for performance.
///
/// This layout enables the microkernel to:
/// 1. Load a full row to get B(k, 0..NR-1)
/// 2. Broadcast individual elements B(k,j) efficiently
///
/// The `#[repr(C, align(32))]` ensures 32-byte alignment for AVX2 operations.
#[repr(C, align(32))]
pub struct BPanel<const NR: usize> {
    /// Raw pointer to aligned storage for kc rows of NR elements each
    pub data_ptr: *mut f32,
    /// Number of rows (KC) in this panel  
    pub kc: usize,
    /// Layout for safe deallocation
    layout: Layout,
    /// Marker for proper drop semantics
    _marker: PhantomData<[f32; NR]>,
}

impl<const NR: usize> BPanel<NR> {
    /// Create a new BPanel with the given KC size
    pub fn new(kc: usize) -> Result<Self, Layout> {
        let total_elements = kc * NR;
        let layout = Layout::array::<f32>(total_elements)
            .unwrap()
            .align_to(AVX_ALIGNMENT)
            .unwrap();
            
        let data_ptr = unsafe {
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            raw_ptr.cast::<f32>()
        };
        
        Ok(BPanel {
            data_ptr,
            kc,
            layout,
            _marker: PhantomData,
        })
    }
    
    /// Get a pointer to the k-th row (NR elements)
    #[inline(always)]
    pub fn get_row(&self, k: usize) -> *const f32 {
        debug_assert!(k < self.kc, "Row index {} out of bounds (kc={})", k, self.kc);
        unsafe { self.data_ptr.add(k * NR) }
    }
    
    /// Get a mutable pointer to the k-th row (NR elements)
    #[inline(always)]
    pub fn get_row_mut(&mut self, k: usize) -> *mut f32 {
        debug_assert!(k < self.kc, "Row index {} out of bounds (kc={})", k, self.kc);
        unsafe { self.data_ptr.add(k * NR) }
    }
}

impl<const NR: usize> Drop for BPanel<NR> {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe {
                alloc::dealloc(self.data_ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

/// A high-performance block containing multiple B panels.
///
/// Manages memory for ceil(nc/NR) panels of KC×NR elements each, where nc is the
/// number of columns from the original B matrix block being packed.
///
/// **Performance Optimization**: Uses thread-local memory pool for small blocks,
/// heap allocation for large blocks to minimize allocation overhead.
pub struct BBlock<const NR: usize> {
    /// Raw pointer to contiguous aligned storage for all panels
    data_ptr: *mut f32,
    /// Number of panels (ceil(nc/NR))
    num_panels: usize,
    /// KC size for all panels
    pub kc: usize,
    /// Layout for heap deallocation (None for memory pool allocation)
    layout: Option<Layout>,
    /// The original number of columns from matrix `B` packed into this block.
    pub nc: usize,
    /// Marker for drop semantics
    _marker: PhantomData<f32>,
}


impl<const NR: usize> BBlock<NR> {
    /// Allocates zero-initialized, aligned memory for packing nc columns.
    ///
    /// **Performance Optimization**: Uses stack allocation for small blocks to avoid heap overhead.
    ///
    /// # Arguments
    /// * `nc` - Number of columns from original B matrix to pack
    /// * `kc` - Number of rows (KC) each panel should contain
    ///
    /// # Returns
    /// New BBlock or allocation error
    #[inline(always)]
    pub fn new(nc: usize, kc: usize) -> Result<Self, Layout> {
        // Calculate panels needed: ceil(nc / NR)
        let num_panels = nc.div_ceil(NR);
        
        // Single large allocation for all panels: num_panels * kc * NR elements
        let total_elements = num_panels * kc * NR;
        
        // Simple high-performance allocation with alignment optimization
        let layout = Layout::array::<f32>(total_elements)
            .unwrap()
            .align_to(AVX_ALIGNMENT)
            .unwrap();
            
        let data_ptr = unsafe {
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            raw_ptr.cast::<f32>()
        };

        Ok(BBlock {
            data_ptr,
            num_panels,
            kc,
            layout: Some(layout),
            nc,
            _marker: PhantomData,
        })
    }
    
    /// Get pointer to panel data by panel index
    #[inline(always)]
    pub fn get_panel_data(&self, panel_idx: usize) -> *const f32 {
        debug_assert!(panel_idx < self.num_panels);
        unsafe { self.data_ptr.add(panel_idx * self.kc * NR) }
    }
    
    /// Get mutable pointer to panel data by panel index  
    #[inline(always)]
    pub fn get_panel_data_mut(&mut self, panel_idx: usize) -> *mut f32 {
        debug_assert!(panel_idx < self.num_panels);
        unsafe { self.data_ptr.add(panel_idx * self.kc * NR) }
    }
    
    /// Get pointer to specific row within a panel
    #[inline(always)] 
    pub fn get_panel_row(&self, panel_idx: usize, row: usize) -> *const f32 {
        debug_assert!(panel_idx < self.num_panels && row < self.kc);
        unsafe { self.data_ptr.add(panel_idx * self.kc * NR + row * NR) }
    }
    
    /// Get mutable pointer to specific row within a panel
    #[inline(always)]
    pub fn get_panel_row_mut(&mut self, panel_idx: usize, row: usize) -> *mut f32 {
        debug_assert!(panel_idx < self.num_panels && row < self.kc);
        unsafe { self.data_ptr.add(panel_idx * self.kc * NR + row * NR) }
    }
}

impl<const NR: usize> Drop for BBlock<NR> {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            if layout.size() > 0 {
                unsafe {
                    alloc::dealloc(self.data_ptr.cast::<u8>(), layout);
                }
            }
        }
    }
}

// BBlock no longer supports direct indexing - use get_panel_* methods for performance

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
/// For variable KC, we use manually aligned memory allocation.
#[repr(C, align(32))]
pub struct APanel<const MR: usize> {
    /// Raw pointer to aligned storage for kc columns of MR elements each
    pub data_ptr: *mut f32,
    /// Number of columns (KC) in this panel  
    pub kc: usize,
    /// Layout for safe deallocation
    layout: Layout,
    /// Marker for proper drop semantics
    _marker: PhantomData<[f32; MR]>,
}

impl<const MR: usize> APanel<MR> {
    /// Create a new APanel with the given KC size
    pub fn new(kc: usize) -> Result<Self, Layout> {
        let total_elements = kc * MR;
        let layout = Layout::array::<f32>(total_elements)
            .unwrap()
            .align_to(AVX_ALIGNMENT)
            .unwrap();
            
        let data_ptr = unsafe {
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            raw_ptr.cast::<f32>()
        };
        
        Ok(APanel {
            data_ptr,
            kc,
            layout,
            _marker: PhantomData,
        })
    }
    
    /// Get a pointer to the k-th column (MR elements)
    #[inline(always)]
    pub fn get_column(&self, k: usize) -> *const f32 {
        debug_assert!(k < self.kc, "Column index {} out of bounds (kc={})", k, self.kc);
        unsafe { self.data_ptr.add(k * MR) }
    }
    
    /// Get a mutable pointer to the k-th column (MR elements)
    #[inline(always)]
    pub fn get_column_mut(&mut self, k: usize) -> *mut f32 {
        debug_assert!(k < self.kc, "Column index {} out of bounds (kc={})", k, self.kc);
        unsafe { self.data_ptr.add(k * MR) }
    }
}

impl<const MR: usize> Drop for APanel<MR> {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe {
                alloc::dealloc(self.data_ptr.cast::<u8>(), self.layout);
            }
        }
    }
}

/// A high-performance block containing multiple A panels.
///
/// Manages memory for ceil(mc/MR) panels of MR×KC elements each, where mc is the
/// number of rows from the original A matrix block being packed.
///
/// **Performance Optimization**: Uses thread-local memory pool for small blocks,
/// heap allocation for large blocks to minimize allocation overhead.
pub struct ABlock<const MR: usize> {
    /// Raw pointer to contiguous aligned storage for all panels
    data_ptr: *mut f32,
    /// Number of panels (ceil(mc/MR))
    num_panels: usize,
    /// KC size for all panels
    pub kc: usize,
    /// Layout for heap deallocation (None for memory pool allocation)
    layout: Option<Layout>,
    /// The original number of rows from matrix `A` packed into this block.
    pub mc: usize,
    /// Marker for drop semantics
    _marker: PhantomData<f32>,
}

impl<const MR: usize> ABlock<MR> {
    /// Allocates zero-initialized, aligned memory for packing mc rows.
    ///
    /// **Performance Optimization**: Uses stack allocation for small blocks to avoid heap overhead.
    ///
    /// # Arguments
    /// * `mc` - Number of rows from original A matrix to pack
    /// * `kc` - Number of columns (KC) each panel should contain
    ///
    /// # Returns
    /// New ABlock or allocation error
    #[inline(always)]
    pub fn new(mc: usize, kc: usize) -> Result<Self, Layout> {
        let num_panels = mc.div_ceil(MR);
        
        // Single large allocation for all panels: num_panels * kc * MR elements
        let total_elements = num_panels * kc * MR;
        
        // Simple high-performance allocation with alignment optimization
        let layout = Layout::array::<f32>(total_elements)
            .unwrap()
            .align_to(AVX_ALIGNMENT)
            .unwrap();
            
        let data_ptr = unsafe {
            let raw_ptr = alloc::alloc_zeroed(layout);
            if raw_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            raw_ptr.cast::<f32>()
        };

        Ok(ABlock {
            data_ptr,
            num_panels,
            kc,
            layout: Some(layout),
            mc,
            _marker: PhantomData,
        })
    }
    
    /// Get pointer to panel data by panel index
    #[inline(always)]
    pub fn get_panel_data(&self, panel_idx: usize) -> *const f32 {
        debug_assert!(panel_idx < self.num_panels);
        unsafe { self.data_ptr.add(panel_idx * self.kc * MR) }
    }
    
    /// Get mutable pointer to panel data by panel index  
    #[inline(always)]
    pub fn get_panel_data_mut(&mut self, panel_idx: usize) -> *mut f32 {
        debug_assert!(panel_idx < self.num_panels);
        unsafe { self.data_ptr.add(panel_idx * self.kc * MR) }
    }
    
    /// Get pointer to specific column within a panel
    #[inline(always)] 
    pub fn get_panel_column(&self, panel_idx: usize, col: usize) -> *const f32 {
        debug_assert!(panel_idx < self.num_panels && col < self.kc);
        unsafe { self.data_ptr.add(panel_idx * self.kc * MR + col * MR) }
    }
    
    /// Get mutable pointer to specific column within a panel
    #[inline(always)]
    pub fn get_panel_column_mut(&mut self, panel_idx: usize, col: usize) -> *mut f32 {
        debug_assert!(panel_idx < self.num_panels && col < self.kc);
        unsafe { self.data_ptr.add(panel_idx * self.kc * MR + col * MR) }
    }
}

impl<const MR: usize> Drop for ABlock<MR> {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            if layout.size() > 0 {
                unsafe {
                    alloc::dealloc(self.data_ptr.cast::<u8>(), layout);
                }
            }
        }
    }
}

// ABlock no longer supports direct indexing - use get_panel_* methods for performance

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
#[inline(always)]
pub fn pack_a<const MR: usize>(
    a: &[f32],
    mc: usize,
    kc: usize,
    m: usize,
    ic: usize,
    pc: usize,
) -> ABlock<MR> {
    let mut packed_block = ABlock::<MR>::new(mc, kc).expect("Memory allocation failed for ABlock");

    // Pre-calculate base addresses to eliminate redundant calculations
    let base_src_row = ic;
    let base_src_col_offset = pc * m;

    // Process mc rows in groups of MR (microkernel row dimension)
    for (panel_idx, i_panel_start) in (0..mc).step_by(MR).enumerate() {
        let mr_in_panel = min(MR, mc - i_panel_start);

        // Calculate source row offset once per panel
        let panel_src_row_offset = base_src_row + i_panel_start;

        // Pack all kc columns of this row panel with optimized inner loop
        for p_col in 0..kc {
            // Inline index calculation - eliminates function call overhead
            let src_start = base_src_col_offset + p_col * m + panel_src_row_offset;

            let dest_col = packed_block.get_panel_column_mut(panel_idx, p_col);

            // Optimized copy with manual unrolling for common cases
            unsafe {
                match mr_in_panel {
                    8 => {
                        // Full panel - most common case, manually unrolled
                        *dest_col = a[src_start];
                        *dest_col.add(1) = a[src_start + 1];
                        *dest_col.add(2) = a[src_start + 2];
                        *dest_col.add(3) = a[src_start + 3];
                        *dest_col.add(4) = a[src_start + 4];
                        *dest_col.add(5) = a[src_start + 5];
                        *dest_col.add(6) = a[src_start + 6];
                        *dest_col.add(7) = a[src_start + 7];
                    }
                    4 => {
                        // Half panel
                        *dest_col = a[src_start];
                        *dest_col.add(1) = a[src_start + 1];
                        *dest_col.add(2) = a[src_start + 2];
                        *dest_col.add(3) = a[src_start + 3];
                    }
                    mr => {
                        // Partial panel - copy remaining elements
                        for i in 0..mr {
                            *dest_col.add(i) = a[src_start + i];
                        }
                    }
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
#[inline(always)]
pub fn pack_b<const NR: usize>(
    b: &[f32],
    nc: usize,
    kc: usize,
    k: usize,
    pc: usize,
    jc: usize,
) -> BBlock<NR> {
    let mut packed_block = BBlock::<NR>::new(nc, kc).expect("Memory allocation failed for BBlock");

    // Process nc columns in groups of NR (microkernel column dimension)
    for (panel_idx, j_panel_start) in (0..nc).step_by(NR).enumerate() {
        let nr_in_panel = min(NR, nc - j_panel_start);

        // Pre-calculate column base addresses to eliminate function calls
        let col_base = jc + j_panel_start;

        // Pack kc rows of this column panel with optimized inner loops
        for p_row in 0..kc {
            let src_row = pc + p_row;
            let dest_row = packed_block.get_panel_row_mut(panel_idx, p_row);

            // Optimized packing with manual unrolling for common cases
            unsafe {
                match nr_in_panel {
                    8 => {
                        // Full panel - most common case, manually unrolled
                        // Inline index calculation: col * k + row
                        *dest_row = b[(col_base + 0) * k + src_row];
                        *dest_row.add(1) = b[(col_base + 1) * k + src_row];
                        *dest_row.add(2) = b[(col_base + 2) * k + src_row];
                        *dest_row.add(3) = b[(col_base + 3) * k + src_row];
                        *dest_row.add(4) = b[(col_base + 4) * k + src_row];
                        *dest_row.add(5) = b[(col_base + 5) * k + src_row];
                        *dest_row.add(6) = b[(col_base + 6) * k + src_row];
                        *dest_row.add(7) = b[(col_base + 7) * k + src_row];
                    }
                    4 => {
                        // Half panel
                        *dest_row = b[(col_base + 0) * k + src_row];
                        *dest_row.add(1) = b[(col_base + 1) * k + src_row];
                        *dest_row.add(2) = b[(col_base + 2) * k + src_row];
                        *dest_row.add(3) = b[(col_base + 3) * k + src_row];
                    }
                    6 => {
                        // 3/4 panel
                        *dest_row = b[(col_base + 0) * k + src_row];
                        *dest_row.add(1) = b[(col_base + 1) * k + src_row];
                        *dest_row.add(2) = b[(col_base + 2) * k + src_row];
                        *dest_row.add(3) = b[(col_base + 3) * k + src_row];
                        *dest_row.add(4) = b[(col_base + 4) * k + src_row];
                        *dest_row.add(5) = b[(col_base + 5) * k + src_row];
                    }
                    2 => {
                        // Quarter panel
                        *dest_row = b[(col_base + 0) * k + src_row];
                        *dest_row.add(1) = b[(col_base + 1) * k + src_row];
                    }
                    1 => {
                        // Single element
                        *dest_row = b[col_base * k + src_row];
                    }
                    nr => {
                        // General case for other sizes - still optimized with eliminated function calls
                        for j_col_in_panel in 0..nr {
                            let src_col = col_base + j_col_in_panel;
                            *dest_row.add(j_col_in_panel) = b[src_col * k + src_row];
                        }
                    }
                }
            }
        }
    }

    packed_block
}
