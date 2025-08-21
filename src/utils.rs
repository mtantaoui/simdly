use std::alloc::{alloc, alloc_zeroed, Layout};

#[cfg(target_os = "windows")]
use std::alloc::dealloc;

#[cfg(target_os = "windows")]
use std::mem;

#[cfg(target_os = "windows")]
use std::ops::{Deref, DerefMut};

#[cfg(target_os = "windows")]
use std::ptr::NonNull;

use std::alloc::handle_alloc_error;

/// A Windows-safe container for aligned memory allocation.
///
/// This container provides aligned memory allocation using the system allocator
/// on Windows platforms. It ensures proper cleanup through RAII and provides
/// safe conversion to `Vec<T>` by copying data to avoid allocator mismatches.
///
/// # Platform Availability
///
/// This type is only available on Windows. On Linux/Mac, use the faster
/// direct allocation functions like `alloc_uninit_vec<T>()`.
///
/// # Memory Safety
///
/// - Uses `std::alloc::alloc()` for allocation and `std::alloc::dealloc()` for cleanup
/// - Conversion to `Vec<T>` creates a copy to avoid Windows heap corruption
/// - All operations are memory-safe despite internal `unsafe` code
///
/// # Performance Characteristics
///
/// - **Allocation**: Single aligned allocation with specified alignment
/// - **Conversion**: O(n) copy operation to create `Vec<T>`
/// - **SIMD**: Maintains alignment for optimal SIMD performance during computation
///
/// # Example
///
/// ```rust,ignore
/// // AlignedVec is used internally by the library for Windows compatibility
/// // Users should use the high-level SIMD traits which handle memory allocation automatically
/// # #[cfg(target_os = "windows")]
/// # {
/// use simdly::simd::SimdMath;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0];
/// let result = data.cos(); // Library handles aligned allocation internally
/// # }
/// ```
#[cfg(target_os = "windows")]
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
}

#[cfg(target_os = "windows")]
impl<T> AlignedVec<T> {
    /// Creates a new `AlignedVec<T>` with uninitialized memory.
    ///
    /// Allocates memory with the specified alignment using the system allocator.
    /// The memory is not initialized, so accessing it before writing is undefined behavior.
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements to allocate
    /// * `align` - Required memory alignment in bytes (must be power of 2)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `align` is not a power of 2
    /// - `align` is less than the natural alignment of `T`
    /// - Memory allocation fails
    ///
    /// # Safety
    ///
    /// The returned memory is uninitialized. You must initialize all elements
    /// before reading them to avoid undefined behavior.
    pub fn new_uninit(len: usize, align: usize) -> Self {
        assert!(align.is_power_of_two());
        assert!(align >= mem::align_of::<T>());

        let layout = Layout::from_size_align(len * mem::size_of::<T>(), align)
            .expect("Failed to create layout");

        let ptr = unsafe { alloc(layout) };

        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None => std::alloc::handle_alloc_error(layout),
        };

        AlignedVec { ptr, len, layout }
    }

    /// Creates a new `AlignedVec<T>` with zero-initialized memory.
    ///
    /// Allocates memory with the specified alignment and initializes all bytes to zero.
    /// This is safer than `new_uninit` but slightly slower due to initialization.
    ///
    /// # Arguments
    ///
    /// * `len` - Number of elements to allocate
    /// * `align` - Required memory alignment in bytes (must be power of 2)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `align` is not a power of 2
    /// - `align` is less than the natural alignment of `T`
    /// - Memory allocation fails
    pub fn new_zeroed(len: usize, align: usize) -> Self {
        assert!(align.is_power_of_two());
        assert!(align >= mem::align_of::<T>());

        let layout = Layout::from_size_align(len * mem::size_of::<T>(), align)
            .expect("Failed to create layout");

        let ptr = unsafe { alloc_zeroed(layout) };

        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None => std::alloc::handle_alloc_error(layout),
        };

        AlignedVec { ptr, len, layout }
    }
}

/// Automatic memory deallocation for `AlignedVec<T>`.
///
/// This implementation ensures that aligned memory is properly freed
/// using the same allocator that was used for allocation (`std::alloc::dealloc`).
#[cfg(target_os = "windows")]
impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        // Deallocate memory using the same allocator used for allocation
        if self.layout.size() > 0 {
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
            }
        }
    }
}

/// Enables slice-like access to `AlignedVec<T>`.
///
/// This allows `AlignedVec<T>` to be used transparently as a `&[T]` slice,
/// providing access to all slice methods while maintaining alignment guarantees.
#[cfg(target_os = "windows")]
impl<T> Deref for AlignedVec<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

/// Enables mutable slice-like access to `AlignedVec<T>`.
///
/// This allows `AlignedVec<T>` to be used transparently as a `&mut [T]` slice,
/// enabling SIMD operations that require aligned memory access.
#[cfg(target_os = "windows")]
impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

/// Windows-safe conversion from `AlignedVec<T>` to `Vec<T>`.
///
/// This implementation creates a copy of the aligned data to avoid allocator
/// mismatch issues on Windows. While this adds O(n) copy overhead, it prevents
/// heap corruption that would occur if we tried to transfer ownership directly.
///
/// # Performance Impact
///
/// - **Copy Cost**: O(n) where n is the number of elements
/// - **Memory**: Temporarily uses 2x memory during conversion
/// - **Safety**: Guaranteed to work on all Windows configurations
///
/// # Why Copy is Necessary on Windows
///
/// - `AlignedVec<T>` uses `std::alloc::alloc()` (system allocator)
/// - `Vec<T>` uses the global allocator (may be different on Windows)
/// - Direct ownership transfer can cause `STATUS_HEAP_CORRUPTION`
/// - Copying ensures each type uses its own consistent allocator
#[cfg(target_os = "windows")]
impl<T: Clone> From<AlignedVec<T>> for Vec<T> {
    fn from(aligned_vec: AlignedVec<T>) -> Self {
        // Create a copy using Vec's allocator to avoid Windows heap corruption
        let data: Vec<T> =
            unsafe { std::slice::from_raw_parts(aligned_vec.ptr.as_ptr(), aligned_vec.len) }
                .to_vec();

        // AlignedVec will be automatically dropped, cleaning up its allocation
        data
    }
}

/// Fast zero-copy aligned vector allocation for Linux/Mac platforms.
///
/// This function provides maximum performance aligned allocation by using direct
/// system allocation with zero-copy ownership transfer. It avoids the copy overhead
/// of the Windows-safe approach while maintaining compatibility with Vec's allocator.
///
/// # Platform Availability
///
/// **Linux/Mac ONLY** - This function is not available on Windows due to
/// allocator compatibility issues. Use `AlignedVec<T>` on Windows instead.
///
/// # Memory Safety
///
/// - **Linux/Mac**: Memory safe - Vec and std::alloc use compatible allocators
/// - **Windows**: Would cause `STATUS_HEAP_CORRUPTION` - function not available
/// - **Uninitialized**: Returns uninitialized memory - caller must initialize before use
/// - **Type Safety**: Caller responsible for proper initialization of `T`
///
/// # Arguments
///
/// * `len` - Number of elements to allocate
/// * `align` - Required alignment in bytes (must be power of 2)
///
/// # Returns
///
/// A `Vec<T>` with aligned but uninitialized memory. Caller must initialize
/// all elements before reading to avoid undefined behavior.
///
/// # Panics
///
/// Panics if:
/// - `align` is not a power of 2
/// - Memory allocation fails
///
/// # Performance
///
/// - **Zero copy**: Direct ownership transfer, no copying overhead
/// - **Fast allocation**: Uses system allocator directly for optimal speed
/// - **Memory**: Single allocation, no temporary double memory usage
#[cfg(not(target_os = "windows"))]
pub fn alloc_uninit_vec<T>(len: usize, align: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }

    let layout = Layout::from_size_align(len * std::mem::size_of::<T>(), align)
        .expect("Invalid layout for aligned allocation");

    let ptr = unsafe { alloc(layout) as *mut T };

    if ptr.is_null() {
        handle_alloc_error(layout);
    }

    // SAFETY:
    // - ptr is non-null and properly aligned
    // - len elements of size T were allocated
    // - Memory is uninitialized - caller must initialize before use
    // - On Linux/Mac, Vec uses the same allocator as std::alloc::alloc
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

/// Fast zero-copy aligned vector allocation with zero initialization for Linux/Mac.
///
/// **Note**: This function is currently unused but maintained for future SIMD operations
/// that may require zero-initialized memory (e.g., reduction operations, accumulation buffers).
///
/// Similar to `alloc_uninit_vec<T>` but initializes all memory to zero bytes.
/// This is safer than the uninitialized version but slightly slower due to
/// the initialization overhead.
///
/// # Platform Availability
///
/// **Linux/Mac ONLY** - Not available on Windows to prevent heap corruption.
///
/// # Memory Safety
///
/// - **Initialization**: All bytes are set to zero before returning
/// - **Platform**: Safe on Linux/Mac, would corrupt heap on Windows
/// - **Type Safety**: Zero bytes may not be valid for all types `T`
///
/// # Arguments
///
/// * `len` - Number of elements to allocate
/// * `align` - Required alignment in bytes (must be power of 2)
///
/// # Returns
///
/// A `Vec<T>` with zero-initialized, aligned memory.
///
/// # Performance vs Alternatives
///
/// - **vs `alloc_uninit_vec`**: Slightly slower due to zero initialization
/// - **vs Windows `AlignedVec`**: Much faster due to zero-copy allocation
/// - **vs `Vec::new() + resize`**: Faster due to aligned allocation
#[cfg(not(target_os = "windows"))]
#[allow(dead_code)]
pub(crate) fn alloc_zeroed_vec<T>(len: usize, align: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }

    let layout = Layout::from_size_align(len * std::mem::size_of::<T>(), align)
        .expect("Invalid layout for aligned allocation");

    let ptr = unsafe { alloc_zeroed(layout) as *mut T };

    if ptr.is_null() {
        handle_alloc_error(layout);
    }

    // SAFETY: Same as alloc_uninit_vec, but memory is zeroed
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}
