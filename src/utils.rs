use std::alloc::{alloc, alloc_zeroed, Layout};

use crate::error::{allocation_error, layout_error, SimdlyError};

/// Allocates an aligned `Vec<f32>` with uninitialized contents.
///
/// # Arguments
///
/// * `len` - The number of f32 elements to allocate
/// * `align` - The desired byte alignment (must be a power of two)
///
/// # Returns
///
/// A `Result` containing the allocated vector or an error if allocation fails.
///
/// # Safety
///
/// The caller must ensure that the elements of the returned vector are
/// initialized before being read. Reading from uninitialized memory is
/// undefined behavior.
///
/// # Errors
///
/// Returns an error if:
/// - Layout creation fails (invalid size/alignment combination)
/// - Memory allocation fails
#[allow(dead_code)]
#[inline(always)]
pub fn alloc_uninit_f32_vec(len: usize, align: usize) -> Result<Vec<f32>, SimdlyError> {
    if len == 0 {
        return Ok(Vec::new());
    }

    // Check for potential overflow
    let size_bytes = match len.checked_mul(std::mem::size_of::<f32>()) {
        Some(size) => size,
        None => {
            return Err(layout_error(
                len * std::mem::size_of::<f32>(),
                align,
                "Size calculation overflow",
            ))
        }
    };

    let layout = Layout::from_size_align(size_bytes, align).map_err(|_| {
        layout_error(
            size_bytes,
            align,
            "Invalid layout: alignment must be a power of two and size must not overflow",
        )
    })?;

    let ptr = unsafe { alloc(layout) as *mut f32 };

    if ptr.is_null() {
        return Err(allocation_error(
            size_bytes,
            align,
            "Memory allocation failed",
        ));
    }

    // SAFETY: The pointer is non-null and the layout is valid for `len` elements.
    // The capacity is set to `len`, so no re-allocation will occur until it's grown.
    Ok(unsafe { Vec::from_raw_parts(ptr, len, len) })
}

/// Allocates a `Vec<f32>` with the specified alignment, and all elements initialized to zero.
///
/// # Arguments
///
/// * `len` - The number of `f32` elements the vector should hold.
/// * `align` - The desired byte alignment for the allocated memory (must be a power of two).
///
/// # Returns
///
/// A `Result` containing a `Vec<f32>` of the specified length with its underlying
/// buffer allocated with the given alignment and all elements initialized to `0.0f32`,
/// or an error if allocation fails.
///
/// # Errors
///
/// Returns an error if:
/// - Size calculation overflows
/// - Layout creation fails (invalid alignment or size)
/// - Memory allocation fails
#[allow(dead_code)]
#[inline(always)]
pub fn alloc_zeroed_f32_vec(len: usize, align: usize) -> Result<Vec<f32>, SimdlyError> {
    if len == 0 {
        return Ok(Vec::new());
    }

    // Calculate the total size in bytes and check for overflow
    let size_bytes = match len.checked_mul(std::mem::size_of::<f32>()) {
        Some(s) => s,
        None => {
            return Err(layout_error(
                len * std::mem::size_of::<f32>(), // This might overflow, but it's for error reporting
                align,
                format!("Size calculation overflowed for Vec<f32> of len {}", len),
            ));
        }
    };

    let layout = Layout::from_size_align(size_bytes, align).map_err(|_| {
        layout_error(
            size_bytes,
            align,
            format!(
                "Failed to create Layout with size {} and alignment {}",
                size_bytes, align
            ),
        )
    })?;

    // Allocate memory and initialize it to zero
    let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };

    // Check if allocation failed
    if ptr.is_null() {
        return Err(allocation_error(
            size_bytes,
            align,
            "Zero-initialized memory allocation failed",
        ));
    }

    // SAFETY:
    // 1. `ptr` is non-null (checked above).
    // 2. `ptr` was allocated using `layout` with the global allocator (via `alloc_zeroed`).
    // 3. `layout` is the `Layout` that `ptr` was allocated with.
    // 4. `len * size_of::<f32>()` (which is `layout.size()`) is equal to
    //    `capacity * size_of::<f32>()` because we set `capacity` to `len`.
    //    So, `len * size_of::<f32>() <= capacity * size_of::<f32>()` holds.
    // 5. `capacity` (which is `len`) is the capacity `ptr` was allocated with.
    // 6. The memory from `ptr` up to `ptr.add(len)` is properly initialized (to zero by `alloc_zeroed`).
    Ok(unsafe { Vec::from_raw_parts(ptr, len, len) })
}
