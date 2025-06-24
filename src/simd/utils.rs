use std::alloc::{alloc, alloc_zeroed, handle_alloc_error, Layout};

/// Allocates a 32-byte aligned `Vec<f32>` with uninitialized contents.
///
/// # Safety
///
/// The caller must ensure that the elements of the returned vector are
/// initialized before being read. Reading from uninitialized memory is
/// undefined behavior.
#[inline(always)]
pub fn alloc_uninit_f32_vec(len: usize, align: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }

    let layout =
        Layout::from_size_align(len * std::mem::size_of::<f32>(), align).expect("Invalid layout");

    let ptr = unsafe { alloc(layout) as *mut f32 };

    if ptr.is_null() {
        handle_alloc_error(layout);
    }

    // SAFETY: The pointer is non-null and the layout is valid for `len` elements.
    // The capacity is set to `len`, so no re-allocation will occur until it's grown.
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

/// Allocates a `Vec<f32>` with the specified alignment, and all elements initialized to zero.
///
/// # Arguments
///
/// * `len` - The number of `f32` elements the vector should hold.
/// * `align` - The desired byte alignment for the allocated memory.
///
/// # Returns
///
/// A `Vec<f32>` of the specified length, with its underlying buffer allocated
/// to the given alignment and all elements initialized to `0.0f32`.
///
/// # Panics
///
/// * If `layout` cannot be created (e.g., `align` is not a power of two, or `len * size_of::<f32>()` overflows).
/// * If memory allocation fails, this function will trigger the global allocation error handler,
///   which typically panics.
#[inline(always)]
pub fn alloc_zeroed_f32_vec(len: usize, align: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new(); // Or Vec::with_capacity(0) if you prefer
    }

    // Calculate the total size in bytes and create the layout
    // `Layout::array<f32>(len)` can also handle potential overflows for total size.
    // However, to explicitly set alignment, `from_size_align` is more direct.
    let size_bytes = match len.checked_mul(std::mem::size_of::<f32>()) {
        Some(s) => s,
        None => panic!("Total size calculation overflowed for Vec<f32> of len {len}"),
    };

    let layout = match Layout::from_size_align(size_bytes, align) {
        Ok(l) => l,
        Err(_) => panic!("Failed to create Layout with size {size_bytes} and alignment {align}"),
    };

    // Allocate memory and initialize it to zero
    let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };

    // Check if allocation failed
    if ptr.is_null() {
        // `handle_alloc_error` will typically panic.
        // It's part of the standard library's allocation error handling mechanism.
        handle_alloc_error(layout);
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
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}
