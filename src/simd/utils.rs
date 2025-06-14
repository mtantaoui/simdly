use std::alloc::{alloc, handle_alloc_error, Layout};

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
