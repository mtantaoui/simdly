use std::alloc::{alloc, alloc_zeroed, handle_alloc_error, Layout};

#[allow(dead_code)]
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

    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

#[allow(dead_code)]
#[inline(always)]
pub fn alloc_zeroed_f32_vec(len: usize, align: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }

    let layout =
        Layout::from_size_align(len * std::mem::size_of::<f32>(), align).expect("Invalid layout");

    let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };

    if ptr.is_null() {
        handle_alloc_error(layout);
    }

    unsafe { Vec::from_raw_parts(ptr, len, len) }
}
