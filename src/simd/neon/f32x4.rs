#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::simd::traits::SimdVec;

pub const NEON_ALIGNMENT: usize = 16;

pub const LANE_COUNT: usize = 4;

/// A SIMD vector of 4 32-bit floating point values
#[derive(Copy, Clone, Debug)]
pub struct F32x4 {
    size: usize,
    elements: float32x4_t,
}

impl SimdVec<f32> for F32x4 {
    /// The number of lanes in the vector
    #[inline(always)]
    fn new(slice: &[f32]) -> Self {
        assert!(!slice.is_empty(), "Size can't be empty (size zero)");

        // If the slice length is less than LANE_COUNT, load a partial vector
        // Otherwise, load a full vector
        // This ensures that the vector is always created with the correct number of elements
        // and that the remaining elements are filled with zeros if necessary.
        // This is done to avoid unnecessary branching and ensure all elements are set correctly.
        // The `match` statement is used to handle the different cases based on the slice length.
        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), slice.len()) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    /// Creates a new vector with all elements set to the same value.
    #[inline(always)]
    unsafe fn splat(value: f32) -> Self {
        Self {
            elements: unsafe { vdupq_n_f32(value) },
            size: LANE_COUNT,
        }
    }

    /// Checks if the pointer is aligned.
    #[inline(always)]
    fn is_aligned(_ptr: *const f32) -> bool {
        unreachable!()
    }

    #[inline(always)]
    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        // Asserts that the pointer is not null and the size is exactly 4 elements.
        // If the pointer is null or the size is not 4, it will panic with an error message.
        assert!(!ptr.is_null(), "Pointer must not be null");
        assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");

        Self {
            elements: unsafe { vld1q_f32(ptr) },
            size,
        }
    }

    /// Loads a vector from an aligned pointer, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn load_aligned(_ptr: *const f32, _size: usize) -> Self {
        unreachable!()
    }

    /// Loads a vector from an unaligned pointer, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn load_unaligned(_ptr: *const f32, _size: usize) -> Self {
        unreachable!()
    }

    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        assert!(
            size < LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );

        // to avoid unnecessary branching and ensure all elements are set correctly
        assert!(!ptr.is_null(), "Pointer must not be null");

        let elements = match size {
            1 => {
                let v = vdupq_n_f32(0.0);
                vsetq_lane_f32(*ptr.add(0), v, 0)
            }
            2 => {
                let mut v = vdupq_n_f32(0.0);
                v = vsetq_lane_f32(*ptr.add(0), v, 0);
                vsetq_lane_f32(*ptr.add(1), v, 1)
            }
            3 => {
                let mut v = vdupq_n_f32(0.0);
                v = vsetq_lane_f32(*ptr.add(0), v, 0);
                v = vsetq_lane_f32(*ptr.add(1), v, 1);
                vsetq_lane_f32(*ptr.add(2), v, 2)
            }
            _ => unreachable!("Size must be < {}", LANE_COUNT),
        };

        Self { elements, size }
    }

    unsafe fn store_in_vec(&self) -> Vec<f32> {
        todo!()
    }

    unsafe fn store_in_vec_partial(&self) -> Vec<f32> {
        todo!()
    }

    #[inline(always)]
    unsafe fn store_at(&self, ptr: *mut f32) {
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );
        assert!(!ptr.is_null(), "Pointer must not be null");

        vst1q_f32(ptr, self.elements);
    }

    #[inline(always)]
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );
        assert!(!ptr.is_null(), "Pointer must not be null");

        match self.size {
            3 => {
                let low = vget_low_f32(self.elements); // extract [0, 1]
                vst1_f32(ptr, low); // store [0, 1]

                let third = vgetq_lane_f32(self.elements, 2); // extract element 2
                *ptr.add(2) = third; // store element 2
            }
            2 => {
                let low = vget_low_f32(self.elements);
                vst1_f32(ptr, low);
            }
            1 => {
                let first = vgetq_lane_f32(self.elements, 0);
                *ptr = first;
            }
            _ => {
                unreachable!("Size must be < {}", LANE_COUNT)
            }
        }
    }

    fn to_vec(self) -> Vec<f32> {
        todo!()
    }

    unsafe fn eq_elements(&self, rhs: Self) -> Self {
        todo!()
    }

    unsafe fn lt_elements(&self, rhs: Self) -> Self {
        todo!()
    }

    unsafe fn le_elements(&self, rhs: Self) -> Self {
        todo!()
    }

    unsafe fn gt_elements(&self, rhs: Self) -> Self {
        todo!()
    }

    unsafe fn ge_elements(&self, rhs: Self) -> Self {
        todo!()
    }
}

/// Implementing the `Add` and `AddAssign` traits for F32x8
/// This allows for using the `+` operator and `+=` operator with F32x8 vectors.    
impl Add for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Use the add function to perform element-wise addition
        Self {
            size: self.size,
            elements: unsafe { vaddq_f32(self.elements, rhs.elements) },
        }
    }
}
