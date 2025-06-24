#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::simd::{
    avx2::{
        abs::float32::_mm256_abs_ps, acos::float32::_mm256_acos_ps, asin::float32::_mm256_asin_ps,
        cos::float32::_mm256_cos_ps,
    },
    traits::SimdVec,
};

pub const AVX_ALIGNMENT: usize = 32;
pub const LANE_COUNT: usize = 8;

/// A vector of 8 `f32` elements using AVX2 intrinsics.
/// This struct provides methods for creating, loading, and manipulating vectors of `f32` elements
/// using AVX2 instructions. It supports operations like loading from aligned or unaligned memory,
/// loading partial vectors, and performing element-wise comparisons.
/// The vector is represented as an AVX `__m256` type, which contains 8 single-precision floating-point numbers.
/// The size of the vector is fixed at 8 elements, and it requires the input slice to have a length
/// that is either equal to or less than 8. If the length is less than 8, the remaining elements are filled with zeros.
/// The struct implements the `SimdVec` trait, which defines the necessary methods for SIMD operations.     
#[derive(Copy, Clone, Debug)]
pub struct F32x8 {
    pub(crate) size: usize,
    pub(crate) elements: __m256,
}

impl SimdVec<f32> for F32x8 {
    /// The number of lanes in the vector
    #[inline(always)]
    fn new(slice: &[f32]) -> Self {
        debug_assert!(!slice.is_empty(), "Size can't be empty (size zero)");

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
            elements: unsafe { _mm256_set1_ps(value) },
            size: LANE_COUNT,
        }
    }

    /// Checks if the pointer is aligned to 32 bytes.
    /// This is necessary for AVX operations to ensure optimal performance.
    #[inline(always)]
    fn is_aligned(ptr: *const f32) -> bool {
        let ptr = ptr as usize;
        ptr % core::mem::align_of::<__m256>() == 0
    }

    /// Loads a vector from a pointer, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        // Asserts that the pointer is not null and the size is exactly 8 elements.
        // If the pointer is null or the size is not 8, it will panic with an error message.
        debug_assert!(!ptr.is_null(), "Pointer must not be null");
        debug_assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");

        if Self::is_aligned(ptr) {
            unsafe { Self::load_aligned(ptr, size) }
        } else {
            unsafe { Self::load_unaligned(ptr, size) }
        }
    }

    /// Loads a vector from an aligned pointer, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn load_aligned(ptr: *const f32, size: usize) -> Self {
        Self {
            elements: unsafe { _mm256_load_ps(ptr) },
            size,
        }
    }

    /// Loads a vector from an unaligned pointer, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const f32, size: usize) -> Self {
        Self {
            elements: unsafe { _mm256_loadu_ps(ptr) },
            size,
        }
    }

    /// Loads a partial vector from a pointer, filling the rest with zeros.
    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        debug_assert!(
            size < LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );

        // to avoid unnecessary branching and ensure all elements are set correctly
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        // setting elements directly using match statement
        let elements = match size {
            1 => unsafe { _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *ptr.add(0)) },
            2 => unsafe { _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, *ptr.add(1), *ptr.add(0)) },
            3 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            4 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            5 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    0.0,
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            6 => unsafe {
                _mm256_set_ps(
                    0.0,
                    0.0,
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            7 => unsafe {
                _mm256_set_ps(
                    0.0,
                    *ptr.add(6),
                    *ptr.add(5),
                    *ptr.add(4),
                    *ptr.add(3),
                    *ptr.add(2),
                    *ptr.add(1),
                    *ptr.add(0),
                )
            },
            _ => unreachable!("Size must be < {}", LANE_COUNT),
        };

        Self { elements, size }
    }

    /// Stores the vector elements into a `Vec<f32>`, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn store_in_vec(&self) -> Vec<f32> {
        debug_assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {}", LANE_COUNT)
        );

        let mut vec = Vec::with_capacity(LANE_COUNT);

        unsafe {
            _mm256_storeu_ps(vec.as_mut_ptr(), self.elements);
            vec.set_len(LANE_COUNT);
        }

        vec
    }

    /// Stores the vector elements into a `Vec<f32>`, filling only the first `size` elements.
    #[inline(always)]
    unsafe fn store_in_vec_partial(&self) -> Vec<f32> {
        match self.size {
            1..LANE_COUNT => unsafe { self.store_in_vec().into_iter().take(self.size).collect() },
            _ => {
                let msg = "Size must be < LANE_COUNT";
                panic!("{}", msg);
            }
        }
    }

    /// Stores the vector elements into a memory location pointed to by `ptr`, ensuring the size is exactly 8 elements.
    /// This method is unsafe because it assumes that the pointer is valid and aligned.
    /// It uses `_mm256_stream_ps` for aligned storage or `_mm256_storeu_ps` for unaligned storage.
    #[inline(always)]
    unsafe fn store_at(&self, ptr: *mut f32) {
        debug_assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        // Check if the pointer is aligned to 32 bytes
        // If it is aligned, use `_mm256_stream_ps` for better performance
        // If it is not aligned, use `_mm256_storeu_ps` for unaligned storage
        match Self::is_aligned(ptr) {
            #[cfg(not(miri))]
            true => unsafe { _mm256_stream_ps(ptr, self.elements) },
            #[cfg(miri)]
            true => unsafe { _mm256_store_ps(ptr, self.elements) },
            false => unsafe { _mm256_storeu_ps(ptr, self.elements) },
        }
    }

    /// Stores the vector elements into a memory location pointed to by `ptr`, filling only the first `size` elements.
    /// This method is unsafe because it assumes that the pointer is valid and aligned.     
    #[inline(always)]
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        debug_assert!(
            self.size < LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mask: __m256i = match self.size {
            1 => unsafe { _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0) },
            2 => unsafe { _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0) },
            3 => unsafe { _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0) },
            4 => unsafe { _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0) },
            5 => unsafe { _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0) },
            6 => unsafe { _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0) },
            7 => unsafe { _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0) },
            _ => unreachable!("Size must be < LANE_COUNT"),
        };

        unsafe { _mm256_maskstore_ps(ptr, mask, self.elements) };
    }

    /// Converts the vector to a `Vec<f32>`, ensuring the size is less than or equal to 8 elements.
    #[inline(always)]
    fn to_vec(self) -> Vec<f32> {
        debug_assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );

        if self.size == LANE_COUNT {
            unsafe { self.store_in_vec() }
        } else {
            unsafe { self.store_in_vec_partial() }
        }
    }

    /// Compares two vectors elementwise for equality.    
    #[inline(always)]
    unsafe fn eq_elements(&self, rhs: Self) -> Self {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a == b elementwise
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_EQ_OQ) }; // Result as float mask

        Self {
            elements,
            size: self.size,
        }
    }

    /// Compares two vectors elementwise for less than, less than or equal to, greater than, and greater than or equal to.  
    #[inline(always)]
    unsafe fn lt_elements(&self, rhs: Self) -> Self {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a<b elementwise
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_LT_OQ) }; // Result as float mask

        Self {
            elements,
            size: self.size,
        }
    }

    /// Compares two vectors elementwise for less than or equal to.
    #[inline(always)]
    unsafe fn le_elements(&self, rhs: Self) -> Self {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a<=b elementwise
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_LE_OQ) }; // Result as float mask

        Self {
            elements,
            size: self.size,
        }
    }

    /// Compares two vectors elementwise for greater than.
    #[inline(always)]
    unsafe fn gt_elements(&self, rhs: Self) -> Self {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a>b elementwise
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_GT_OQ) }; // Result as float mask

        Self {
            elements,
            size: self.size,
        }
    }

    /// Compares two vectors elementwise for greater than or equal to.
    #[inline(always)]
    unsafe fn ge_elements(&self, rhs: Self) -> Self {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a>=b elementwise
        let elements = unsafe { _mm256_cmp_ps(self.elements, rhs.elements, _CMP_GE_OQ) }; // Result as float mask

        Self {
            elements,
            size: self.size,
        }
    }

    /// Computes the absolute value of each element in the vector.
    #[inline(always)]
    unsafe fn abs(&self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm256_abs_ps(self.elements) },
        }
    }

    #[inline(always)]
    unsafe fn acos(&self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm256_acos_ps(self.elements) },
        }
    }

    /// Computes the absolute value of each element in the vector.
    #[inline(always)]
    unsafe fn asin(&self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm256_asin_ps(self.elements) },
        }
    }

    /// Computes the cosine of each element in the vector.
    #[inline(always)]
    unsafe fn cos(&self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm256_cos_ps(self.elements) },
        }
    }

    #[inline(always)]
    unsafe fn fmadd(&self, a: Self, b: Self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm256_fmadd_ps(a.elements, b.elements, self.elements) },
        }
    }
}

/// Computes the remainder of two `__m256` vectors elementwise.
#[inline(always)]
unsafe fn _rem(lhs: __m256, rhs: __m256) -> __m256 {
    let div = unsafe { _mm256_div_ps(lhs, rhs) };
    let floor = unsafe { _mm256_floor_ps(div) };
    let prod = unsafe { _mm256_mul_ps(floor, rhs) };

    unsafe { _mm256_sub_ps(lhs, prod) }
}

// /// Compares two `__m256` vectors for equality.
#[inline(always)]
unsafe fn _lt(lhs: __m256, rhs: __m256, size: usize) -> bool {
    let lt: __m256 = unsafe { _mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ) };
    let lt_mask = unsafe { _mm256_movemask_ps(lt) };

    let mask = (1 << size) - 1; // Create a mask with the first `size` bits set to 1

    lt_mask == mask
}

/// Compares two `__m256` vectors for less than or equal to.
#[inline(always)]
unsafe fn _le(lhs: __m256, rhs: __m256) -> bool {
    let le: __m256 = unsafe { _mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ) };
    let le_mask = unsafe { _mm256_movemask_ps(le) };

    le_mask == 0xFF
}

/// Compares two `__m256` vectors for greater than.
#[inline(always)]
unsafe fn _gt(lhs: __m256, rhs: __m256, size: usize) -> bool {
    let gt: __m256 = _mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ);
    let gt_mask = _mm256_movemask_ps(gt);

    let mask = (1 << size) - 1; // Create a mask with the first `size` bits set to 1

    gt_mask == mask
}

/// Compares two `__m256` vectors for greater than or equal to.
#[inline(always)]
unsafe fn _ge(lhs: __m256, rhs: __m256) -> bool {
    let ge: __m256 = _mm256_cmp_ps(lhs, rhs, _CMP_GE_OQ);
    let ge_mask = _mm256_movemask_ps(ge);

    ge_mask == 0xFF
}

/// Implementing the `Add` and `AddAssign` traits for F32x8
/// This allows for using the `+` operator and `+=` operator with F32x8 vectors.    
impl Add for F32x8 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Use the add function to perform element-wise addition
        Self {
            size: self.size,
            elements: unsafe { _mm256_add_ps(self.elements, rhs.elements) },
        }
    }
}

impl AddAssign for F32x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Implementing the `Sub` and `SubAssign` traits for F32x8
/// This allows for using the `-` operator and `-=` operator with F32x8 vectors.
impl Sub for F32x8 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Use the sub function to perform element-wise subtraction
        Self {
            size: self.size,
            elements: unsafe { _mm256_sub_ps(self.elements, rhs.elements) },
        }
    }
}

impl SubAssign for F32x8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// Implementing the `Mul` and `MulAssign` traits for F32x8
/// This allows for using the `*` operator and `*=` operator with F32x8 vectors.
impl Mul for F32x8 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        Self {
            size: self.size,
            elements: unsafe { _mm256_mul_ps(self.elements, rhs.elements) },
        }
    }
}

impl MulAssign for F32x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// Implementing the `Div` and `DivAssign` traits for F32x8
/// This allows for using the `/` operator and `/=` operator with F32x8 vectors.
impl Div for F32x8 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        Self {
            size: self.size,
            elements: unsafe { _mm256_div_ps(self.elements, rhs.elements) },
        }
    }
}

impl DivAssign for F32x8 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for F32x8 {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        Self {
            size: self.size,
            elements: unsafe { _rem(self.elements, rhs.elements) },
        }
    }
}

impl RemAssign for F32x8 {
    #[inline(always)]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Eq for F32x8 {}

impl PartialEq for F32x8 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        debug_assert!(
            self.size == other.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            other.size
        );

        unsafe {
            // Compare lane-by-lane
            let cmp = _mm256_cmp_ps(self.elements, other.elements, _CMP_EQ_OQ);

            // Move the mask to integer form
            let mask = _mm256_movemask_ps(cmp);

            mask == 0xFF
        }
    }
}

impl PartialOrd for F32x8 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        debug_assert!(
            self.size == other.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            other.size
        );

        // Pre-calculate the boolean conditions using the correct helper methods.
        let all_eq = self.eq(other);
        let all_lt = self.lt(other);
        let all_gt = self.gt(other);

        // Use a match statement on the tuple of conditions to determine the ordering.
        match (all_eq, all_lt, all_gt) {
            // Case 1: All elements are equal. This is the highest priority check.
            (true, _, _) => Some(std::cmp::Ordering::Equal),

            // Case 2: Not all equal, but all are less than.
            (false, true, false) => Some(std::cmp::Ordering::Less),

            // Case 3: Not all equal or less than, but all are greater than.
            (false, false, true) => Some(std::cmp::Ordering::Greater),

            // All other combinations imply a mixed ordering.
            _ => None,
        }
    }

    #[inline(always)]
    fn lt(&self, other: &Self) -> bool {
        unsafe { _lt(self.elements, other.elements, self.size) }
    }

    #[inline(always)]
    fn le(&self, other: &Self) -> bool {
        unsafe { _le(self.elements, other.elements) }
    }

    #[inline(always)]
    fn gt(&self, other: &Self) -> bool {
        unsafe { _gt(self.elements, other.elements, self.size) }
    }

    #[inline(always)]
    fn ge(&self, other: &Self) -> bool {
        unsafe { _ge(self.elements, other.elements) }
    }
}

impl BitAnd for F32x8 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_and_ps(self.elements, rhs.elements) },
        }
    }
}

impl BitAndAssign for F32x8 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self & rhs;
    }
}

impl BitOr for F32x8 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        F32x8 {
            size: self.size,
            elements: unsafe { _mm256_or_ps(self.elements, rhs.elements) },
        }
    }
}

impl BitOrAssign for F32x8 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        debug_assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self | rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*; // Imports F32x8, LANE_COUNT, AVX_ALIGNMENT, etc.
    use crate::simd::traits::SimdVec; // To call methods from the trait implementation
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::ptr;

    // Helper for creating aligned data arrays
    #[repr(align(32))] // AVX_ALIGNMENT is 32
    struct AlignedData<const N: usize>([f32; N]);

    impl<const N: usize> AlignedData<N> {
        fn new(val: [f32; N]) -> Self {
            Self(val)
        }
    }

    // Helper to get all 8 f32 elements from F32x8.elements
    // This bypasses F32x8::size and F32x8::to_vec() logic.
    fn get_all_elements(v: F32x8) -> [f32; LANE_COUNT] {
        let mut arr = [0.0f32; LANE_COUNT];
        unsafe {
            // Unaligned store is fine for testing purposes
            _mm256_storeu_ps(arr.as_mut_ptr(), v.elements);
        }
        arr
    }

    // Helper for precise f32 slice comparison (bitwise for NaNs)
    fn assert_f32_slice_eq_bitwise(a: &[f32], b: &[f32]) {
        assert_eq!(
            a.len(),
            b.len(),
            "Slice lengths differ (left: {}, right: {})",
            a.len(),
            b.len()
        );
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "Elements at index {} differ: left={}({:08x}), right={}({:08x})",
                i,
                a[i],
                a[i].to_bits(),
                b[i],
                b[i].to_bits()
            );
        }
    }

    // Helper for f32 slice comparison with epsilon
    fn assert_f32_slice_eq_epsilon(a: &[f32], b: &[f32], epsilon: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "Slice lengths differ (left: {}, right: {})",
            a.len(),
            b.len()
        );
        for i in 0..a.len() {
            if a[i].is_nan() && b[i].is_nan() {
                // Allow different NaN bit patterns if just checking for NaN property
                // For specific NaN patterns (like masks), use assert_f32_slice_eq_bitwise
                continue;
            }
            debug_assert!(
                (a[i] - b[i]).abs() <= epsilon,
                "Elements at index {} differ: left={}, right={}, diff={}",
                i,
                a[i],
                b[i],
                (a[i] - b[i]).abs()
            );
        }
    }

    const TRUE_MASK_F32: f32 = f32::from_bits(0xFFFFFFFFu32); // Expected -NaN for true in masks
    const FALSE_MASK_F32: f32 = 0.0f32;

    mod simd_vec_impl {

        use super::*;

        #[test]
        fn test_new_full_slice() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = F32x8::new(&data);
            assert_eq!(v.size, LANE_COUNT);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &data);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }

        #[test]
        fn test_new_larger_slice() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            let v = F32x8::new(&data);
            assert_eq!(v.size, LANE_COUNT);
            let expected = &data[0..LANE_COUNT];
            assert_f32_slice_eq_bitwise(&get_all_elements(v), expected);
            assert_f32_slice_eq_bitwise(&v.to_vec(), expected);
        }

        #[test]
        fn test_new_partial_slice() {
            let data = [1.0, 2.0, 3.0];
            let v = F32x8::new(&data);
            assert_eq!(v.size, data.len());

            let mut expected_raw = [0.0f32; LANE_COUNT];
            expected_raw[0..data.len()].copy_from_slice(&data);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected_raw);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }

        #[test]
        fn test_new_partial_slice_len_1() {
            let data = [1.0];
            let v = F32x8::new(&data);
            assert_eq!(v.size, 1);
            let mut expected_raw = [0.0f32; LANE_COUNT];
            expected_raw[0] = 1.0;
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected_raw);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }

        #[test]
        fn test_new_partial_slice_len_7() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            let v = F32x8::new(&data);
            assert_eq!(v.size, 7);
            let mut expected_raw = [0.0f32; LANE_COUNT];
            expected_raw[0..7].copy_from_slice(&data);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected_raw);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }

        #[test]
        #[should_panic(expected = "Size can't be empty (size zero)")]
        fn test_new_empty_slice_panics() {
            F32x8::new(&[]);
        }

        #[test]
        fn test_splat() {
            let val = std::f32::consts::PI;
            let v = unsafe { F32x8::splat(val) };
            assert_eq!(v.size, LANE_COUNT);
            let expected = [val; LANE_COUNT];
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &expected);
        }

        #[allow(clippy::needless_range_loop)]
        #[test]
        fn test_splat_nan() {
            let val = f32::NAN;
            let v = unsafe { F32x8::splat(val) };
            assert_eq!(v.size, LANE_COUNT);
            let elements = get_all_elements(v);
            for i in 0..LANE_COUNT {
                debug_assert!(elements[i].is_nan());
            }
        }

        #[test]
        fn test_is_aligned() {
            let aligned_arr = AlignedData::new([0.0f32; LANE_COUNT]);
            let unaligned_arr = [0.0f32; LANE_COUNT + 1]; // Make it > LANE_COUNT to get an unaligned slice easily

            debug_assert!(
                F32x8::is_aligned(aligned_arr.0.as_ptr()),
                "Aligned pointer reported as unaligned."
            );

            // Ensure we test an actually unaligned pointer if possible.
            // Most basic allocations might be aligned to 8 or 16 bytes.
            // We need to force non-32-byte alignment.
            let ptr_usize = unaligned_arr.as_ptr() as usize;
            if ptr_usize % AVX_ALIGNMENT != 0 {
                debug_assert!(
                    !F32x8::is_aligned(unaligned_arr.as_ptr()),
                    "Unaligned pointer reported as aligned."
                );
            } else {
                // If base pointer is 32-byte aligned, try offset pointer
                if (ptr_usize + std::mem::size_of::<f32>()) % AVX_ALIGNMENT != 0 {
                    debug_assert!(
                        !F32x8::is_aligned(unaligned_arr.as_ptr().wrapping_add(1)),
                        "Offset unaligned pointer reported as aligned."
                    );
                } else {
                    // This case is unlikely but possible if base is aligned and offset keeps it aligned.
                    // To be robust, one might allocate a larger buffer and find an unaligned spot.
                    // For now, this is a best-effort check for unaligned.
                    eprintln!("Warning: Could not reliably test F32x8::is_aligned for unaligned case. Pointer was unexpectedly aligned.");
                }
            }
        }

        #[test]
        fn test_load_aligned() {
            let data = AlignedData::new([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]);
            let v = unsafe { F32x8::load(data.0.as_ptr(), LANE_COUNT) };
            assert_eq!(v.size, LANE_COUNT);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &data.0);
        }

        #[test]
        fn test_load_unaligned() {
            // Create data that's likely unaligned (slice from a Vec often works)
            let data_vec: Vec<f32> = vec![0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];
            // Take a slice starting at index 1 to increase chance of unalignment wrt 32 bytes
            let data_slice = &data_vec[1..LANE_COUNT + 1];
            assert_eq!(data_slice.len(), LANE_COUNT);

            let v = unsafe { F32x8::load(data_slice.as_ptr(), LANE_COUNT) };
            assert_eq!(v.size, LANE_COUNT);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), data_slice);
        }

        #[test]
        #[should_panic(expected = "Pointer must not be null")]
        fn test_load_null_ptr_panics() {
            unsafe {
                F32x8::load(ptr::null(), LANE_COUNT);
            }
        }

        #[test]
        #[should_panic(expected = "Size must be == 8")]
        fn test_load_incorrect_size_panics() {
            let data = [0.0f32; LANE_COUNT];
            unsafe {
                F32x8::load(data.as_ptr(), LANE_COUNT - 1);
            }
        }

        #[test]
        fn test_load_partial_various_sizes() {
            let data_full = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            for len in 1..LANE_COUNT {
                let data_slice = &data_full[0..len];
                let v = unsafe { F32x8::load_partial(data_slice.as_ptr(), len) };
                assert_eq!(v.size, len, "Size mismatch for len={}", len);

                let mut expected_raw = [0.0f32; LANE_COUNT];
                expected_raw[0..len].copy_from_slice(data_slice);

                assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected_raw);
                assert_f32_slice_eq_bitwise(&v.to_vec(), data_slice);
            }
        }

        #[test]
        #[should_panic(expected = "Size must be < 8")]
        fn test_load_partial_size_too_large_panics() {
            let data = [0.0f32; LANE_COUNT];
            unsafe {
                F32x8::load_partial(data.as_ptr(), LANE_COUNT);
            }
        }

        #[test]
        #[should_panic(expected = "Size must be < 8")]
        fn test_load_partial_size_zero_panics_due_to_unreachable() {
            // The assertion is "Size must be < 8". 0 < 8 is true.
            // So it passes the assert, then hits `_ => unreachable!("Size must be < {}", LANE_COUNT)`.
            // The unreachable message is identical to the assertion message.
            let data = [0.0f32; 1]; // Dummy pointer, content doesn't matter for size 0
            unsafe {
                F32x8::load_partial(data.as_ptr(), 0);
            }
        }

        #[test]
        #[should_panic(expected = "Pointer must not be null")]
        fn test_load_partial_null_ptr_panics() {
            unsafe {
                F32x8::load_partial(ptr::null(), 1);
            }
        }

        #[test]
        fn test_store_in_vec_full() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = F32x8::new(&data);
            let stored_vec = unsafe { v.store_in_vec() };
            assert_eq!(stored_vec.len(), LANE_COUNT);
            assert_f32_slice_eq_bitwise(&stored_vec, &data);
        }

        #[test]
        fn test_store_in_vec_partial_source_size() {
            // store_in_vec itself always stores LANE_COUNT elements from __m256
            // The F32x8.size doesn't change what store_in_vec does,
            // but it influences to_vec.
            let data_partial = [1.0, 2.0, 3.0];
            let v = F32x8::new(&data_partial); // size = 3

            let mut expected_full_from_m256 = [0.0f32; LANE_COUNT];
            expected_full_from_m256[0..3].copy_from_slice(&data_partial);

            let stored_vec = unsafe { v.store_in_vec() };
            assert_eq!(stored_vec.len(), LANE_COUNT);
            assert_f32_slice_eq_bitwise(&stored_vec, &expected_full_from_m256);
        }

        #[test]
        #[should_panic(expected = "Size must be <= 8")]
        fn test_store_in_vec_invalid_size_panics() {
            let mut v = unsafe { F32x8::splat(1.0) };
            v.size = LANE_COUNT + 1; // Manually set invalid size
            let _ = unsafe { v.store_in_vec() };
        }

        #[test]
        fn test_store_in_vec_partial_method() {
            let data_full = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            for len in 1..LANE_COUNT {
                let data_slice = &data_full[0..len];
                let v = F32x8::new(data_slice); // size will be `len`
                assert_eq!(v.size, len);

                let stored_vec = unsafe { v.store_in_vec_partial() };
                assert_eq!(stored_vec.len(), len);
                assert_f32_slice_eq_bitwise(&stored_vec, data_slice);
            }
        }

        #[test]
        #[should_panic(expected = "Size must be < LANE_COUNT")]
        fn test_store_in_vec_partial_method_size_lane_count_panics() {
            let data = [1.0; LANE_COUNT];
            let v = F32x8::new(&data); // size = LANE_COUNT
            assert_eq!(v.size, LANE_COUNT);
            unsafe { v.store_in_vec_partial() };
        }

        #[test]
        fn test_store_at_aligned_and_unaligned() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v = F32x8::new(&data);

            // Test aligned store
            let mut aligned_storage: [f32; 8] = [0.0; 8];
            unsafe {
                v.store_at(aligned_storage.as_mut_ptr());
            }
            assert_f32_slice_eq_bitwise(&aligned_storage, &data);

            // Test unaligned store
            let mut unaligned_storage_vec: Vec<f32> = vec![0.0f32; LANE_COUNT + 1];
            let ptr_unaligned: *mut f32 = unsafe { unaligned_storage_vec.as_mut_ptr().add(1) };

            // Perform the store operation using the raw pointer.
            unsafe {
                v.store_at(ptr_unaligned);
            }

            // This slice is created *after* the raw pointer write in `v.store_at()`.
            let slice_for_assertion: &[f32] =
                unsafe { std::slice::from_raw_parts(ptr_unaligned as *const f32, LANE_COUNT) };
            // Use this newly created slice for the assertion.
            assert_f32_slice_eq_bitwise(slice_for_assertion, &data); // <-- Use the correct slice here
        }

        #[test]
        #[should_panic(expected = "Pointer must not be null")]
        fn test_store_at_null_ptr_panics() {
            let v = unsafe { F32x8::splat(1.0) };
            unsafe {
                v.store_at(ptr::null_mut());
            }
        }

        #[allow(clippy::manual_memcpy)]
        #[test]
        fn test_store_at_partial() {
            let data_full = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            for len in 1..LANE_COUNT {
                let data_slice_source = &data_full[0..len];
                let v = F32x8::new(data_slice_source); // Creates F32x8 with .size = len
                                                       // and elements [data_slice_source..., 0.0s...]
                assert_eq!(v.size, len);

                let mut storage = [f32::NAN; LANE_COUNT]; // Fill with NANs to check masking
                unsafe {
                    v.store_at_partial(storage.as_mut_ptr());
                }

                // Expected: first `len` elements from v, rest are original NANs
                let mut expected_storage = [f32::NAN; LANE_COUNT];
                for i in 0..len {
                    expected_storage[i] = data_slice_source[i];
                }
                assert_f32_slice_eq_bitwise(&storage, &expected_storage);
            }
        }

        #[test]
        #[should_panic(expected = "Size must be < 8")]
        fn test_store_at_partial_size_lane_count_panics() {
            let data = [1.0; LANE_COUNT];
            let v = F32x8::new(&data); // .size = LANE_COUNT
            let mut storage = [0.0; LANE_COUNT];
            unsafe {
                v.store_at_partial(storage.as_mut_ptr());
            }
        }

        #[test]
        fn test_to_vec() {
            // Full vector
            let data_full = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let v_full = F32x8::new(&data_full);
            assert_f32_slice_eq_bitwise(&v_full.to_vec(), &data_full);

            // Partial vector
            let data_partial = [1.0, 2.0, 3.0];
            let v_partial = F32x8::new(&data_partial);
            assert_f32_slice_eq_bitwise(&v_partial.to_vec(), &data_partial);
        }

        #[test]
        fn test_comparison_elements() {
            let d1 = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
            let d2 = [1.0, 3.0, 2.0, 4.0, 0.0, 3.0, 2.0, 5.0];
            let v1 = F32x8::new(&d1);
            let v2 = F32x8::new(&d2);

            let t = TRUE_MASK_F32;
            let f = FALSE_MASK_F32;

            // eq_elements
            let eq_res_v = unsafe { v1.eq_elements(v2) };
            assert_eq!(eq_res_v.size, LANE_COUNT);
            let expected_eq = [t, f, f, t, f, f, f, f];
            assert_f32_slice_eq_bitwise(&eq_res_v.to_vec(), &expected_eq);

            // lt_elements
            let lt_res_v = unsafe { v1.lt_elements(v2) };
            let expected_lt = [f, t, f, f, f, t, f, t];
            assert_f32_slice_eq_bitwise(&lt_res_v.to_vec(), &expected_lt);

            // le_elements
            let le_res_v = unsafe { v1.le_elements(v2) };
            let expected_le = [t, t, f, t, f, t, f, t];
            assert_f32_slice_eq_bitwise(&le_res_v.to_vec(), &expected_le);

            // gt_elements
            let gt_res_v = unsafe { v1.gt_elements(v2) };
            let expected_gt = [f, f, t, f, t, f, t, f];
            assert_f32_slice_eq_bitwise(&gt_res_v.to_vec(), &expected_gt);

            // ge_elements
            let ge_res_v = unsafe { v1.ge_elements(v2) };
            let expected_ge = [t, f, t, t, t, f, t, f];
            assert_f32_slice_eq_bitwise(&ge_res_v.to_vec(), &expected_ge);
        }

        #[test]
        #[should_panic(expected = "Operands must have the same size")]
        fn test_comparison_elements_panic_on_diff_size() {
            let v1 = F32x8::new(&[1.0; LANE_COUNT]);
            let mut v2 = F32x8::new(&[2.0; LANE_COUNT]);
            v2.size = LANE_COUNT - 1; // Manually set different size
            let _ = unsafe { v1.eq_elements(v2) };
        }

        #[test]
        fn test_cos() {
            use std::f32::consts::{FRAC_PI_2, PI};
            let inputs = [
                0.0,
                FRAC_PI_2,
                PI,
                3.0 * FRAC_PI_2,
                2.0 * PI,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ];
            // Pad to LANE_COUNT
            let mut data_in = [0.0f32; LANE_COUNT];
            data_in[0..inputs.len()].copy_from_slice(&inputs);

            let v_in = F32x8::new(&data_in);
            let v_out = unsafe { v_in.cos() };

            let mut expected_outputs_approx = [0.0f32; LANE_COUNT];
            expected_outputs_approx[0] = 1.0; // cos(0)
            expected_outputs_approx[1] = 0.0; // cos(PI/2)
            expected_outputs_approx[2] = -1.0; // cos(PI)
            expected_outputs_approx[3] = 0.0; // cos(3PI/2)
            expected_outputs_approx[4] = 1.0; // cos(2PI)
                                              // For NaN and Inf, cos behavior can be platform/implementation specific.
                                              // _mm256_cos_ps might produce NaN for NaN/Inf.
                                              // Let's assume NaN for these specific inputs.
            expected_outputs_approx[5] = f32::NAN; // cos(NaN)
            expected_outputs_approx[6] = f32::NAN; // cos(Inf)
            expected_outputs_approx[7] = f32::NAN; // cos(-Inf)

            let out_vec = v_out.to_vec();

            // For cos, use epsilon comparison due to precision of approximations
            // The first 5 values are standard angles
            assert_f32_slice_eq_epsilon(&out_vec[0..5], &expected_outputs_approx[0..5], 1.8e-7);
            // For NaN/Inf, check if they are NaN
            for i in 5..inputs.len() {
                debug_assert!(
                    out_vec[i].is_nan(),
                    "cos({}) expected NaN, got {}",
                    data_in[i],
                    out_vec[i]
                );
            }
        }

        #[allow(clippy::needless_range_loop)]
        #[test]
        fn test_abs() {
            // Define inputs that are good test cases for `abs`
            let inputs = [
                0.0,
                -0.0,
                1.0,
                -1.0,
                -123.456,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ];

            // Pad to LANE_COUNT. The rest will be 0.0, which is also a valid test case.
            let mut data_in = [0.0f32; LANE_COUNT];
            data_in[0..inputs.len()].copy_from_slice(&inputs);

            let v_in = F32x8::new(&data_in);
            let v_out = unsafe { v_in.abs() };

            let mut expected_outputs = [0.0f32; LANE_COUNT];
            expected_outputs[0] = 0.0; // abs(0.0)
            expected_outputs[1] = 0.0; // abs(-0.0)
            expected_outputs[2] = 1.0; // abs(1.0)
            expected_outputs[3] = 1.0; // abs(-1.0)
            expected_outputs[4] = 123.456; // abs(-123.456)
            expected_outputs[5] = f32::NAN; // abs(NaN)
            expected_outputs[6] = f32::INFINITY; // abs(Inf)
            expected_outputs[7] = f32::INFINITY; // abs(-Inf)

            let out_vec = v_out.to_vec();

            // The `abs` operation is exact, so we can check for equality.
            // We must handle NaN separately because NaN != NaN.
            for i in 0..inputs.len() {
                let received = out_vec[i];
                let expected = expected_outputs[i];

                if expected.is_nan() {
                    assert!(
                        received.is_nan(),
                        "abs({}) expected NaN, got {}",
                        data_in[i],
                        received
                    );
                } else {
                    assert_eq!(
                        received, expected,
                        "abs({}) expected {}, got {}",
                        data_in[i], expected, received
                    );
                }
            }

            // Check that the padded elements were also processed correctly (abs(0.0) -> 0.0)
            for i in inputs.len()..LANE_COUNT {
                assert_eq!(
                    out_vec[i], 0.0,
                    "padded value at index {} should be 0.0 after abs, but got {}",
                    i, out_vec[i]
                );
            }
        }

        #[test]
        fn test_asin() {
            // A __m256 vector holds 8 f32 values.
            const LANE_COUNT: usize = 8;

            // Key input values for asin. Domain is [-1, 1].
            let inputs = [
                0.0f32,
                1.0f32,
                -1.0f32,
                0.5f32,
                1.0f32 / std::f32::consts::SQRT_2,
                // Out of domain / special values
                1.1f32,
                f32::NAN,
                f32::INFINITY,
            ];

            // Pad the input array to the full lane count.
            let mut data_in = [0.0f32; LANE_COUNT];
            data_in[0..inputs.len()].copy_from_slice(&inputs);

            // Load, compute, and store results.
            let v_in = F32x8::new(&data_in);
            let v_out = unsafe { v_in.asin() };
            let results = v_out.to_vec();

            // Define expected outputs for valid inputs using std::f32::asin.
            let mut expected_outputs_approx = [0.0f32; LANE_COUNT];
            expected_outputs_approx[0] = data_in[0].asin(); // asin(0.0)  -> 0.0
            expected_outputs_approx[1] = data_in[1].asin(); // asin(1.0)  -> PI/2
            expected_outputs_approx[2] = data_in[2].asin(); // asin(-1.0) -> -PI/2
            expected_outputs_approx[3] = data_in[3].asin(); // asin(0.5)  -> PI/6
            expected_outputs_approx[4] = data_in[4].asin(); // asin(1/sqrt(2)) -> PI/4

            // Compare the valid results with an epsilon.
            assert_f32_slice_eq_epsilon(&results[0..5], &expected_outputs_approx[0..5], 3e-7);

            // For out-of-domain and special values, expect NaN.
            for i in 5..inputs.len() {
                assert!(
                    results[i].is_nan(),
                    "asin({}) expected NaN, got {}",
                    data_in[i],
                    results[i]
                );
            }
        }

        #[test]
        fn test_acos() {
            // A __m256 vector holds 8 f32 values.
            const LANE_COUNT: usize = 8;

            // Key input values for acos. Domain is [-1, 1].
            let inputs = [
                // Standard values
                0.0f32,                            // acos(0) -> PI/2
                1.0f32,                            // acos(1) -> 0
                -1.0f32,                           // acos(-1) -> PI
                0.5f32,                            // acos(0.5) -> PI/3
                1.0f32 / std::f32::consts::SQRT_2, // acos(1/sqrt(2)) -> PI/4
                // Out of domain / special values
                1.1f32,
                f32::NAN,
                f32::NEG_INFINITY,
            ];

            // Pad the input array to the full lane count.
            let mut data_in = [0.0f32; LANE_COUNT];
            data_in[0..inputs.len()].copy_from_slice(&inputs);

            // Load, compute, and store results.
            let v_in = F32x8::new(&data_in);
            let v_out = unsafe { v_in.acos() };
            let results = v_out.to_vec();
            // Define expected outputs for valid inputs using std::f32::acos as ground truth.
            let mut expected_outputs_approx = [0.0f32; LANE_COUNT];
            expected_outputs_approx[0] = data_in[0].acos();
            expected_outputs_approx[1] = data_in[1].acos();
            expected_outputs_approx[2] = data_in[2].acos();
            expected_outputs_approx[3] = data_in[3].acos();
            expected_outputs_approx[4] = data_in[4].acos();

            // Compare the valid results with an epsilon. An epsilon of 1e-7 is reasonable for f32.
            assert_f32_slice_eq_epsilon(&results[0..5], &expected_outputs_approx[0..5], 3e-7);

            // For out-of-domain and special values, expect NaN.
            for i in 5..inputs.len() {
                assert!(
                    results[i].is_nan(),
                    "acos({}) expected NaN, got {}",
                    data_in[i],
                    results[i]
                );
            }
        }
    }

    mod operator_overloads {
        use super::*;

        fn setup_vecs() -> (F32x8, F32x8, F32x8) {
            let d1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let d2 = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
            let d3 = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]; // For division/modulo
            (F32x8::new(&d1), F32x8::new(&d2), F32x8::new(&d3))
        }

        #[test]
        fn test_add_sub_mul_div_rem() {
            let (v1, v2, v_div) = setup_vecs();

            // Add
            let add_res = v1 + v2;
            let expected_add = [9.0; LANE_COUNT];
            assert_f32_slice_eq_bitwise(&add_res.to_vec(), &expected_add);

            // Sub
            let sub_res = v1 - v_div;
            let expected_sub = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            assert_f32_slice_eq_bitwise(&sub_res.to_vec(), &expected_sub);

            // Mul
            let mul_res = v1 * v_div;
            let expected_mul = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
            assert_f32_slice_eq_bitwise(&mul_res.to_vec(), &expected_mul);

            // Div
            let div_res = v1 / v_div;
            let expected_div = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
            assert_f32_slice_eq_bitwise(&div_res.to_vec(), &expected_div);

            // Rem
            // v1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            // v_div: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
            let rem_res = v1 % v_div;
            let expected_rem = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
            assert_f32_slice_eq_bitwise(&rem_res.to_vec(), &expected_rem);
        }

        #[test]
        fn test_assign_ops() {
            let (_v1_orig, v2, _v_div_orig) = setup_vecs();
            let (v1_add, _, _) = setup_vecs(); // v1 for add test
            let (mut v1_sub, _, v_div_sub) = setup_vecs();
            let (mut v1_mul, _, v_div_mul) = setup_vecs();
            let (mut v1_div, _, v_div_div) = setup_vecs();
            let (mut v1_rem, _, v_div_rem) = setup_vecs();

            let mut v1 = v1_add; // Use a copy for each assign op
            v1 += v2;
            let expected_add = [9.0; LANE_COUNT];
            assert_f32_slice_eq_bitwise(&v1.to_vec(), &expected_add);

            v1_sub -= v_div_sub;
            let expected_sub = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            assert_f32_slice_eq_bitwise(&v1_sub.to_vec(), &expected_sub);

            v1_mul *= v_div_mul;
            let expected_mul = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
            assert_f32_slice_eq_bitwise(&v1_mul.to_vec(), &expected_mul);

            v1_div /= v_div_div;
            let expected_div = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
            assert_f32_slice_eq_bitwise(&v1_div.to_vec(), &expected_div);

            v1_rem %= v_div_rem;
            let expected_rem = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
            assert_f32_slice_eq_bitwise(&v1_rem.to_vec(), &expected_rem);
        }

        #[test]
        fn test_div_by_zero() {
            let data_ones = [1.0; LANE_COUNT];
            let data_zeros = [0.0; LANE_COUNT];
            let v_ones = F32x8::new(&data_ones);
            let v_zeros = F32x8::new(&data_zeros);

            let res = v_ones / v_zeros;
            let elements = res.to_vec();
            for &x in &elements {
                debug_assert!(
                    x.is_infinite() && x.is_sign_positive(),
                    "Expected positive infinity from 1.0/0.0"
                );
            }
        }

        #[test]
        fn test_rem_by_zero() {
            let data_ones = [1.0; LANE_COUNT];
            let data_zeros = [0.0; LANE_COUNT];
            let v_ones = F32x8::new(&data_ones);
            let v_zeros = F32x8::new(&data_zeros);

            let res = v_ones % v_zeros; // x % 0 should be NaN
            let elements = res.to_vec();
            for &x in &elements {
                debug_assert!(x.is_nan(), "Expected NaN from x % 0.0");
            }
        }

        #[test]
        fn test_partial_eq() {
            let (v1, v2, _) = setup_vecs();
            let v1_clone = v1; // F32x8 is Copy

            assert_eq!(v1, v1_clone); // Should be true (all 8 lanes equal)
            assert_ne!(v1, v2); // Should be false (lanes differ)

            // Test with partial vectors (size < LANE_COUNT)
            // Note: PartialEq checks all 8 underlying floats if sizes are equal.
            // If F32x8 has size=3, and raw elements are [1,2,3,0,0,0,0,0] for both,
            // they will be equal.
            let p_data1 = [1.0, 2.0, 3.0];
            let vp1 = F32x8::new(&p_data1); // size=3, raw=[1,2,3,0,0,0,0,0]
            let vp2 = F32x8::new(&p_data1); // size=3, raw=[1,2,3,0,0,0,0,0]
            assert_eq!(vp1, vp2);

            let p_data2 = [1.0, 2.0, 4.0];
            let vp3 = F32x8::new(&p_data2); // size=3, raw=[1,2,4,0,0,0,0,0]
            assert_ne!(vp1, vp3);

            // Test edge case: values in padded area differ, but size-defined areas are same.
            // Example: vA size=3, raw=[1,2,3, 9,9,0,0,0]
            //          vB size=3, raw=[1,2,3, 8,8,0,0,0]
            // Current `eq` checks all 8 floats, so vA != vB.
            let mut v_pad_diff1 = vp1; // raw=[1,2,3,0,0,0,0,0], size=3
            let mut v_pad_diff2 = vp2; // raw=[1,2,3,0,0,0,0,0], size=3

            unsafe {
                // Manually alter padding area (highly unsafe, just for test)
                // This simulates if load_partial was implemented differently or state got corrupted
                // Or if two vectors were constructed from different full arrays but with same partial size.
                let mut raw1 = get_all_elements(v_pad_diff1);
                raw1[LANE_COUNT - 1] = 99.0;
                v_pad_diff1.elements = _mm256_loadu_ps(raw1.as_ptr());

                let mut raw2 = get_all_elements(v_pad_diff2);
                raw2[LANE_COUNT - 1] = 88.0;
                v_pad_diff2.elements = _mm256_loadu_ps(raw2.as_ptr());
            }
            // v_pad_diff1.size and v_pad_diff2.size are still 3.
            // First 3 elements are [1,2,3] for both.
            // Padded areas differ.
            // Current `eq` will find them not equal because it checks all 8 lanes.
            assert_ne!(v_pad_diff1, v_pad_diff2);
        }

        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        #[test]
        fn test_partial_ord_full_lanes() {
            // Test lt, le, gt, ge which use self.size or 0xFF mask
            // Here self.size = LANE_COUNT (8)
            let v_small = unsafe { F32x8::splat(1.0) }; // size=8
            let v_large = unsafe { F32x8::splat(2.0) }; // size=8
            let v_equal = unsafe { F32x8::splat(1.0) }; // size=8

            // lt: uses mask (1 << size) - 1. For size=8, this is 0xFF.
            debug_assert!(v_small < v_large);
            debug_assert!(!(v_large < v_small));
            debug_assert!(!(v_small < v_equal));

            // le: uses mask 0xFF.
            debug_assert!(v_small <= v_large);
            debug_assert!(v_small <= v_equal);
            debug_assert!(!(v_large <= v_small));

            // gt: uses mask (1 << size) - 1. For size=8, this is 0xFF.
            debug_assert!(v_large > v_small);
            debug_assert!(!(v_small > v_large));
            debug_assert!(!(v_small > v_equal));

            // ge: uses mask 0xFF.
            debug_assert!(v_large >= v_small);
            debug_assert!(v_small >= v_equal);
            debug_assert!(!(v_small >= v_large));
        }

        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        #[test]
        fn test_partial_ord_partial_lanes_size_matters() {
            // Test lt, le, gt, ge with self.size < LANE_COUNT
            // lt/gt use size-dependent mask. le/ge use fixed 0xFF mask.
            let data_s = [1.0, 1.0, 1.0, 1.0]; // first 4 elements for size=4
            let data_l = [2.0, 2.0, 2.0, 2.0];
            let data_e = [1.0, 1.0, 1.0, 1.0];

            // Create F32x8 vectors with size=4. Raw vectors will be zero-padded.
            // v_s_p = [1,1,1,1, 0,0,0,0], size=4
            // v_l_p = [2,2,2,2, 0,0,0,0], size=4
            // v_e_p = [1,1,1,1, 0,0,0,0], size=4
            let v_s_p = F32x8::new(&data_s);
            let v_l_p = F32x8::new(&data_l);
            let v_e_p = F32x8::new(&data_e);

            // lt: uses mask (1 << 4) - 1 = 0xF. Compares first 4 elements of raw __m256.
            // v_s_p.elements[0..4] < v_l_p.elements[0..4] is true.
            // Padded elements are 0.0 == 0.0, so they are not <.
            // The comparison _mm256_cmp_ps(v_s_p.elements, v_l_p.elements, _CMP_LT_OQ)
            // will result in [T,T,T,T, F,F,F,F] (represented as mask values).
            // _mm256_movemask_ps will give 0x0F.
            // (0x0F == ((1 << 4) - 1)) is true.
            debug_assert!(v_s_p < v_l_p, "v_s_p < v_l_p should be true for size=4");
            debug_assert!(!(v_l_p < v_s_p));
            debug_assert!(!(v_s_p < v_e_p));

            // le: uses mask 0xFF. Compares all 8 elements of raw __m256.
            // For v_s_p <= v_l_p:
            // First 4: 1.0 <= 2.0 (True)
            // Padded:  0.0 <= 0.0 (True)
            // So all 8 are LE. _mm256_movemask_ps gives 0xFF.
            // (0xFF == 0xFF) is true.
            debug_assert!(
                v_s_p <= v_l_p,
                "v_s_p <= v_l_p should be true for size=4 if padding matches"
            );
            debug_assert!(
                v_s_p <= v_e_p,
                "v_s_p <= v_e_p should be true for size=4 if padding matches"
            );
            debug_assert!(!(v_l_p <= v_s_p));

            // gt: analogous to lt
            debug_assert!(v_l_p > v_s_p, "v_l_p > v_s_p should be true for size=4");
            debug_assert!(!(v_s_p > v_l_p));
            debug_assert!(!(v_s_p > v_e_p));

            // ge: analogous to le
            debug_assert!(
                v_l_p >= v_s_p,
                "v_l_p >= v_s_p should be true for size=4 if padding matches"
            );
            debug_assert!(
                v_s_p >= v_e_p,
                "v_s_p >= v_e_p should be true for size=4 if padding matches"
            );
            debug_assert!(!(v_s_p >= v_l_p));

            // Create a case where le/ge is false due to padding
            // v_mixed_s = [1,1,1,1, 9,9,9,9], size=4
            // v_mixed_l = [2,2,2,2, 0,0,0,0], size=4
            let mut raw_mixed_s_arr = [0.0; LANE_COUNT];
            raw_mixed_s_arr[0..4].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
            raw_mixed_s_arr[4..8].copy_from_slice(&[9.0, 9.0, 9.0, 9.0]);
            let mut v_mixed_s = F32x8::new(&raw_mixed_s_arr[0..4]); // Initialized with [1,1,1,1,0,0,0,0], size=4
            v_mixed_s.elements = unsafe { _mm256_loadu_ps(raw_mixed_s_arr.as_ptr()) }; // Force raw elements

            // v_mixed_s <= v_l_p should be FALSE for le (due to 0xFF mask) because 9.0 !<= 0.0
            debug_assert!(
                !(v_mixed_s <= v_l_p),
                "v_mixed_s <= v_l_p should be false due to padding"
            );
            // v_mixed_s < v_l_p should be TRUE for lt (due to 0x0F mask for size=4)
            debug_assert!(
                v_mixed_s < v_l_p,
                "v_mixed_s < v_l_p should be true based on first 4 elements"
            );
        }

        #[test]
        fn test_partial_cmp_specific_0xf_mask_behavior() {
            // Test partial_cmp's specific behavior with 0xF masks.
            // This means first 4 lanes are <, next 4 lanes are == (not < and not >).
            let mut arr_a = [0.0f32; LANE_COUNT];
            let mut arr_b = [0.0f32; LANE_COUNT];

            // Case 1: a[0..4] < b[0..4], a[4..8] == b[4..8] -> None
            for i in 0..4 {
                arr_a[i] = 1.0;
                arr_b[i] = 2.0;
            } // a < b
            for i in 4..8 {
                arr_a[i] = 3.0;
                arr_b[i] = 3.0;
            } // a == b
            let va = F32x8::new(&arr_a);
            let vb = F32x8::new(&arr_b);
            assert_eq!(va.partial_cmp(&vb), None);

            // Case 2: a[0..4] > b[0..4], a[4..8] == b[4..8] -> None
            for i in 0..4 {
                arr_a[i] = 2.0;
                arr_b[i] = 1.0;
            } // a > b
            for i in 4..8 {
                arr_a[i] = 3.0;
                arr_b[i] = 3.0;
            } // a == b
            let va = F32x8::new(&arr_a);
            let vb = F32x8::new(&arr_b);
            assert_eq!(va.partial_cmp(&vb), None);

            // Case 3: a[0..4] == b[0..4], a[4..8] are mixed (e.g. a < b) -> Ordering::Equal
            // This happens because eq_mask is 0xF, and lt_mask/gt_mask for remaining are not 0xF.
            // The condition is (lt_mask=0, gt_mask=0, eq_mask=0xF)
            // This means a[0..4] == b[0..4].
            // And for a[4..8], they are NOT all <, NOT all >, NOT all ==.
            // This means for a[4..8], they are not compared for equality for this branch.
            // The definition of `eq_elements` provides an `F32x8` mask.
            // `eq_mask = _mm256_movemask_ps(eq_elements.elements)`
            // If `eq_mask == 0xF`, then only first 4 elements of `eq_elements` are true.
            // The other elements of `eq_elements` must be false.
            // So: a[0..4] == b[0..4], and a[4..8] != b[4..8].
            for i in 0..4 {
                arr_a[i] = 1.0;
                arr_b[i] = 1.0;
            } // a == b
            for i in 4..8 {
                arr_a[i] = 1.0;
                arr_b[i] = 2.0;
            } // a < b (makes a[4..8] != b[4..8])
            let va = F32x8::new(&arr_a);
            let vb = F32x8::new(&arr_b);
            assert_eq!(va.partial_cmp(&vb), None);

            // Case 4: All elements equal (all 8)
            let v_all_eq1 = unsafe { F32x8::splat(1.0) };
            let v_all_eq2 = unsafe { F32x8::splat(1.0) };
            assert_eq!(
                v_all_eq1.partial_cmp(&v_all_eq2),
                Some(std::cmp::Ordering::Equal)
            );

            // Case 5: All elements less (all 8) -> None, because lt_mask will be 0xFF, not 0xF.
            let v_all_lt1 = unsafe { F32x8::splat(1.0) };
            let v_all_lt2 = unsafe { F32x8::splat(2.0) };
            assert_eq!(
                v_all_lt1.partial_cmp(&v_all_lt2),
                Some(std::cmp::Ordering::Less)
            );

            let v_all_lt1 = unsafe { F32x8::splat(1.0) };
            let v_all_lt2 = unsafe { F32x8::splat(2.0) };
            assert_eq!(
                v_all_lt2.partial_cmp(&v_all_lt1),
                Some(std::cmp::Ordering::Greater)
            );
        }

        #[test]
        fn test_bitwise_ops() {
            // Bitwise ops on floats are uncommon unless using their bit patterns as masks.
            // Values are typically all-bits-one (like TRUE_MASK_F32) or all-bits-zero.
            let v_true = unsafe { F32x8::splat(TRUE_MASK_F32) };
            let v_false = unsafe { F32x8::splat(FALSE_MASK_F32) };

            let data_pattern_arr = [
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
            ];
            let v_pattern = F32x8::new(&data_pattern_arr); // 10101010 pattern

            // BitAnd
            let and_res1 = v_true & v_pattern; // T & P = P
            assert_f32_slice_eq_bitwise(&and_res1.to_vec(), &data_pattern_arr);
            let and_res2 = v_false & v_pattern; // F & P = F
            assert_f32_slice_eq_bitwise(&and_res2.to_vec(), &[FALSE_MASK_F32; LANE_COUNT]);

            // BitOr
            let or_res1 = v_true | v_pattern; // T | P = T
            assert_f32_slice_eq_bitwise(&or_res1.to_vec(), &[TRUE_MASK_F32; LANE_COUNT]);
            let or_res2 = v_false | v_pattern; // F | P = P
            assert_f32_slice_eq_bitwise(&or_res2.to_vec(), &data_pattern_arr);

            // BitAndAssign
            let mut v_pat_copy = v_pattern;
            v_pat_copy &= v_true;
            assert_f32_slice_eq_bitwise(&v_pat_copy.to_vec(), &data_pattern_arr);

            // BitOrAssign
            v_pat_copy |= v_false; // P |= F is still P
            assert_f32_slice_eq_bitwise(&v_pat_copy.to_vec(), &data_pattern_arr);
        }

        #[test]
        fn test_operator_panics_on_diff_size() {
            let v1 = F32x8::new(&[1.0; LANE_COUNT]);
            let mut v2 = F32x8::new(&[2.0; LANE_COUNT]);
            v2.size = LANE_COUNT - 1; // Manually make size different

            macro_rules! check_panic {
                ($op:expr) => {
                    let result = catch_unwind(AssertUnwindSafe(|| $op));
                    debug_assert!(
                        result.is_err(),
                        "Operation did not panic with different sizes"
                    );
                    // Can also check panic message if it's consistent
                    // let panic_payload = result.err().unwrap();
                    // if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    //     debug_assert!(s.contains("Operands must have the same size"));
                    // } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    //     debug_assert!(s.contains("Operands must have the same size"));
                    // }
                };
            }

            check_panic!(v1 + v2);
            check_panic!(v1 - v2);
            check_panic!(v1 * v2);
            check_panic!(v1 / v2);
            check_panic!(v1 % v2);
            check_panic!(v1 == v2); // PartialEq panics due to assert
            check_panic!(v1.partial_cmp(&v2)); // PartialOrd panics due to assert
                                               // Bitwise ops
            check_panic!(v1 & v2);
            check_panic!(v1 | v2);

            // Assign ops need mutable v1
            let mut vm = v1;
            check_panic!(vm += v2);
            vm = v1;
            check_panic!(vm -= v2);
            vm = v1;
            check_panic!(vm *= v2);
            vm = v1;
            check_panic!(vm /= v2);
            vm = v1;
            check_panic!(vm %= v2);
            vm = v1;
            check_panic!(vm &= v2);
            vm = v1;
            check_panic!(vm |= v2);
        }
    }
}
