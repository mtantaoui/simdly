#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::simd::traits::SimdVec;

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
    size: usize,
    elements: __m256,
}

impl SimdVec<f32> for F32x8 {
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
            elements: unsafe { _mm256_set1_ps(value) },
            size: LANE_COUNT,
        }
    }

    /// Checks if the pointer is aligned to 32 bytes.
    /// This is necessary for AVX operations to ensure optimal performance.
    #[inline(always)]
    fn is_aligned(ptr: *const f32) -> bool {
        let ptr = ptr as usize;
        ptr % 32 == 0
    }

    /// Loads a vector from a pointer, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        // Asserts that the pointer is not null and the size is exactly 8 elements.
        // If the pointer is null or the size is not 8, it will panic with an error message.
        assert!(!ptr.is_null(), "Pointer must not be null");
        assert!(size == LANE_COUNT, "Size must be == {}", LANE_COUNT);

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
            elements: unsafe { _load_aligned(ptr) },
            size,
        }
    }

    /// Loads a vector from an unaligned pointer, ensuring the size is exactly 8 elements.
    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const f32, size: usize) -> Self {
        Self {
            elements: unsafe { _load_unaligned(ptr) },
            size,
        }
    }

    /// Loads a partial vector from a pointer, filling the rest with zeros.
    #[target_feature(enable = "avx")]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        assert!(
            size < LANE_COUNT,
            "{}",
            format!("Size must be < {}", LANE_COUNT)
        );

        // to avoid unnecessary branching and ensure all elements are set correctly
        assert!(!ptr.is_null(), "Pointer must not be null");

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
    #[target_feature(enable = "avx")]
    unsafe fn store_in_vec(&self) -> Vec<f32> {
        assert!(
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
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {}", LANE_COUNT)
        );
        assert!(!ptr.is_null(), "Pointer must not be null");

        // Check if the pointer is aligned to 32 bytes
        // If it is aligned, use `_mm256_stream_ps` for better performance
        // If it is not aligned, use `_mm256_storeu_ps` for unaligned storage
        match Self::is_aligned(ptr) {
            true => unsafe { _stream(ptr, self.elements) },
            // true => unsafe { _store_aligned(ptr, self.elements) },
            false => unsafe { _store_unaligned(ptr, self.elements) },
        }
    }

    /// Stores the vector elements into a memory location pointed to by `ptr`, filling only the first `size` elements.
    /// This method is unsafe because it assumes that the pointer is valid and aligned.     
    #[target_feature(enable = "avx")]
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be < {}", LANE_COUNT)
        );
        assert!(!ptr.is_null(), "Pointer must not be null");

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
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {}", LANE_COUNT)
        );

        if self.size == LANE_COUNT {
            unsafe { self.store_in_vec() }
        } else {
            unsafe { self.store_in_vec_partial() }
        }
    }

    /// Compares two vectors elementwise for equality.    
    #[target_feature(enable = "avx")]
    unsafe fn eq_elements(&self, rhs: Self) -> Self {
        assert!(
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
    #[target_feature(enable = "avx")]
    unsafe fn lt_elements(&self, rhs: Self) -> Self {
        assert!(
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
    #[target_feature(enable = "avx")]
    unsafe fn le_elements(&self, rhs: Self) -> Self {
        assert!(
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
    #[target_feature(enable = "avx")]
    unsafe fn gt_elements(&self, rhs: Self) -> Self {
        assert!(
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
    #[target_feature(enable = "avx")]
    unsafe fn ge_elements(&self, rhs: Self) -> Self {
        assert!(
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
}

/// Loads a vector from an aligned pointer, ensuring the size is exactly 8 elements.
#[inline(always)]
unsafe fn _load_aligned(ptr: *const f32) -> __m256 {
    unsafe { _mm256_load_ps(ptr) }
}

/// Loads a vector from an unaligned pointer, ensuring the size is exactly 8 elements.
#[inline(always)]
unsafe fn _load_unaligned(ptr: *const f32) -> __m256 {
    unsafe { _mm256_loadu_ps(ptr) }
}

/// Stores a __m256.
#[inline(always)]
unsafe fn _store_aligned(ptr: *mut f32, elements: __m256) {
    unsafe { _mm256_store_ps(ptr, elements) }
}

/// Stores a __m256.
#[inline(always)]
unsafe fn _store_unaligned(ptr: *mut f32, elements: __m256) {
    unsafe { _mm256_storeu_ps(ptr, elements) }
}

#[inline(always)]
unsafe fn _stream(ptr: *mut f32, elements: __m256) {
    unsafe { _mm256_stream_ps(ptr, elements) }
}

/// Adds two `__m256` vectors elementwise.
#[inline(always)]
unsafe fn _add(lhs: __m256, rhs: __m256) -> __m256 {
    unsafe { _mm256_add_ps(lhs, rhs) }
}

/// Subtracts two `__m256` vectors elementwise.
#[target_feature(enable = "avx")]
unsafe fn _sub(lhs: __m256, rhs: __m256) -> __m256 {
    unsafe { _mm256_sub_ps(lhs, rhs) }
}

/// Multiplies two `__m256` vectors elementwise.
#[target_feature(enable = "avx")]
unsafe fn _mul(lhs: __m256, rhs: __m256) -> __m256 {
    unsafe { _mm256_mul_ps(lhs, rhs) }
}

/// Divides two `__m256` vectors elementwise.
#[target_feature(enable = "avx")]
unsafe fn _div(lhs: __m256, rhs: __m256) -> __m256 {
    unsafe { _mm256_div_ps(lhs, rhs) }
}

/// Computes the remainder of two `__m256` vectors elementwise.
#[target_feature(enable = "avx")]
unsafe fn _rem(lhs: __m256, rhs: __m256) -> __m256 {
    let div = unsafe { _mm256_div_ps(lhs, rhs) };
    let floor = unsafe { _mm256_floor_ps(div) };
    let prod = unsafe { _mm256_mul_ps(floor, rhs) };

    unsafe { _mm256_sub_ps(lhs, prod) }
}

/// Compares two `__m256` vectors for equality.
#[target_feature(enable = "avx")]
unsafe fn _lt(lhs: __m256, rhs: __m256, size: usize) -> bool {
    let lt: __m256 = unsafe { _mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ) };
    let lt_mask = unsafe { _mm256_movemask_ps(lt) };

    let mask = (1 << size) - 1; // Create a mask with the first `size` bits set to 1

    lt_mask == mask
}

/// Compares two `__m256` vectors for less than or equal to.
#[target_feature(enable = "avx")]
unsafe fn _le(lhs: __m256, rhs: __m256) -> bool {
    let le: __m256 = unsafe { _mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ) };
    let le_mask = unsafe { _mm256_movemask_ps(le) };

    le_mask == 0xFF
}

/// Compares two `__m256` vectors for greater than.
#[target_feature(enable = "avx")]
unsafe fn _gt(lhs: __m256, rhs: __m256, size: usize) -> bool {
    let gt: __m256 = _mm256_cmp_ps(lhs, rhs, _CMP_GT_OQ);
    let gt_mask = _mm256_movemask_ps(gt);

    let mask = (1 << size) - 1; // Create a mask with the first `size` bits set to 1

    gt_mask == mask
}

/// Compares two `__m256` vectors for greater than or equal to.
#[target_feature(enable = "avx")]
unsafe fn _ge(lhs: __m256, rhs: __m256) -> bool {
    let ge: __m256 = _mm256_cmp_ps(lhs, rhs, _CMP_GE_OQ);
    let ge_mask = _mm256_movemask_ps(ge);

    ge_mask == 0xFF
}

/// Performs a bitwise AND operation on two `__m256` vectors.
#[target_feature(enable = "avx")]
unsafe fn _and(lhs: __m256, rhs: __m256) -> __m256 {
    unsafe { _mm256_and_ps(lhs, rhs) }
}

#[target_feature(enable = "avx")]
unsafe fn _or(lhs: __m256, rhs: __m256) -> __m256 {
    unsafe { _mm256_or_ps(lhs, rhs) }
}

/// Implementing the `Add` and `AddAssign` traits for F32x8
/// This allows for using the `+` operator and `+=` operator with F32x8 vectors.    
impl Add for F32x8 {
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
            elements: unsafe { _add(self.elements, rhs.elements) },
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
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Use the sub function to perform element-wise subtraction
        Self {
            size: self.size,
            elements: unsafe { _sub(self.elements, rhs.elements) },
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
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        Self {
            size: self.size,
            elements: unsafe { _mul(self.elements, rhs.elements) },
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
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        Self {
            size: self.size,
            elements: unsafe { _div(self.elements, rhs.elements) },
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
        assert!(
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
        assert!(
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
        assert!(
            self.size == other.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            other.size
        );

        unsafe {
            let lt = self.lt_elements(*other).elements;
            let gt = self.gt_elements(*other).elements;
            let eq = self.eq_elements(*other).elements;

            let lt_mask = _mm256_movemask_ps(lt);
            let gt_mask = _mm256_movemask_ps(gt);
            let eq_mask = _mm256_movemask_ps(eq);

            match (lt_mask, gt_mask, eq_mask) {
                (0xF, 0x0, _) => Some(std::cmp::Ordering::Less), // all lanes less
                (0x0, 0xF, _) => Some(std::cmp::Ordering::Greater), // all lanes greater
                (0x0, 0x0, 0xF) => Some(std::cmp::Ordering::Equal), // all lanes equal
                _ => None,                                       // mixed
            }
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
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        F32x8 {
            size: self.size,
            elements: unsafe { _and(self.elements, rhs.elements) },
        }
    }
}

impl BitAndAssign for F32x8 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        assert!(
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
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        F32x8 {
            size: self.size,
            elements: unsafe { _or(self.elements, rhs.elements) },
        }
    }
}

impl BitOrAssign for F32x8 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        assert!(
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
mod f32x8_tests {
    use std::vec;

    use super::*;

    #[test]
    /// __m256 fields are private and cannot be compared directly
    /// test consist on loading elements to __m256 then fetching them using .to_vec method
    /// implicitly tests load, load_partial, store, store_partial and to_vec methods
    fn test_new() {
        let a1 = vec![1.0];
        let v1 = F32x8::new(&a1);

        assert_eq!(a1, v1.to_vec());
        assert_eq!(a1.len(), v1.size);

        let a2 = vec![1.0, 2.0];
        let v2 = F32x8::new(&a2);

        assert_eq!(a2, v2.to_vec());
        assert_eq!(a2.len(), v2.size);

        let a3 = vec![1.0, 2.0, 3.0];
        let v3 = F32x8::new(&a3);

        assert_eq!(a3, v3.to_vec());
        assert_eq!(a3.len(), v3.size);

        let a4 = vec![1.0, 2.0, 3.0, 4.0];
        let v4 = F32x8::new(&a4);

        assert_eq!(a4, v4.to_vec());
        assert_eq!(a4.len(), v4.size);
    }

    /// Splat method should duplicate one value for all elements of __m256
    #[test]
    fn test_splat() {
        let a = vec![1.0; LANE_COUNT];

        let v = unsafe { F32x8::splat(1.0) };

        assert_eq!(a, v.to_vec())
    }

    #[test]
    fn test_store_at() {
        let mut a = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];

        let v = F32x8::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        unsafe { v.store_at(a[1..].as_mut_ptr()) };

        assert_eq!(vec![11.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 22.0], a);
    }

    #[test]
    fn test_store_at_partial() {
        let mut a3 = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];
        let v3 = F32x8::new(&[1.0, 2.0, 3.0]);

        unsafe { v3.store_at_partial(a3[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 3.0, 19.0, 22.0],
            a3
        );

        let mut a2 = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];
        let v2 = F32x8::new(&[1.0, 2.0]);

        unsafe { v2.store_at_partial(a2[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 2.0, 18.0, 19.0, 22.0],
            a2
        );

        let mut a1 = vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 22.0];
        let v1 = F32x8::new(&[1.0]);

        unsafe { v1.store_at_partial(a1[5..].as_mut_ptr()) };

        assert_eq!(
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 1.0, 17.0, 18.0, 19.0, 22.0],
            a1
        );
    }

    #[test]
    fn test_add() {
        let v1 = F32x8::new(&[1.0]);
        let u1 = F32x8::new(&[5.0]);

        assert_eq!(vec![6.0], (u1 + v1).to_vec());

        let v2 = F32x8::new(&[1.0, 10.0]);
        let u2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![6.0, 21.0], (u2 + v2).to_vec());

        let v3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let u3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![6.0, 21.0, 16.0], (u3 + v3).to_vec());

        let v4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let u4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(vec![6.0, 21.0, 16.0, 7.0], (u4 + v4).to_vec());

        let v5 = F32x8::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let u5 = F32x8::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(vec![6.0, 21.0, 16.0, 7.0, 2.0], (u5 + v5).to_vec());
    }

    #[test]
    fn test_add_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a += b;

        assert_eq!(vec![5.0; 4], a.to_vec());
    }

    #[test]
    fn test_sub() {
        let v1 = F32x8::new(&[1.0]);
        let u1 = F32x8::new(&[5.0]);

        assert_eq!(vec![6.0], (u1 + v1).to_vec());

        let v2 = F32x8::new(&[1.0, 10.0]);
        let u2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![6.0, 21.0], (u2 + v2).to_vec());

        let v3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let u3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![6.0, 21.0, 16.0], (u3 + v3).to_vec());

        let v4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let u4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(vec![6.0, 21.0, 16.0, 7.0], (u4 + v4).to_vec());
    }

    #[test]
    fn test_sub_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a -= b;

        assert_eq!(vec![-3.0, -1.0, 1.0, 3.0], a.to_vec());
    }

    #[test]
    fn test_mul() {
        let v1 = F32x8::new(&[1.0]);
        let u1 = F32x8::new(&[5.0]);

        assert_eq!(vec![5.0], (u1 * v1).to_vec());

        let v2 = F32x8::new(&[1.0, 10.0]);
        let u2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![5.0, 110.0], (u2 * v2).to_vec());

        let v3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let u3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![5.0, 110.0, 63.0], (u3 * v3).to_vec());

        let v4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let u4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(vec![5.0, 110.0, 63.0, 10.0], (u4 * v4).to_vec());

        let v5 = F32x8::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let u5 = F32x8::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(vec![5.0, 110.0, 63.0, 10.0, 1.0], (u5 * v5).to_vec());
    }

    #[test]
    fn test_mul_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a *= b;

        assert_eq!(vec![4.0, 6.0, 6.0, 4.0], a.to_vec());
    }

    #[test]
    fn test_div() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!(vec![1.0 / 5.0], (u1 / v1).to_vec());

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![1.0 / 5.0, 10.0 / 11.0], (u2 / v2).to_vec());

        let u3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![1.0 / 5.0, 10.0 / 11.0, 7.0 / 9.0], (u3 / v3).to_vec());

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 / 5.0, 10.0 / 11.0, 7.0 / 9.0, 2.0 / 5.0],
            (u4 / v4).to_vec()
        );

        let u5 = F32x8::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x8::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 / 5.0, 10.0 / 11.0, 7.0 / 9.0, 2.0 / 5.0, 1.0 / 1.0],
            (u5 / v5).to_vec()
        );
    }

    #[test]
    fn test_div_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a /= b;

        assert_eq!(vec![1.0 / 4.0, 2.0 / 3.0, 3.0 / 2.0, 4.0], a.to_vec());
    }

    #[test]
    fn test_rem() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!(vec![1.0 % 5.0], (u1 % v1).to_vec());

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(vec![1.0 % 5.0, 10.0 % 11.0], (u2 % v2).to_vec());

        let u3 = F32x8::new(&[1.0, 10.0, 7.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 9.0]);

        assert_eq!(vec![1.0 % 5.0, 10.0 % 11.0, 7.0 % 9.0], (u3 % v3).to_vec());

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![1.0 % 5.0, 10.0 % 11.0, 7.0 % 9.0, 2.0 % 5.0],
            (u4 % v4).to_vec()
        );

        let u5 = F32x8::new(&[1.0, 10.0, 7.0, 2.0, 1.0]);
        let v5 = F32x8::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        assert_eq!(
            vec![1.0 % 5.0, 10.0 % 11.0, 7.0 % 9.0, 2.0 % 5.0, 1.0 % 1.0],
            (u5 % v5).to_vec()
        );
    }

    #[test]
    fn test_rem_assign() {
        let mut a = F32x8::new(&[1.0, 2.0, 3.0, 4.0]);
        let b = F32x8::new(&[4.0, 3.0, 2.0, 1.0]);

        a %= b;

        assert_eq!(vec![1.0 % 4.0, 2.0 % 3.0, 3.0 % 2.0, 4.0 % 1.0], a.to_vec());
    }

    #[test]
    fn test_lt_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        let lt = unsafe { u1.lt_elements(v1) };

        assert_eq!(
            vec![1.0 < 5.0],
            lt.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        let lt = unsafe { u2.lt_elements(v2) };

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0],
            lt.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        let lt = unsafe { u3.lt_elements(v3) };

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0, 9.0 < 7.0],
            lt.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        let lt = unsafe { u4.lt_elements(v4) };

        assert_eq!(
            vec![1.0 < 5.0, 10.0 < 11.0, 7.0 < 9.0, 2.0 < 5.0],
            lt.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_le_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        let le = unsafe { u1.le_elements(v1) };

        assert_eq!(
            vec![1.0 <= 5.0],
            le.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        let le = unsafe { u2.le_elements(v2) };

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0],
            le.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        let le = unsafe { u3.le_elements(v3) };

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0, 9.0 <= 7.0],
            le.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        let le = unsafe { u4.le_elements(v4) };

        assert_eq!(
            vec![1.0 <= 5.0, 10.0 <= 11.0, 7.0 <= 9.0, 2.0 <= 5.0],
            le.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_gt_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        let gt = unsafe { u1.gt_elements(v1) };

        assert_eq!(
            vec![1.0 > 5.0],
            gt.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        let gt = unsafe { u2.gt_elements(v2) };

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0],
            gt.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        let gt = unsafe { u3.gt_elements(v3) };

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0, 9.0 > 7.0],
            gt.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        let gt = unsafe { u4.gt_elements(v4) };

        assert_eq!(
            vec![1.0 > 5.0, 10.0 > 11.0, 7.0 > 9.0, 2.0 > 5.0],
            gt.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_ge_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        let ge = unsafe { u1.ge_elements(v1) };

        assert_eq!(
            vec![1.0 >= 5.0],
            ge.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        let ge = unsafe { u2.ge_elements(v2) };

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0],
            ge.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        let ge = unsafe { u3.ge_elements(v3) };

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0, 9.0 >= 7.0],
            ge.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        let ge = unsafe { u4.ge_elements(v4) };

        assert_eq!(
            vec![1.0 >= 5.0, 10.0 >= 11.0, 7.0 >= 9.0, 2.0 >= 5.0],
            ge.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_eq_elementwise() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        let eq = unsafe { u1.eq_elements(v1) };

        assert_eq!(
            vec![1.0 == 5.0],
            eq.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 10.0]);

        let eq = unsafe { u2.eq_elements(v2) };

        assert_eq!(
            vec![0x00000000, 0xFFFFFFFF],
            eq.to_vec()
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<u32>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        let eq = unsafe { u3.eq_elements(v3) };

        assert_eq!(
            vec![1.0 == 5.0, 10.0 == 11.0, 9.0 == 7.0],
            eq.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        let eq = unsafe { u4.eq_elements(v4) };

        assert_eq!(
            vec![1.0 == 5.0, 10.0 == 11.0, 7.0 == 9.0, 2.0 == 5.0],
            eq.to_vec().iter().map(|f| *f != 0.0).collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0, 0.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0, 1.0]);

        let eq = u4 == v4;

        assert!(!eq);
    }

    #[test]
    fn test_eq() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 == 5.0].iter().all(|f| *f), u1 == v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 == 5.0, 10.0 == 11.0].iter().all(|f| *f), u2 == v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 == 5.0, 10.0 == 11.0, 9.0 == 7.0].iter().all(|f| *f),
            u3 == v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 == 5.0, 10.0 == 11.0, 7.0 == 9.0, 2.0 == 5.0]
                .iter()
                .all(|f| *f),
            u4 == v4
        );
    }

    #[test]
    fn test_lt() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 < 5.0].iter().all(|f| *f), u1 < v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 < 5.0, 10.0 < 11.0].iter().all(|f| *f), u2 < v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 < 5.0, 10.0 < 11.0, 9.0 < 7.0].iter().all(|f| *f),
            u3 < v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 < 5.0, 10.0 < 11.0, 7.0 < 9.0, 2.0 < 5.0]
                .iter()
                .all(|f| *f),
            u4 < v4
        );

        let u4 = F32x8::new(&[1.0, 12.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 < 5.0, 12.0 < 11.0, 7.0 < 9.0, 2.0 < 5.0]
                .iter()
                .all(|f| *f),
            u4 < v4
        );
    }

    #[test]
    fn test_le() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 <= 5.0].iter().all(|f| *f), u1 <= v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 <= 5.0, 10.0 <= 11.0].iter().all(|f| *f), u2 <= v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 <= 5.0, 10.0 <= 11.0, 9.0 <= 7.0].iter().all(|f| *f),
            u3 <= v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 <= 5.0, 10.0 <= 11.0, 7.0 <= 9.0, 2.0 <= 5.0]
                .iter()
                .all(|f| *f),
            u4 <= v4
        );

        let u4 = F32x8::new(&[1.0, 12.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 <= 5.0, 12.0 <= 11.0, 7.0 <= 9.0, 2.0 <= 5.0]
                .iter()
                .all(|f| *f),
            u4 <= v4
        );
    }

    #[test]
    fn test_gt() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 > 5.0].iter().all(|f| *f), u1 > v1);

        let u1 = F32x8::new(&[5.0]);
        let v1 = F32x8::new(&[1.0]);

        assert_eq!([5.0 > 1.0].iter().all(|f| *f), u1 > v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 > 5.0, 10.0 > 11.0].iter().all(|f| *f), u2 > v2);

        let u2 = F32x8::new(&[5.0, 11.0]);
        let v2 = F32x8::new(&[1.0, 10.0]);

        assert_eq!([5.0 > 1.0, 11.0 > 10.0].iter().all(|f| *f), u2 > v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 > 5.0, 10.0 > 11.0, 9.0 > 7.0].iter().all(|f| *f),
            u3 > v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 > 5.0, 10.0 > 11.0, 7.0 > 9.0, 2.0 > 5.0]
                .iter()
                .all(|f| *f),
            u4 > v4
        );

        let u4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);
        let v4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);

        assert_eq!(
            [5.0 > 1.0, 11.0 > 10.0, 9.0 > 7.0, 5.0 > 2.0]
                .iter()
                .all(|f| *f),
            u4 > v4
        );
    }

    #[test]
    fn test_ge() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[5.0]);

        assert_eq!([1.0 >= 5.0].iter().all(|f| *f), u1 >= v1);

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!([1.0 >= 5.0, 10.0 >= 11.0].iter().all(|f| *f), u2 >= v2);

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            [1.0 >= 5.0, 10.0 >= 11.0, 9.0 >= 7.0].iter().all(|f| *f),
            u3 >= v3
        );

        let u4 = F32x8::new(&[1.0, 10.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            [1.0 >= 5.0, 10.0 >= 11.0, 7.0 >= 9.0, 2.0 >= 5.0]
                .iter()
                .all(|f| *f),
            u4 >= v4
        );
    }

    #[test]
    fn test_and() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[0.0]);

        assert_eq!(
            vec![false],
            (u1 & v1)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(
            vec![true, true],
            (u2 & v2)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![true, true, true],
            (u3 & v3)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[1.0, 0.0, 7.0, 2.0]);
        let v4 = F32x8::new(&[5.0, 11.0, 9.0, 5.0]);

        assert_eq!(
            vec![true, false, true, true],
            (u4 & v4)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }

    #[test]
    fn test_or() {
        let u1 = F32x8::new(&[1.0]);
        let v1 = F32x8::new(&[0.0]);

        assert_eq!(
            vec![true],
            (u1 | v1)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u2 = F32x8::new(&[1.0, 10.0]);
        let v2 = F32x8::new(&[5.0, 11.0]);

        assert_eq!(
            vec![true, true],
            (u2 | v2)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u3 = F32x8::new(&[1.0, 10.0, 9.0]);
        let v3 = F32x8::new(&[5.0, 11.0, 7.0]);

        assert_eq!(
            vec![true, true, true],
            (u3 | v3)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );

        let u4 = F32x8::new(&[0.0, 0.0, 7.0, 0.0]);
        let v4 = F32x8::new(&[0.0, 11.0, 9.0, 0.0]);

        assert_eq!(
            vec![false, true, true, false],
            (u4 | v4)
                .to_vec()
                .iter()
                .map(|f| *f != 0.0)
                .collect::<Vec<bool>>()
        );
    }
}
