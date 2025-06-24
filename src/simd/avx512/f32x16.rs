#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

use crate::simd::{
    avx512::{
        acos::float32::_mm512_acos_ps, asin::float32::_mm512_asin_ps, cos::float32::_mm512_cos_ps,
    },
    traits::SimdVec,
};

pub const AVX512_ALIGNMENT: usize = 64;

/// The number of f32 lanes in an AVX-512 vector.
pub const LANE_COUNT: usize = 16;

/// A 16-lane, 32-bit floating-point SIMD vector using AVX-512F intrinsics.
///
/// This struct holds a `__m512` vector and a `size` field. The `size` field
/// is used to track the number of active lanes, enabling support for vectors
/// with fewer than 16 elements (i.e., at the end of a data slice).
#[derive(Copy, Clone, Debug)]
#[repr(C)] // Ensure memory layout is predictable
pub struct F32x16 {
    elements: __m512,
    size: usize,
}

impl SimdVec<f32> for F32x16 {
    /// Creates a new vector from a slice.
    ///
    /// If the slice has fewer than 16 elements, it performs a partial load.
    /// If the slice has 16 or more elements, it loads the first 16.
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

    /// Creates a new vector with all lanes set to the same value.
    #[inline(always)]
    unsafe fn splat(value: f32) -> Self {
        Self {
            // SAFETY: AVX-512F is expected to be enabled by the caller or compilation flags.
            elements: unsafe { _mm512_set1_ps(value) },
            size: LANE_COUNT,
        }
    }

    /// Loads 16 elements from a pointer into a vector.
    ///
    /// This function uses an unaligned load for better general-purpose performance.
    #[inline(always)]
    unsafe fn load(ptr: *const f32, size: usize) -> Self {
        // Asserts that the pointer is not null and the size is exactly 8 elements.
        // If the pointer is null or the size is not 8, it will panic with an error message.
        debug_assert!(!ptr.is_null(), "Pointer must not be null");
        debug_assert!(size == LANE_COUNT, "Size must be exactly {LANE_COUNT}");

        match Self::is_aligned(ptr) {
            true => unsafe { Self::load_aligned(ptr, size) },
            false => unsafe { Self::load_unaligned(ptr, size) },
        }
    }

    /// Loads `size` elements from a pointer into a vector, zeroing the remaining lanes.
    /// `size` must be less than 16.
    ///
    /// This function uses an unaligned load for better general-purpose performance.
    #[inline(always)]
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self {
        debug_assert!(size < LANE_COUNT, "Size must be less than {LANE_COUNT}");

        // Create a bitmask with the lower `size` bits set to 1.
        // e.g., size = 3 -> mask = 0b0111
        let mask: __mmask16 = (1 << size) - 1;

        // SAFETY: The caller must ensure `ptr` is valid for reading `size` f32 values.
        // `_mm512_maskz_loadu_ps` loads `size` elements from `ptr` according to the mask
        // and sets the remaining elements in the vector to zero.
        let elements = _mm512_maskz_loadu_ps(mask, ptr);

        Self { elements, size }
    }

    /// Stores all 16 vector lanes to the memory location pointed to by `ptr`.
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
            // #[cfg(not(miri))]
            // true => unsafe { _mm512_stream_ps(ptr, self.elements) },
            // #[cfg(miri)]
            true => unsafe { _mm512_store_ps(ptr, self.elements) },
            false => unsafe { _mm512_storeu_ps(ptr, self.elements) },
        }
    }

    /// Stores the active `self.size` lanes to the memory location pointed to by `ptr`.
    #[inline(always)]
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        debug_assert!(self.size < LANE_COUNT, "Size must be < {LANE_COUNT}");

        // Create a bitmask for the active lanes.
        let mask: __mmask16 = (1 << self.size) - 1;

        // SAFETY: The caller must ensure `ptr` is valid for writing `self.size` f32 values.
        // The mask ensures that only the first `self.size` lanes are written.
        _mm512_mask_storeu_ps(ptr, mask, self.elements);
    }

    /// Converts the vector to a `Vec<f32>`.
    #[inline(always)]
    fn to_vec(self) -> Vec<f32> {
        debug_assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );
        // SAFETY: The `store` and `store_partial` methods contain the necessary `unsafe`
        // blocks and safety checks.

        if self.size == LANE_COUNT {
            unsafe { self.store_in_vec() }
        } else {
            unsafe { self.store_in_vec_partial() }
        }
    }

    /// Performs an element-wise equality comparison (a == b).
    /// Returns a vector where each lane is `-1.0` (all bits set) if true, and `0.0` if false.
    #[inline(always)]
    unsafe fn eq_elements(&self, rhs: Self) -> Self {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");

        // SAFETY: AVX-512F must be enabled.
        unsafe {
            // 1. Compare elements, producing a bitmask (`__mmask16`).
            let mask = _mm512_cmpeq_ps_mask(self.elements, rhs.elements);
            // 2. Convert the bitmask to a 32-bit integer vector mask (0 for false, -1 for true).
            let int_mask = _mm512_movm_epi32(mask);
            // 3. Convert the integer vector to a float vector (-1 -> -1.0, 0 -> 0.0).
            let elements = _mm512_cvtepi32_ps(int_mask);
            Self {
                elements,
                size: self.size,
            }
        }
    }

    /// Performs an element-wise less-than comparison (a < b).
    /// See `eq_elements` for return value details.
    #[inline(always)]
    unsafe fn lt_elements(&self, rhs: Self) -> Self {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        // SAFETY: AVX-512F must be enabled.
        unsafe {
            let mask = _mm512_cmplt_ps_mask(self.elements, rhs.elements);
            let elements = _mm512_cvtepi32_ps(_mm512_movm_epi32(mask));
            Self {
                elements,
                size: self.size,
            }
        }
    }

    /// Performs an element-wise less-than-or-equal comparison (a <= b).
    /// See `eq_elements` for return value details.
    #[inline(always)]
    unsafe fn le_elements(&self, rhs: Self) -> Self {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        // SAFETY: AVX-512F must be enabled.
        unsafe {
            let mask = _mm512_cmple_ps_mask(self.elements, rhs.elements);
            let elements = _mm512_cvtepi32_ps(_mm512_movm_epi32(mask));
            Self {
                elements,
                size: self.size,
            }
        }
    }

    /// Performs an element-wise greater-than comparison (a > b).
    /// See `eq_elements` for return value details.
    #[inline(always)]
    unsafe fn gt_elements(&self, rhs: Self) -> Self {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        // SAFETY: AVX-512F must be enabled.
        unsafe {
            // Reuse `cmplt` by swapping operands: (a > b) is equivalent to (b < a).
            let mask = _mm512_cmplt_ps_mask(rhs.elements, self.elements);
            let elements = _mm512_cvtepi32_ps(_mm512_movm_epi32(mask));
            Self {
                elements,
                size: self.size,
            }
        }
    }

    /// Performs an element-wise greater-than-or-equal comparison (a >= b).
    /// See `eq_elements` for return value details.
    #[inline(always)]
    unsafe fn ge_elements(&self, rhs: Self) -> Self {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        // SAFETY: AVX-512F must be enabled.
        unsafe {
            // Reuse `cmple` by swapping operands: (a >= b) is equivalent to (b <= a).
            let mask = _mm512_cmple_ps_mask(rhs.elements, self.elements);
            let elements = _mm512_cvtepi32_ps(_mm512_movm_epi32(mask));
            Self {
                elements,
                size: self.size,
            }
        }
    }

    /// Checks if the pointer is aligned to 32 bytes.
    /// This is necessary for AVX operations to ensure optimal performance.
    #[inline(always)]
    fn is_aligned(ptr: *const f32) -> bool {
        let ptr = ptr as usize;
        ptr % core::mem::align_of::<__m512>() == 0
    }

    #[inline(always)]
    /// Loads 16 elements from an aligned pointer into a vector.
    unsafe fn load_aligned(ptr: *const f32, size: usize) -> Self {
        Self {
            elements: unsafe { _mm512_load_ps(ptr) },
            size,
        }
    }

    #[inline(always)]
    /// Loads 16 elements from an unaligned pointer into a vector.
    unsafe fn load_unaligned(ptr: *const f32, size: usize) -> Self {
        Self {
            elements: unsafe { _mm512_loadu_ps(ptr) },
            size,
        }
    }

    #[inline(always)]
    unsafe fn store_in_vec(&self) -> Vec<f32> {
        debug_assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );

        let mut vec = Vec::with_capacity(LANE_COUNT);

        unsafe {
            _mm512_storeu_ps(vec.as_mut_ptr(), self.elements);
            vec.set_len(LANE_COUNT);
        }

        vec
    }

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

    unsafe fn abs(&self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm512_abs_ps(self.elements) },
        }
    }

    unsafe fn acos(&self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm512_acos_ps(self.elements) },
        }
    }

    unsafe fn asin(&self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm512_asin_ps(self.elements) },
        }
    }

    #[inline(always)]
    unsafe fn cos(&self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm512_cos_ps(self.elements) },
        }
    }

    #[inline(always)]
    unsafe fn fmadd(&self, a: Self, b: Self) -> Self {
        Self {
            size: self.size,
            elements: unsafe { _mm512_fmadd_ps(a.elements, b.elements, self.elements) },
        }
    }
}

/// Computes the remainder of two `__m512` vectors elementwise.
#[inline(always)]
unsafe fn _rem(lhs: __m512, rhs: __m512) -> __m512 {
    let div = _mm512_div_ps(lhs, rhs);
    // Use _mm512_floor_ps to match the behavior of Rust's % operator for floats.
    let floor = _mm512_roundscale_ps(div, _MM_FROUND_FLOOR | _MM_FROUND_NO_EXC);
    // Fused-multiply-add: -(floor * rhs) + lhs
    _mm512_fnmadd_ps(floor, rhs, lhs)
}

// Computes the bitwise AND of two `__m512` vectors.
#[target_feature(enable = "avx512dq")]
fn _and(lhs: __m512, rhs: __m512) -> __m512 {
    // This function is a placeholder for the actual implementation of bitwise AND.
    _mm512_and_ps(lhs, rhs)
}

#[target_feature(enable = "avx512dq")]
fn _or(lhs: __m512, rhs: __m512) -> __m512 {
    // This function is a placeholder for the actual implementation of bitwise AND.
    _mm512_or_ps(lhs, rhs)
}

// --- Operator Implementations ---

impl Add for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        Self {
            size: self.size,
            elements: unsafe { _mm512_add_ps(self.elements, rhs.elements) },
        }
    }
}
impl AddAssign for F32x16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        Self {
            size: self.size,
            elements: unsafe { _mm512_sub_ps(self.elements, rhs.elements) },
        }
    }
}
impl SubAssign for F32x16 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        Self {
            size: self.size,
            elements: unsafe { _mm512_mul_ps(self.elements, rhs.elements) },
        }
    }
}
impl MulAssign for F32x16 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        Self {
            size: self.size,
            elements: unsafe { _mm512_div_ps(self.elements, rhs.elements) },
        }
    }
}
impl DivAssign for F32x16 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        Self {
            size: self.size,
            elements: unsafe { _rem(self.elements, rhs.elements) },
        }
    }
}
impl RemAssign for F32x16 {
    #[inline(always)]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Eq for F32x16 {}

impl PartialEq for F32x16 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        debug_assert_eq!(self.size, other.size, "Operands must have the same size");
        // Compare the first `self.size` elements.
        let mask = unsafe { _mm512_cmpeq_ps_mask(self.elements, other.elements) };

        // Use u32 to prevent overflow on shift
        let active_mask: u16 = ((1u32 << self.size) - 1) as u16;

        (mask & active_mask) == active_mask
    }
}

impl PartialOrd for F32x16 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        assert!(
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
        debug_assert_eq!(self.size, other.size, "Operands must have the same size");

        let mask = unsafe { _mm512_cmplt_ps_mask(self.elements, other.elements) };

        // Use u32 to prevent overflow on shift
        let active_mask: u16 = ((1u32 << self.size) - 1) as u16;

        (mask & active_mask) == active_mask
    }

    #[inline(always)]
    fn le(&self, other: &Self) -> bool {
        debug_assert_eq!(self.size, other.size, "Operands must have the same size");
        let mask = unsafe { _mm512_cmple_ps_mask(self.elements, other.elements) };
        // Use u32 to prevent overflow on shift
        let active_mask: u16 = ((1u32 << self.size) - 1) as u16;

        (mask & active_mask) == active_mask
    }

    #[inline(always)]
    fn gt(&self, other: &Self) -> bool {
        other.lt(self)
    }

    #[inline(always)]
    fn ge(&self, other: &Self) -> bool {
        other.le(self)
    }
}

impl BitAnd for F32x16 {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        Self {
            size: self.size,
            elements: unsafe { _and(self.elements, rhs.elements) },
        }
    }
}
impl BitAndAssign for F32x16 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

impl BitOr for F32x16 {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.size, rhs.size, "Operands must have the same size");
        Self {
            size: self.size,
            elements: unsafe { _or(self.elements, rhs.elements) },
        }
    }
}
impl BitOrAssign for F32x16 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*; // Imports F32x16, LANE_COUNT, AVX_ALIGNMENT, etc.
    use crate::simd::traits::SimdVec; // To call methods from the trait implementation
    use std::panic::{catch_unwind, AssertUnwindSafe};

    // Helper for creating aligned data arrays.
    #[repr(align(64))] // AVX_ALIGNMENT for AVX-512 is 64
    struct AlignedData<const N: usize>([f32; N]);

    impl<const N: usize> AlignedData<N> {
        fn new(val: [f32; N]) -> Self {
            Self(val)
        }
    }

    // Helper to get all 16 f32 elements from F32x16.elements, bypassing size logic.
    fn get_all_elements(v: F32x16) -> [f32; LANE_COUNT] {
        let mut arr = [0.0f32; LANE_COUNT];
        unsafe {
            // Unaligned store is fine for testing purposes.
            _mm512_storeu_ps(arr.as_mut_ptr(), v.elements);
        }
        arr
    }

    // Helper for precise f32 slice comparison (bitwise for NaNs and masks).
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

    // Helper for f32 slice comparison with epsilon, for functions like cos.
    #[cfg(not(miri))]
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
                continue;
            }
            assert!(
                (a[i] - b[i]).abs() <= epsilon,
                "Elements at index {} differ: left={}, right={}, diff={}",
                i,
                a[i],
                b[i],
                (a[i] - b[i]).abs()
            );
        }
    }

    // For AVX-512 comparison masks, true is represented by -1.0 (all bits set).
    const TRUE_MASK_F32: f32 = -1.0f32;
    const FALSE_MASK_F32: f32 = 0.0f32;

    mod simd_vec_impl {
        use super::*;

        #[test]
        fn test_new_full_slice() {
            let data: [f32; LANE_COUNT] = [1.0; LANE_COUNT];
            let v = F32x16::new(&data);
            assert_eq!(v.size, LANE_COUNT);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &data);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }

        #[test]
        fn test_new_larger_slice() {
            let data: [f32; LANE_COUNT + 4] = [2.0; LANE_COUNT + 4];
            let v = F32x16::new(&data);
            assert_eq!(v.size, LANE_COUNT);
            let expected = &data[0..LANE_COUNT];
            assert_f32_slice_eq_bitwise(&get_all_elements(v), expected);
            assert_f32_slice_eq_bitwise(&v.to_vec(), expected);
        }

        #[cfg(not(miri))]
        #[test]
        fn test_new_partial_slice() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0];
            let v = F32x16::new(&data);
            assert_eq!(v.size, data.len());

            let mut expected_raw = [0.0f32; LANE_COUNT];
            expected_raw[0..data.len()].copy_from_slice(&data);

            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected_raw);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }

        #[test]
        #[should_panic(expected = "Size can't be empty (size zero)")]
        fn test_new_empty_slice_panics() {
            F32x16::new(&[]);
        }

        #[test]
        fn test_splat() {
            let val = std::f32::consts::E;
            let v = unsafe { F32x16::splat(val) };
            assert_eq!(v.size, LANE_COUNT);
            let expected = [val; LANE_COUNT];
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &expected);
        }

        #[test]
        fn test_is_aligned() {
            let aligned_arr = AlignedData::new([0.0f32; LANE_COUNT]);
            let unaligned_vec = [0.0f32; LANE_COUNT * 2];
            // Get a pointer that is guaranteed not to be 64-byte aligned
            let unaligned_ptr = unsafe { unaligned_vec.as_ptr().add(1) };

            assert!(
                F32x16::is_aligned(aligned_arr.0.as_ptr()),
                "Aligned pointer reported as unaligned"
            );
            assert!(
                !F32x16::is_aligned(unaligned_ptr),
                "Unaligned pointer reported as aligned"
            );
        }

        #[test]
        fn test_load_unaligned() {
            let data_vec: Vec<f32> = (0..LANE_COUNT + 1).map(|i| i as f32).collect();
            // Create a slice that is likely not 64-byte aligned
            let data_slice = &data_vec[1..=LANE_COUNT];
            let v = unsafe { F32x16::load(data_slice.as_ptr(), LANE_COUNT) };
            assert_eq!(v.size, LANE_COUNT);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), data_slice);
        }

        #[test]
        #[should_panic(expected = "Size must be exactly 16")]
        fn test_load_incorrect_size_panics() {
            let data = [0.0f32; LANE_COUNT];
            unsafe {
                F32x16::load(data.as_ptr(), LANE_COUNT - 1);
            }
        }

        #[cfg(not(miri))]
        #[test]
        fn test_load_partial_various_sizes() {
            let data_full: Vec<f32> = (0..LANE_COUNT).map(|i| i as f32).collect();
            for len in 1..LANE_COUNT {
                let data_slice = &data_full[0..len];
                let v = unsafe { F32x16::load_partial(data_slice.as_ptr(), len) };
                assert_eq!(v.size, len, "Size mismatch for len={len}");

                let mut expected_raw = [0.0f32; LANE_COUNT];
                expected_raw[0..len].copy_from_slice(data_slice);

                assert_f32_slice_eq_bitwise(
                    &get_all_elements(v),
                    &expected_raw,
                    // "Raw elements mismatch for len={}",
                    // len,
                );
                assert_f32_slice_eq_bitwise(
                    &v.to_vec(),
                    data_slice,
                    // "to_vec() output mismatch for len={}",
                    // len,
                );
            }
        }

        #[test]
        #[should_panic(expected = "Size must be less than 16")]
        fn test_load_partial_size_too_large_panics() {
            let data = [0.0f32; LANE_COUNT];
            unsafe {
                F32x16::load_partial(data.as_ptr(), LANE_COUNT);
            }
        }

        #[test]
        fn test_store_at() {
            let data: [f32; LANE_COUNT] = (0..LANE_COUNT)
                .map(|i| i as f32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let v = F32x16::new(&data);
            let mut storage = [0.0f32; LANE_COUNT];
            unsafe {
                v.store_at(storage.as_mut_ptr());
            }
            assert_f32_slice_eq_bitwise(&storage, &data);
        }

        #[cfg(not(miri))]
        #[test]
        fn test_store_at_partial() {
            let data_full: Vec<f32> = (0..LANE_COUNT).map(|i| i as f32).collect();
            for len in 1..LANE_COUNT {
                let data_slice_source = &data_full[0..len];
                let v = F32x16::new(data_slice_source);
                assert_eq!(v.size, len);

                // Fill storage with NANs to ensure masked store only writes where it should.
                let mut storage = [f32::NAN; LANE_COUNT];
                unsafe {
                    v.store_at_partial(storage.as_mut_ptr());
                }

                // Expected: first `len` elements are from the vector, rest are original NANs.
                let mut expected_storage = [f32::NAN; LANE_COUNT];
                expected_storage[0..len].copy_from_slice(data_slice_source);

                assert_f32_slice_eq_bitwise(
                    &storage,
                    &expected_storage,
                    // "Partial store failed for len={}",
                    // len,
                );
            }
        }

        #[cfg(not(miri))]
        #[test]
        fn test_to_vec() {
            // Full vector
            let data_full: [f32; LANE_COUNT] = [5.0; LANE_COUNT];
            let v_full = F32x16::new(&data_full);
            assert_f32_slice_eq_bitwise(&v_full.to_vec(), &data_full);

            // Partial vector
            let data_partial = [1.0, 2.0, 3.0];
            let v_partial = F32x16::new(&data_partial);
            assert_f32_slice_eq_bitwise(&v_partial.to_vec(), &data_partial);
        }

        #[cfg(not(miri))]
        #[test]
        fn test_comparison_elements() {
            let d1 = [
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
            ];
            let d2 = [
                1.0, 3.0, 2.0, 4.0, 0.0, 3.0, 2.0, 5.0, 1.0, 3.0, 2.0, 4.0, 0.0, 3.0, 2.0, 5.0,
            ];
            let v1 = F32x16::new(&d1);
            let v2 = F32x16::new(&d2);

            let t = TRUE_MASK_F32;
            let f = FALSE_MASK_F32;

            // eq_elements
            let eq_res_v = unsafe { v1.eq_elements(v2) };
            assert_eq!(eq_res_v.size, LANE_COUNT);
            let expected_eq = [t, f, f, t, f, f, f, f, t, f, f, t, f, f, f, f];
            assert_f32_slice_eq_bitwise(&eq_res_v.to_vec(), &expected_eq);

            // lt_elements
            let lt_res_v = unsafe { v1.lt_elements(v2) };
            let expected_lt = [f, t, f, f, f, t, f, t, f, t, f, f, f, t, f, t];
            assert_f32_slice_eq_bitwise(&lt_res_v.to_vec(), &expected_lt);

            // le_elements
            let le_res_v = unsafe { v1.le_elements(v2) };
            let expected_le = [t, t, f, t, f, t, f, t, t, t, f, t, f, t, f, t];
            assert_f32_slice_eq_bitwise(&le_res_v.to_vec(), &expected_le);

            // gt_elements
            let gt_res_v = unsafe { v1.gt_elements(v2) };
            let expected_gt = [f, f, t, f, t, f, t, f, f, f, t, f, t, f, t, f];
            assert_f32_slice_eq_bitwise(&gt_res_v.to_vec(), &expected_gt);

            // ge_elements
            let ge_res_v = unsafe { v1.ge_elements(v2) };
            let expected_ge = [t, f, t, t, t, f, t, f, t, f, t, t, t, f, t, f];
            assert_f32_slice_eq_bitwise(&ge_res_v.to_vec(), &expected_ge);
        }

        #[test]
        #[should_panic(expected = "Operands must have the same size")]
        fn test_comparison_elements_panic_on_diff_size() {
            let v1 = F32x16::new(&[1.0; LANE_COUNT]);
            let mut v2 = F32x16::new(&[2.0; LANE_COUNT]);
            v2.size = LANE_COUNT - 1; // Manually set different size.
            let _ = unsafe { v1.eq_elements(v2) };
        }

        #[cfg(not(miri))]
        #[test]
        fn test_cos() {
            use std::f32::consts::{FRAC_PI_2, PI};
            let mut data_in = [0.0f32; LANE_COUNT];
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
            data_in[0..inputs.len()].copy_from_slice(&inputs);

            let v_in = F32x16::new(&data_in);
            let v_out = unsafe { v_in.cos() };

            let mut expected_outputs = [1.0f32; LANE_COUNT];
            expected_outputs[0] = 1.0; // cos(0)
            expected_outputs[1] = 0.0; // cos(pi/2)
            expected_outputs[2] = -1.0; // cos(pi)
            expected_outputs[3] = 0.0; // cos(3pi/2)
            expected_outputs[4] = 1.0; // cos(2pi)
            expected_outputs[5] = f32::NAN; // cos(NaN)
            expected_outputs[6] = f32::NAN; // cos(Inf)
            expected_outputs[7] = f32::NAN; // cos(-Inf)

            let out_vec = v_out.to_vec();
            assert_f32_slice_eq_epsilon(&out_vec, &expected_outputs, 1e-7);
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

            let v_in = F32x16::new(&data_in);
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
        fn test_acos() {
            const LANE_COUNT: usize = 16;

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
            let v_in = F32x16::new(&data_in);
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

        #[test]
        fn test_asin() {
            const LANE_COUNT: usize = 16;

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
            let v_in = F32x16::new(&data_in);
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
    }

    // --- `operator_overloads` tests ---
    mod operator_overloads {
        use super::*;

        fn setup_vecs() -> (F32x16, F32x16, F32x16) {
            let d1: [f32; 16] = (0..16)
                .map(|x| x as f32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let d2: [f32; 16] = (0..16)
                .map(|x| (30.0 - 2.0 * x as f32))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            let d3: [f32; 16] = [2.0; 16];
            (F32x16::new(&d1), F32x16::new(&d2), F32x16::new(&d3))
        }

        #[cfg(not(miri))]
        #[test]
        fn test_add_sub_mul_div_rem() {
            let (v1, v2, v_two) = setup_vecs();

            let add_res = v1 + v_two;
            let expected_add: [f32; 16] = (0..16)
                .map(|x| (x + 2) as f32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            assert_f32_slice_eq_bitwise(&add_res.to_vec(), &expected_add);

            let sub_res = v2 - v_two;
            let expected_sub: [f32; 16] = (0..16)
                .map(|x| (28.0 - 2.0 * x as f32))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            assert_f32_slice_eq_bitwise(&sub_res.to_vec(), &expected_sub);

            let mul_res = v1 * v_two;
            let expected_mul: [f32; 16] = (0..16)
                .map(|x| (2 * x) as f32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            assert_f32_slice_eq_bitwise(&mul_res.to_vec(), &expected_mul);

            let div_res = v1 / v_two;
            let expected_div: [f32; 16] = (0..16)
                .map(|x| x as f32 / 2.0)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            assert_f32_slice_eq_bitwise(&div_res.to_vec(), &expected_div);

            let rem_res = v1 % v_two;
            let expected_rem: [f32; 16] = (0..16)
                .map(|x| x as f32 % 2.0)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            assert_f32_slice_eq_bitwise(&rem_res.to_vec(), &expected_rem);
        }

        #[test]
        fn test_assign_ops() {
            let (v1_orig, _v2, v_two) = setup_vecs();

            let mut v1 = v1_orig;
            v1 += v_two;
            let expected_add: [f32; 16] = (0..16)
                .map(|x| (x + 2) as f32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            assert_f32_slice_eq_bitwise(&v1.to_vec(), &expected_add);

            let mut v1 = v1_orig;
            v1 -= v_two;
            let expected_sub: [f32; 16] = (0..16)
                .map(|x| (x - 2) as f32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            assert_f32_slice_eq_bitwise(&v1.to_vec(), &expected_sub);

            let mut v1 = v1_orig;
            v1 *= v_two;
            let expected_mul: [f32; 16] = (0..16)
                .map(|x| (x * 2) as f32)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            assert_f32_slice_eq_bitwise(&v1.to_vec(), &expected_mul);
        }

        #[test]
        fn test_div_by_zero() {
            let v_ones = unsafe { F32x16::splat(1.0) };
            let v_zeros = unsafe { F32x16::splat(0.0) };
            let res = v_ones / v_zeros;
            for &x in &res.to_vec() {
                assert!(
                    x.is_infinite() && x.is_sign_positive(),
                    "Expected positive infinity"
                );
            }
        }
        #[cfg(not(miri))]
        #[test]
        fn test_rem_by_zero() {
            let v_ones = unsafe { F32x16::splat(1.0) };
            let v_zeros = unsafe { F32x16::splat(0.0) };
            let res = v_ones % v_zeros;
            for &x in &res.to_vec() {
                assert!(x.is_nan(), "Expected NaN");
            }
        }

        #[cfg(not(miri))]
        #[test]
        fn test_partial_eq() {
            let v1 = unsafe { F32x16::splat(5.0) };
            let v2 = unsafe { F32x16::splat(5.0) };
            let v3 = unsafe { F32x16::splat(6.0) };
            assert_eq!(v1, v2);
            assert_ne!(v1, v3);

            let p_data1 = &[1.0, 2.0, 3.0];
            let vp1 = F32x16::new(p_data1); // size=3
            let vp2 = F32x16::new(p_data1); // size=3
            assert_eq!(vp1, vp2);
        }

        #[cfg(not(miri))]
        #[test]
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        fn test_partial_ord() {
            let v_small = unsafe { F32x16::splat(1.0) };
            let v_large = unsafe { F32x16::splat(2.0) };

            assert!(v_small < v_large);
            assert!(v_small <= v_large);
            assert!(v_large > v_small);
            assert!(v_large >= v_small);
            assert!(!(v_large < v_small));
            assert_eq!(
                v_small.partial_cmp(&v_large),
                Some(std::cmp::Ordering::Less)
            );
            assert_eq!(
                v_large.partial_cmp(&v_small),
                Some(std::cmp::Ordering::Greater)
            );
            assert_eq!(
                v_small.partial_cmp(&v_small),
                Some(std::cmp::Ordering::Equal)
            );

            let v_mixed = F32x16::new(&[
                1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 15.0, 14.0,
                16.0,
            ]);
            assert_eq!(v_small.partial_cmp(&v_mixed), None);
        }

        #[test]
        fn test_bitwise_ops() {
            // Bitwise ops on floats are uncommon unless using their bit patterns as masks.
            // Values are typically all-bits-one (like TRUE_MASK_F32) or all-bits-zero.
            let v_true = unsafe { F32x16::splat(TRUE_MASK_F32) };
            let v_false = unsafe { F32x16::splat(FALSE_MASK_F32) };

            let data_pattern_arr = [
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
                TRUE_MASK_F32,
                FALSE_MASK_F32,
            ];

            let v_pattern = F32x16::new(&data_pattern_arr); // 10101010 pattern

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
            let v1 = F32x16::new(&[1.0; 16]);
            let mut v2 = F32x16::new(&[2.0; 16]);
            v2.size = 15; // Mismatched size

            macro_rules! check_panic {
                ($op:expr, $op_name:expr) => {
                    let result = catch_unwind(AssertUnwindSafe(|| $op));
                    assert!(result.is_err(), "Operation '{}' did not panic", $op_name);
                };
            }

            check_panic!(v1 + v2, "+");
            check_panic!(v1 - v2, "-");
            check_panic!(v1 * v2, "*");
            check_panic!(v1 / v2, "/");
            check_panic!(v1 % v2, "%");
            check_panic!(v1 == v2, "==");
            check_panic!(v1 < v2, "<");
        }
    }
}
