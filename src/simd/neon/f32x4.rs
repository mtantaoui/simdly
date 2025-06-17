#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::simd::{neon::cos::f32::vcosq_f32, traits::SimdVec};
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Rem,
    RemAssign, Sub, SubAssign,
};

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

    #[inline(always)]
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

    #[inline(always)]
    unsafe fn store_in_vec(&self) -> Vec<f32> {
        assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );

        let mut vec = Vec::with_capacity(LANE_COUNT);

        unsafe {
            vst1q_f32(vec.as_mut_ptr(), self.elements);
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
            self.size < LANE_COUNT,
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
            _ => unreachable!("Size must be < LANE_COUNT"),
        }
    }

    #[inline(always)]
    fn to_vec(self) -> Vec<f32> {
        assert!(
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

    #[inline(always)]
    unsafe fn eq_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a == b elementwise
        let mask = unsafe { vceqq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    #[inline(always)]
    unsafe fn lt_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a<b elementwise
        let mask = unsafe { vcltq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    #[inline(always)]
    unsafe fn le_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a<=b elementwise
        let mask = unsafe { vcleq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    #[inline(always)]
    unsafe fn gt_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a>b elementwise
        let mask = unsafe { vcgtq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    #[inline(always)]
    unsafe fn ge_elements(&self, rhs: Self) -> Self {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Compare a>=b elementwise
        let mask = unsafe { vcgeq_f32(self.elements, rhs.elements) }; // Result as float mask

        let elements = unsafe { vreinterpretq_f32_u32(mask) };

        Self {
            elements,
            size: self.size,
        }
    }

    unsafe fn cos(&self) -> Self {
        Self {
            elements: vcosq_f32(self.elements),
            size: self.size,
        }
    }

    unsafe fn abs(&self) -> Self {
        todo!()
    }

    unsafe fn acos(&self) -> Self {
        todo!()
    }

    unsafe fn asin(&self) -> Self {
        todo!()
    }
}

/// Implementing the `Add` and `AddAssign` traits for F32x4
/// This allows for using the `+` operator and `+=` operator with F32x4 vectors.    
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

impl AddAssign for F32x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for F32x4 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            F32x4 {
                size: self.size,
                elements: vsubq_f32(self.elements, rhs.elements),
            }
        }
    }
}

impl SubAssign for F32x4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self - rhs;
    }
}

impl Mul for F32x4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            F32x4 {
                size: self.size,
                elements: vmulq_f32(self.elements, rhs.elements),
            }
        }
    }
}

impl MulAssign for F32x4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self * rhs;
    }
}

impl Div for F32x4 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        unsafe {
            F32x4 {
                size: self.size,
                elements: vdivq_f32(self.elements, rhs.elements),
            }
        }
    }
}

impl DivAssign for F32x4 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self / rhs;
    }
}

impl Rem for F32x4 {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        #[cfg(not(miri))]
        unsafe {
            let div = vdivq_f32(self.elements, rhs.elements);
            let floor = vrndq_f32(div);
            let prod = vmulq_f32(floor, rhs.elements);

            let elements = vsubq_f32(self.elements, prod);

            F32x4 {
                size: self.size,
                elements,
            }
        }

        #[cfg(miri)]
        unsafe {
            // Miri-specific fallback for Rem
            // vrndq_f32 is not supported by Miri. We'll implement element-wise rounding.
            // Note: f32::round() in Rust rounds half to even, which matches vrndq_f32's
            // default behavior on AArch64 (when FPCR.RMode is RNE).

            let mut self_arr = [0.0f32; LANE_COUNT];
            let mut rhs_arr = [0.0f32; LANE_COUNT];
            vst1q_f32(self_arr.as_mut_ptr(), self.elements);
            vst1q_f32(rhs_arr.as_mut_ptr(), rhs.elements);

            let mut result_arr = [0.0f32; LANE_COUNT];
            for i in 0..LANE_COUNT {
                if rhs_arr[i] == 0.0 {
                    result_arr[i] = f32::NAN; // Consistent with x % 0 = NaN
                } else {
                    let div_val = self_arr[i] / rhs_arr[i];
                    // f32::round() rounds to the nearest integer. Ties are rounded to the even integer.
                    // This matches the default behavior of vrndq_f32 (Round to Nearest, ties to Even).
                    let rounded_div_val = div_val.round();
                    result_arr[i] = self_arr[i] - rounded_div_val * rhs_arr[i];
                }
            }

            F32x4 {
                size: self.size,
                elements: vld1q_f32(result_arr.as_ptr()),
            }
        }
    }
}

impl RemAssign for F32x4 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        *self = *self % rhs;
    }
}
impl Eq for F32x4 {}

impl PartialEq for F32x4 {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        // SimdVec trait methods already assert size equality for operations they call.
        // However, PartialEq is a standalone trait, so asserting size here is good practice
        // if F32x4 instances with different `size` fields but same LANE_COUNT could exist
        // and be compared. Given `new()`'s behavior, `self.size` should ideally reflect
        // the number of *meaningful* elements.
        // For PartialEq on SIMD types, it's conventional to compare all underlying hardware lanes.
        // If `self.size` could be < LANE_COUNT and you only wanted to compare `self.size`
        // elements, the logic would be more complex and involve masking.
        // Assuming we always compare all hardware lanes for PartialEq:
        assert!(
            self.size == other.size || (self.size == LANE_COUNT && other.size == LANE_COUNT),
            "Operands for PartialEq should ideally have the same active size or both be full ({} lanes), got {} and {}",
            LANE_COUNT,
            self.size,
            other.size
        );

        unsafe {
            // cmp is uint32x4_t where true lanes are 0xFFFFFFFF, false lanes are 0x0
            let cmp_mask: uint32x4_t = vceqq_u32(
                vreinterpretq_u32_f32(self.elements),
                vreinterpretq_u32_f32(other.elements),
            );

            // Check if all lanes are true (0xFFFFFFFF)
            // vminvq_u32 returns the smallest u32 value among the lanes.
            // If all lanes are 0xFFFFFFFF, the min is 0xFFFFFFFF.
            // Otherwise, if any lane is 0x0, the min will be 0x0.
            let all_lanes_true: u32 = vminvq_u32(cmp_mask);
            all_lanes_true == 0xFFFFFFFF
        }
    }
}

impl PartialOrd for F32x4 {
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
            // Get the raw uint32x4_t masks (0xFFFFFFFF for true, 0x0 for false)
            let lt_mask_u32: uint32x4_t = vcltq_f32(self.elements, other.elements);
            let gt_mask_u32: uint32x4_t = vcgtq_f32(self.elements, other.elements);

            // Check if all lanes satisfy the condition
            let all_lt = vminvq_u32(lt_mask_u32) == 0xFFFFFFFF;
            let all_gt = vminvq_u32(gt_mask_u32) == 0xFFFFFFFF;

            match (all_lt, all_gt) {
                (true, false) => Some(std::cmp::Ordering::Less), // All elements are less than
                (false, true) => Some(std::cmp::Ordering::Greater), // All elements are greater than
                (false, false) => {
                    // Not all < and not all >. Now check if all are equal.
                    let eq_mask_u32: uint32x4_t = vceqq_f32(self.elements, other.elements);
                    if vminvq_u32(eq_mask_u32) == 0xFFFFFFFF {
                        Some(std::cmp::Ordering::Equal) // All elements are equal
                    } else {
                        None // Elements are mixed (some <, some >, some ==, or NaNs involved)
                    }
                }
                (true, true) => {
                    // This case (all_lt is true AND all_gt is true) should be logically impossible
                    // for non-NaN f32 values. If x < y and x > y for all elements, it's a contradiction.
                    // This could only happen if NaNs are involved in a way that tricks the comparison,
                    // or if the SIMD operations themselves have unusual behavior with specific bit patterns
                    // that aren't true NaNs but still satisfy both < and > (highly unlikely for standard ops).
                    // Standard float comparisons with NaN result in false for <, >, <=, >=, ==.
                    // Thus, vcltq_f32(NaN, x) would be all false, vminvq_u32 -> 0.
                    // So, (true, true) should ideally not be reachable with IEEE 754 floats.
                    // We can treat this as a "mixed" or "unordered" case.
                    None
                }
            }
        }
    }

    #[inline(always)]
    fn lt(&self, other: &Self) -> bool {
        assert!(self.size == other.size, "Size mismatch for lt");
        unsafe {
            let cmp_mask: uint32x4_t = vcltq_f32(self.elements, other.elements);
            vminvq_u32(cmp_mask) == 0xFFFFFFFF
        }
    }

    #[inline(always)]
    fn le(&self, other: &Self) -> bool {
        assert!(self.size == other.size, "Size mismatch for le");
        unsafe {
            let cmp_mask: uint32x4_t = vcleq_f32(self.elements, other.elements);
            vminvq_u32(cmp_mask) == 0xFFFFFFFF
        }
    }

    #[inline(always)]
    fn gt(&self, other: &Self) -> bool {
        assert!(self.size == other.size, "Size mismatch for gt");
        unsafe {
            let cmp_mask: uint32x4_t = vcgtq_f32(self.elements, other.elements);
            vminvq_u32(cmp_mask) == 0xFFFFFFFF
        }
    }

    #[inline(always)]
    fn ge(&self, other: &Self) -> bool {
        assert!(self.size == other.size, "Size mismatch for ge");
        unsafe {
            let cmp_mask: uint32x4_t = vcgeq_f32(self.elements, other.elements);
            vminvq_u32(cmp_mask) == 0xFFFFFFFF
        }
    }
}

impl BitAnd for F32x4 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );

        // Convert float32x4_t to uint32x4_t
        let self_u32: uint32x4_t = unsafe { vreinterpretq_u32_f32(self.elements) }; // Reinterpret as uint32x4_t
        let rhs_u32: uint32x4_t = unsafe { vreinterpretq_u32_f32(rhs.elements) }; // Reinterpret as uint32x4_t

        // Perform bitwise AND between the two uint32x4_t vectors
        let result: uint32x4_t = unsafe { vandq_u32(self_u32, rhs_u32) };

        let elements: float32x4_t = unsafe { vreinterpretq_f32_u32(result) };

        F32x4 {
            size: self.size,
            elements,
        }
    }
}

impl BitAndAssign for F32x4 {
    #[inline]
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

impl BitOr for F32x4 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        assert!(
            self.size == rhs.size,
            "Operands must have the same size (expected {} lanes, got {} and {})",
            LANE_COUNT,
            self.size,
            rhs.size
        );
        // Convert float32x4_t to uint32x4_t
        let self_u32: uint32x4_t = unsafe { vreinterpretq_u32_f32(self.elements) }; // Reinterpret as uint32x4_t
        let rhs_u32: uint32x4_t = unsafe { vreinterpretq_u32_f32(rhs.elements) }; // Reinterpret as uint32x4_t

        // Perform bitwise AND between the two uint32x4_t vectors
        let result: uint32x4_t = unsafe { vorrq_u32(self_u32, rhs_u32) };

        let elements: float32x4_t = unsafe { vreinterpretq_f32_u32(result) };

        F32x4 {
            size: self.size,
            elements,
        }
    }
}

impl BitOrAssign for F32x4 {
    #[inline]
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
#[cfg(target_arch = "aarch64")] // NEON tests only make sense on aarch64
mod tests {
    // ... (all previous helper functions like AlignedNeonData, get_all_elements, etc. remain)
    use super::*;
    use crate::simd::traits::SimdVec;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::ptr;

    #[repr(align(16))]
    struct AlignedNeonData<const N: usize>([f32; N]);

    impl<const N: usize> AlignedNeonData<N> {
        fn new(val: [f32; N]) -> Self {
            Self(val)
        }
    }

    fn get_all_elements(v: F32x4) -> [f32; LANE_COUNT] {
        let mut arr = [0.0f32; LANE_COUNT];
        unsafe {
            vst1q_f32(arr.as_mut_ptr(), v.elements);
        }
        arr
    }

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
    const TRUE_MASK_F32: f32 = f32::from_bits(0xFFFFFFFFu32);
    const FALSE_MASK_F32: f32 = 0.0f32;

    mod simd_vec_impl {
        use core::f32;

        use super::*;
        // ... (all other tests from simd_vec_impl should remain here)
        #[test]
        fn test_new_full_slice() {
            let data = [1.0, 2.0, 3.0, 4.0];
            let v = F32x4::new(&data);
            assert_eq!(v.size, LANE_COUNT);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &data);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }

        #[test]
        fn test_new_larger_slice() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let v = F32x4::new(&data);
            assert_eq!(v.size, LANE_COUNT);
            let expected = &data[0..LANE_COUNT];
            assert_f32_slice_eq_bitwise(&get_all_elements(v), expected);
            assert_f32_slice_eq_bitwise(&v.to_vec(), expected);
        }

        #[test]
        fn test_new_partial_slice_len_1() {
            let data = [1.0];
            let v = F32x4::new(&data);
            assert_eq!(v.size, 1);
            let mut expected_raw = [0.0f32; LANE_COUNT];
            expected_raw[0] = 1.0;
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected_raw);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }

        #[test]
        fn test_new_partial_slice_len_2() {
            let data = [1.0, 2.0];
            let v = F32x4::new(&data);
            assert_eq!(v.size, data.len());
            let mut expected_raw = [0.0f32; LANE_COUNT];
            expected_raw[0..data.len()].copy_from_slice(&data);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected_raw);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }
        #[test]
        fn test_new_partial_slice_len_3() {
            let data = [1.0, 2.0, 3.0];
            let v = F32x4::new(&data);
            assert_eq!(v.size, 3);
            let mut expected_raw = [0.0f32; LANE_COUNT];
            expected_raw[0..3].copy_from_slice(&data);
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected_raw);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &data);
        }

        #[test]
        #[should_panic(expected = "Size can't be empty (size zero)")]
        fn test_new_empty_slice_panics() {
            F32x4::new(&[]);
        }

        #[test]
        fn test_splat() {
            let val = f32::consts::PI;
            let v = unsafe { F32x4::splat(val) };
            assert_eq!(v.size, LANE_COUNT);
            let expected = [val; LANE_COUNT];
            assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected);
            assert_f32_slice_eq_bitwise(&v.to_vec(), &expected);
        }

        #[allow(clippy::needless_range_loop)]
        #[test]
        fn test_splat_nan() {
            let val = f32::NAN;
            let v = unsafe { F32x4::splat(val) };
            assert_eq!(v.size, LANE_COUNT);
            let elements = get_all_elements(v);
            for i in 0..LANE_COUNT {
                assert!(elements[i].is_nan(), "Element {i} should be NaN");
            }
        }

        #[test]
        #[should_panic] // is_aligned always panics with unreachable!()
        fn test_is_aligned_panics() {
            let data = [0.0f32; LANE_COUNT];
            F32x4::is_aligned(data.as_ptr());
        }

        #[test]
        fn test_load() {
            let data_aligned = AlignedNeonData::new([1.1, 2.2, 3.3, 4.4]);
            let v_aligned = unsafe { F32x4::load(data_aligned.0.as_ptr(), LANE_COUNT) };
            assert_eq!(v_aligned.size, LANE_COUNT);
            assert_f32_slice_eq_bitwise(&get_all_elements(v_aligned), &data_aligned.0);

            let data_vec: Vec<f32> = vec![0.0, 5.5, 6.6, 7.7, 8.8, 0.0];
            let data_slice = &data_vec[1..(LANE_COUNT + 1)];
            assert_eq!(data_slice.len(), LANE_COUNT);

            let v_unaligned = unsafe { F32x4::load(data_slice.as_ptr(), LANE_COUNT) };
            assert_eq!(v_unaligned.size, LANE_COUNT);
            assert_f32_slice_eq_bitwise(&get_all_elements(v_unaligned), data_slice);
        }

        #[test]
        #[should_panic(expected = "Pointer must not be null")]
        fn test_load_null_ptr_panics() {
            unsafe {
                F32x4::load(ptr::null(), LANE_COUNT);
            }
        }

        #[test]
        #[should_panic(expected = "Size must be == 4")]
        fn test_load_incorrect_size_panics() {
            let data = [0.0f32; LANE_COUNT];
            unsafe {
                F32x4::load(data.as_ptr(), LANE_COUNT - 1);
            }
        }

        #[test]
        #[should_panic]
        fn test_load_aligned_panics() {
            let data = AlignedNeonData::new([0.0f32; LANE_COUNT]);
            unsafe {
                F32x4::load_aligned(data.0.as_ptr(), LANE_COUNT);
            }
        }

        #[test]
        #[should_panic]
        fn test_load_unaligned_panics() {
            let data_vec: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
            let data_slice = &data_vec[1..];
            unsafe {
                F32x4::load_unaligned(data_slice.as_ptr(), LANE_COUNT);
            }
        }

        #[test]
        fn test_load_partial_various_sizes() {
            let data_full = [1.0, 2.0, 3.0, 4.0];
            for len in 1..LANE_COUNT {
                let data_slice = &data_full[0..len];
                let v = unsafe { F32x4::load_partial(data_slice.as_ptr(), len) };
                assert_eq!(v.size, len, "Size mismatch for len={len}");

                let mut expected_raw = [0.0f32; LANE_COUNT];
                expected_raw[0..len].copy_from_slice(data_slice);

                assert_f32_slice_eq_bitwise(&get_all_elements(v), &expected_raw);
                assert_f32_slice_eq_bitwise(&v.to_vec(), data_slice);
            }
        }

        #[test]
        #[should_panic(expected = "Size must be < 4")]
        fn test_load_partial_size_too_large_panics() {
            let data = [0.0f32; LANE_COUNT];
            unsafe {
                F32x4::load_partial(data.as_ptr(), LANE_COUNT);
            }
        }

        #[test]
        #[should_panic(expected = "Size must be < 4")]
        fn test_load_partial_size_zero_panics_due_to_unreachable() {
            let data = [0.0f32; 1];
            unsafe {
                F32x4::load_partial(data.as_ptr(), 0);
            }
        }

        #[test]
        #[should_panic(expected = "Pointer must not be null")]
        fn test_load_partial_null_ptr_panics() {
            unsafe {
                F32x4::load_partial(ptr::null(), 1);
            }
        }

        #[test]
        fn test_store_in_vec_full() {
            let data = [1.0, 2.0, 3.0, 4.0];
            let v = F32x4::new(&data);
            let stored_vec = unsafe { v.store_in_vec() };
            assert_eq!(stored_vec.len(), LANE_COUNT);
            assert_f32_slice_eq_bitwise(&stored_vec, &data);
        }

        #[test]
        fn test_store_in_vec_partial_source_size() {
            let data_partial = [1.0, 2.0];
            let v = F32x4::new(&data_partial);

            let mut expected_full_from_m256 = [0.0f32; LANE_COUNT];
            expected_full_from_m256[0..2].copy_from_slice(&data_partial);

            let stored_vec = unsafe { v.store_in_vec() };
            assert_eq!(stored_vec.len(), LANE_COUNT);
            assert_f32_slice_eq_bitwise(&stored_vec, &expected_full_from_m256);
        }

        #[test]
        #[should_panic(expected = "Size must be <= 4")]
        fn test_store_in_vec_invalid_size_panics() {
            let mut v = unsafe { F32x4::splat(1.0) };
            v.size = LANE_COUNT + 1;
            let _ = unsafe { v.store_in_vec() };
        }

        #[test]
        fn test_store_in_vec_partial_method() {
            let data_full = [1.0, 2.0, 3.0, 4.0];
            for len in 1..LANE_COUNT {
                let data_slice = &data_full[0..len];
                let v = F32x4::new(data_slice);
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
            let v = F32x4::new(&data);
            assert_eq!(v.size, LANE_COUNT);
            unsafe { v.store_in_vec_partial() };
        }

        #[test]
        fn test_store_at() {
            let data = [1.0, 2.0, 3.0, 4.0];
            let v = F32x4::new(&data);

            let mut aligned_storage = AlignedNeonData::new([0.0f32; LANE_COUNT]);
            unsafe {
                v.store_at(aligned_storage.0.as_mut_ptr());
            }
            assert_f32_slice_eq_bitwise(&aligned_storage.0, &data);

            let mut unaligned_storage_vec = vec![0.0f32; LANE_COUNT + 1];
            let ptr_unaligned = unsafe { unaligned_storage_vec.as_mut_ptr().add(1) };

            unsafe {
                v.store_at(ptr_unaligned);
            }
            let target_slice_after_store =
                unsafe { std::slice::from_raw_parts(ptr_unaligned as *const f32, LANE_COUNT) };
            assert_f32_slice_eq_bitwise(target_slice_after_store, &data);
        }

        #[test]
        #[should_panic(expected = "Pointer must not be null")]
        fn test_store_at_null_ptr_panics() {
            let v = unsafe { F32x4::splat(1.0) };
            unsafe {
                v.store_at(ptr::null_mut());
            }
        }

        #[test]
        #[should_panic(expected = "Size must be <= 4")]
        fn test_store_at_invalid_size_panics() {
            let mut v = unsafe { F32x4::splat(1.0) };
            v.size = LANE_COUNT + 1;
            let mut storage = [0.0f32; LANE_COUNT];
            unsafe {
                v.store_at(storage.as_mut_ptr());
            }
        }

        #[allow(clippy::manual_memcpy)]
        #[test]
        fn test_store_at_partial() {
            let data_full = [1.0, 2.0, 3.0, 4.0];
            for len in 1..LANE_COUNT {
                let source_slice_for_f32x4 = &data_full[0..len];
                let v = F32x4::new(source_slice_for_f32x4);
                assert_eq!(v.size, len);

                let mut storage_array = [f32::NAN; LANE_COUNT];
                unsafe {
                    v.store_at_partial(storage_array.as_mut_ptr());
                }

                let mut expected_in_storage = [f32::NAN; LANE_COUNT];
                let v_elements_raw = get_all_elements(v);
                for i in 0..len {
                    expected_in_storage[i] = v_elements_raw[i];
                }
                assert_f32_slice_eq_bitwise(&storage_array, &expected_in_storage);
            }
        }

        #[test]
        #[should_panic(expected = "Size must be < 4")]
        fn test_store_at_partial_size_lane_count_panics() {
            let data = [1.0; LANE_COUNT];
            let v = F32x4::new(&data);
            let mut storage = [0.0; LANE_COUNT];
            unsafe {
                v.store_at_partial(storage.as_mut_ptr());
            }
        }

        #[test]
        #[should_panic(expected = "Pointer must not be null")]
        fn test_store_at_partial_null_ptr_panics() {
            let v = F32x4::new(&[1.0]);
            unsafe {
                v.store_at_partial(ptr::null_mut());
            }
        }

        #[test]
        #[should_panic(expected = "Size must be < LANE_COUNT")]
        fn test_store_at_partial_size_zero_panics() {
            let mut v = unsafe { F32x4::splat(0.0) };
            v.size = 0;
            let mut storage = [0.0; LANE_COUNT];
            unsafe {
                v.store_at_partial(storage.as_mut_ptr());
            }
        }

        #[test]
        fn test_to_vec() {
            let data_full = [1.0, 2.0, 3.0, 4.0];
            let v_full = F32x4::new(&data_full);
            assert_f32_slice_eq_bitwise(&v_full.to_vec(), &data_full);

            let data_partial = [1.0, 2.0];
            let v_partial = F32x4::new(&data_partial);
            assert_f32_slice_eq_bitwise(&v_partial.to_vec(), &data_partial);

            let data_partial_1 = [5.0];
            let v_partial_1 = F32x4::new(&data_partial_1);
            assert_f32_slice_eq_bitwise(&v_partial_1.to_vec(), &data_partial_1);
        }

        #[test]
        #[should_panic(expected = "Size must be <= 4")]
        fn test_to_vec_invalid_size_panics() {
            let mut v = unsafe { F32x4::splat(1.0) };
            v.size = LANE_COUNT + 1;
            let _ = v.to_vec();
        }

        #[test]
        fn test_comparison_elements() {
            let d1 = [1.0, 2.0, 3.0, 4.0];
            let d2 = [1.0, 3.0, 2.0, 4.0];
            let v1 = F32x4::new(&d1);
            let v2 = F32x4::new(&d2);

            let tm = TRUE_MASK_F32;
            let fm = FALSE_MASK_F32;

            let eq_res_v = unsafe { v1.eq_elements(v2) };
            assert_eq!(eq_res_v.size, LANE_COUNT);
            let expected_eq_raw = [tm, fm, fm, tm];
            assert_f32_slice_eq_bitwise(&get_all_elements(eq_res_v), &expected_eq_raw);

            let lt_res_v = unsafe { v1.lt_elements(v2) };
            let expected_lt_raw = [fm, tm, fm, fm];
            assert_f32_slice_eq_bitwise(&get_all_elements(lt_res_v), &expected_lt_raw);

            let le_res_v = unsafe { v1.le_elements(v2) };
            let expected_le_raw = [tm, tm, fm, tm];
            assert_f32_slice_eq_bitwise(&get_all_elements(le_res_v), &expected_le_raw);

            let gt_res_v = unsafe { v1.gt_elements(v2) };
            let expected_gt_raw = [fm, fm, tm, fm];
            assert_f32_slice_eq_bitwise(&get_all_elements(gt_res_v), &expected_gt_raw);

            let ge_res_v = unsafe { v1.ge_elements(v2) };
            let expected_ge_raw = [tm, fm, tm, tm];
            assert_f32_slice_eq_bitwise(&get_all_elements(ge_res_v), &expected_ge_raw);
        }

        #[test]
        #[should_panic(expected = "Operands must have the same size")]
        fn test_comparison_elements_panic_on_diff_size() {
            let v1 = F32x4::new(&[1.0; LANE_COUNT]);
            let v2_elems = [2.0; LANE_COUNT - 1];
            let v2 = F32x4::new(&v2_elems);
            let _ = unsafe { v1.eq_elements(v2) };
        }

        #[cfg(not(miri))]
        #[test]
        fn test_cos() {
            use std::f32::consts::{FRAC_PI_2, PI};
            let inputs = [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2];
            let v_in = F32x4::new(&inputs);
            let v_out = unsafe { v_in.cos() };

            let expected_outputs_approx = [1.0, 0.0, -1.0, 0.0];
            let out_vec = v_out.to_vec();
            assert_f32_slice_eq_epsilon(&out_vec, &expected_outputs_approx, 1e-4);

            let special_inputs = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0 * PI];
            let v_special_in = F32x4::new(&special_inputs);
            let v_special_out = unsafe { v_special_in.cos() };
            let special_out_vec = v_special_out.to_vec();

            assert!(special_out_vec[0].is_nan(), "cos(NaN) should be NaN");
            assert!(special_out_vec[1].is_nan(), "cos(INF) should be NaN");
            assert!(special_out_vec[2].is_nan(), "cos(-INF) should be NaN");
            assert!((special_out_vec[3] - 1.0).abs() < 1e-4, "cos(2PI) check");
        }
    }

    mod operator_overloads {
        use super::*;

        #[cfg(not(miri))]
        fn setup_vecs() -> (F32x4, F32x4, F32x4) {
            let d1 = [1.0, 2.0, 3.0, 4.0];
            let d2 = [4.0, 3.0, 2.0, 1.0];
            let d3 = [2.0, 2.0, 2.0, 2.0];
            (F32x4::new(&d1), F32x4::new(&d2), F32x4::new(&d3))
        }

        #[cfg(not(miri))]
        #[test]
        fn test_arithmetic_ops() {
            let (v1, v2, v_operand) = setup_vecs();

            let add_res = v1 + v2;
            let expected_add = [5.0, 5.0, 5.0, 5.0];
            assert_f32_slice_eq_bitwise(&add_res.to_vec(), &expected_add);

            let sub_res = v1 - v_operand;
            let expected_sub = [-1.0, 0.0, 1.0, 2.0];
            assert_f32_slice_eq_bitwise(&sub_res.to_vec(), &expected_sub);

            let mul_res = v1 * v_operand;
            let expected_mul = [2.0, 4.0, 6.0, 8.0];
            assert_f32_slice_eq_bitwise(&mul_res.to_vec(), &expected_mul);

            let div_res = v1 / v_operand;
            let expected_div = [0.5, 1.0, 1.5, 2.0];
            assert_f32_slice_eq_bitwise(&div_res.to_vec(), &expected_div);

            let rem_res = v1 % v_operand;
            let expected_rem = [1.0, 0.0, 1.0, 0.0];
            assert_f32_slice_eq_bitwise(&rem_res.to_vec(), &expected_rem);
        }

        #[cfg(not(miri))]
        #[test]
        fn test_assign_ops() {
            let (v1_orig, v2, v_operand_orig) = setup_vecs();

            let mut v1_mut = v1_orig;
            v1_mut += v2;
            assert_f32_slice_eq_bitwise(&v1_mut.to_vec(), &[5.0; LANE_COUNT]);

            v1_mut = v1_orig;
            v1_mut -= v_operand_orig;
            assert_f32_slice_eq_bitwise(&v1_mut.to_vec(), &[-1.0, 0.0, 1.0, 2.0]);

            v1_mut = v1_orig;
            v1_mut *= v_operand_orig;
            assert_f32_slice_eq_bitwise(&v1_mut.to_vec(), &[2.0, 4.0, 6.0, 8.0]);

            v1_mut = v1_orig;
            v1_mut /= v_operand_orig;
            assert_f32_slice_eq_bitwise(&v1_mut.to_vec(), &[0.5, 1.0, 1.5, 2.0]);

            v1_mut = v1_orig;
            v1_mut %= v_operand_orig;
            assert_f32_slice_eq_bitwise(&v1_mut.to_vec(), &[1.0, 0.0, 1.0, 0.0]);
        }

        #[test]
        fn test_div_by_zero() {
            let data_ones = [1.0; LANE_COUNT];
            let data_zeros = [0.0; LANE_COUNT];
            let v_ones = F32x4::new(&data_ones);
            let v_zeros = F32x4::new(&data_zeros);

            let res = v_ones / v_zeros;
            let elements = res.to_vec();
            for &x in &elements {
                assert!(
                    x.is_infinite() && x.is_sign_positive(),
                    "Expected positive infinity from 1.0/0.0, got {x}"
                );
            }
        }

        #[test]
        fn test_rem_by_zero() {
            let data_ones = [1.0; LANE_COUNT];
            let data_zeros = [0.0; LANE_COUNT];
            let v_ones = F32x4::new(&data_ones);
            let v_zeros = F32x4::new(&data_zeros);

            let res = v_ones % v_zeros;
            let elements = res.to_vec();
            for &x in &elements {
                assert!(x.is_nan(), "Expected NaN from x % 0.0, got {x}");
            }
        }

        #[cfg(not(miri))]
        #[test]
        fn test_partial_eq_all_lanes() {
            let d1_data = [1.0, 2.0, 3.0, 4.0];
            let d2_data = [4.0, 3.0, 2.0, 1.0];
            let v1 = F32x4::new(&d1_data);
            let v2 = F32x4::new(&d2_data);
            let v1_clone = F32x4::new(&d1_data);

            assert_eq!(v1, v1_clone, "v1 should be equal to v1_clone");
            assert_ne!(v1, v2, "v1 should not be equal to v2");

            let p_data1 = [1.0, 2.0];
            let vp1 = F32x4::new(&p_data1);
            let vp2 = F32x4::new(&p_data1);
            assert_eq!(
                vp1, vp2,
                "Partially filled vectors with same content and padding should be equal"
            );

            let p_data2 = [1.0, 3.0];
            let vp3 = F32x4::new(&p_data2);
            assert_ne!(
                vp1, vp3,
                "Partially filled vectors with different content should not be equal"
            );

            let mut v_pad_diff1 = vp1;
            let mut v_pad_diff2 = vp2;

            unsafe {
                let mut raw1 = get_all_elements(v_pad_diff1);
                raw1[LANE_COUNT - 1] = 99.0;
                v_pad_diff1.elements = vld1q_f32(raw1.as_ptr());

                let mut raw2 = get_all_elements(v_pad_diff2);
                raw2[LANE_COUNT - 1] = 88.0;
                v_pad_diff2.elements = vld1q_f32(raw2.as_ptr());
            }
            assert_ne!(
                v_pad_diff1, v_pad_diff2,
                "Should be unequal due to differing padding and full hardware lane comparison"
            );

            let nan_data1 = [1.0, f32::NAN, 3.0, 4.0];
            let nan_data2 = [1.0, f32::NAN, 3.0, 4.0];
            let v_nan1 = F32x4::new(&nan_data1);
            let v_nan2 = F32x4::new(&nan_data2);
            assert_eq!(
                v_nan1, v_nan2,
                "Vectors with identical NaN patterns should be equal by vceqq_f32"
            );

            let mut v_nan_different_data = nan_data1;
            v_nan_different_data[1] = f32::from_bits(f32::NAN.to_bits() ^ 0x0000_0001);
            let v_nan_different = F32x4::new(&v_nan_different_data);
            assert_ne!(
                v_nan1, v_nan_different,
                "Vectors with different NaN bit patterns should not be equal"
            );
        }

        #[cfg(not(miri))]
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        #[test]
        fn test_partial_ord_all_lanes_semantic() {
            let v_1111 = unsafe { F32x4::splat(1.0) };
            let v_2222 = unsafe { F32x4::splat(2.0) };
            let v_1212 = F32x4::new(&[1.0, 2.0, 1.0, 2.0]);
            let v_2121 = F32x4::new(&[2.0, 1.0, 2.0, 1.0]);
            let v_1112 = F32x4::new(&[1.0, 1.0, 1.0, 2.0]);

            assert!(v_1111 < v_2222, "All 1.0s should be < all 2.0s");
            assert!(!(v_2222 < v_1111));
            assert!(!(v_1111 < v_1111));
            assert!(!(v_1212 < v_1111));
            assert!(!(v_1111 < v_1212));
            assert!(!(v_1112 < v_1111));

            assert!(v_1111 <= v_2222);
            assert!(v_1111 <= v_1111);
            assert!(!(v_2222 <= v_1111));
            assert!(!(v_1212 <= v_1111));
            assert!(v_1111 <= v_1112);

            assert!(v_2222 > v_1111);
            assert!(!(v_1111 > v_2222));
            assert!(!(v_1111 > v_1111));
            assert!(!(v_1212 > v_2222));
            assert!(!(v_1111 > v_1212));

            assert!(v_2222 >= v_1111);
            assert!(v_1111 >= v_1111);
            assert!(!(v_1111 >= v_2222));
            assert!(v_1212 >= v_1111);
            assert!(!(v_1111 >= v_1212));

            assert_eq!(v_1111.partial_cmp(&v_2222), Some(std::cmp::Ordering::Less));
            assert_eq!(
                v_2222.partial_cmp(&v_1111),
                Some(std::cmp::Ordering::Greater)
            );
            assert_eq!(v_1111.partial_cmp(&v_1111), Some(std::cmp::Ordering::Equal));

            assert_eq!(v_1212.partial_cmp(&v_1111), None);
            assert_eq!(v_1111.partial_cmp(&v_1212), None);
            assert_eq!(v_1212.partial_cmp(&v_2121), None);
            assert_eq!(v_1112.partial_cmp(&v_1111), None);
            let v_2221 = F32x4::new(&[2.0, 2.0, 2.0, 1.0]);
            assert_eq!(v_1111.partial_cmp(&v_2221), None);

            let v_nan_arr = [1.0, f32::NAN, 3.0, 4.0];
            let v_nan = F32x4::new(&v_nan_arr);

            assert!(!(v_nan < v_1111));
            assert!(!(v_nan <= v_1111));
            assert!(!(v_nan > v_1111));
            assert!(!(v_nan >= v_1111));
            assert_eq!(v_nan.partial_cmp(&v_1111), None);

            assert!(!(v_1111 < v_nan));
            assert!(!(v_1111 <= v_nan));
            assert!(!(v_1111 > v_nan));
            assert!(!(v_1111 >= v_nan));
            assert_eq!(v_1111.partial_cmp(&v_nan), None);
        }

        #[test]
        fn test_bitwise_ops() {
            let v_true_mask = unsafe { F32x4::splat(TRUE_MASK_F32) };
            let v_false_mask = unsafe { F32x4::splat(FALSE_MASK_F32) };

            let data_pattern_arr = [TRUE_MASK_F32, FALSE_MASK_F32, TRUE_MASK_F32, FALSE_MASK_F32];
            let v_pattern = F32x4::new(&data_pattern_arr);

            let and_res1 = v_true_mask & v_pattern;
            assert_f32_slice_eq_bitwise(&and_res1.to_vec(), &data_pattern_arr);
            let and_res2 = v_false_mask & v_pattern;
            assert_f32_slice_eq_bitwise(&and_res2.to_vec(), &[FALSE_MASK_F32; LANE_COUNT]);

            let or_res1 = v_true_mask | v_pattern;
            assert_f32_slice_eq_bitwise(&or_res1.to_vec(), &[TRUE_MASK_F32; LANE_COUNT]);
            let or_res2 = v_false_mask | v_pattern;
            assert_f32_slice_eq_bitwise(&or_res2.to_vec(), &data_pattern_arr);

            let mut v_pat_copy = v_pattern;
            v_pat_copy &= v_true_mask;
            assert_f32_slice_eq_bitwise(&v_pat_copy.to_vec(), &data_pattern_arr);

            v_pat_copy = v_pattern;
            v_pat_copy |= v_false_mask;
            assert_f32_slice_eq_bitwise(&v_pat_copy.to_vec(), &data_pattern_arr);
        }

        #[test]
        fn test_operator_panics_on_diff_size() {
            let v1 = F32x4::new(&[1.0; LANE_COUNT]);
            let mut v2 = F32x4::new(&[2.0; LANE_COUNT]);
            v2.size = LANE_COUNT - 1; // Manually make size different

            macro_rules! check_panic {
                ($op:expr) => {
                    let result = catch_unwind(AssertUnwindSafe(|| $op));
                    assert!(
                        result.is_err(),
                        "Operation did not panic with different sizes"
                    );
                    // Can also check panic message if it's consistent
                    // let panic_payload = result.err().unwrap();
                    // if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    //     assert!(s.contains("Operands must have the same size"));
                    // } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    //     assert!(s.contains("Operands must have the same size"));
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
