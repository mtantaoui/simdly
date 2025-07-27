//! AVX2 8-lane f32 SIMD vector implementation.
//!
//! This module provides `F32x8`, a SIMD vector type that wraps Intel's AVX2 `__m256`
//! intrinsic to perform vectorized operations on 8 single-precision floating-point
//! values simultaneously using 256-bit AVX2 instructions.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::simd::{Alignment, SimdLoad, SimdMath, SimdStore};

// const AVX_ALIGNMENT: usize = 32;
const LANE_COUNT: usize = 8;

/// AVX2 SIMD vector containing 8 packed f32 values.
///
/// This structure provides efficient vectorized operations on 8 single-precision
/// floating-point numbers using AVX2 instructions. It maintains both the underlying
/// AVX2 register and the count of valid elements for partial operations.
///
/// # Memory Alignment
///
/// For optimal performance, data should be aligned to 32-byte boundaries when possible.
/// The structure automatically detects alignment and uses the most efficient load/store
/// operations available.
///
/// # Examples
///
/// ```rust
/// # use simdly::simd::avx2::f32x8::F32x8;
/// # use simdly::simd::SimdLoad;
/// // Load 8 f32 values
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let vec = F32x8::from_slice(&data);
///
/// // Partial load (< 8 elements)
/// let partial = [1.0f32, 2.0, 3.0];
/// let partial_vec = F32x8::from_slice(&partial);
/// ```
#[derive(Copy, Clone, Debug)]
pub struct F32x8 {
    /// Number of valid elements in the vector (1-8)
    pub size: usize,
    /// AVX2 256-bit vector register containing 8 packed f32 values
    pub elements: __m256,
}

impl Alignment<f32> for F32x8 {
    /// Checks if a pointer is properly aligned for AVX2 operations.
    ///
    /// AVX2 operations perform optimally when data is aligned to 32-byte boundaries.
    /// This function checks if the given pointer meets this alignment requirement.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to check for alignment
    ///
    /// # Returns
    ///
    /// `true` if the pointer is 32-byte aligned, `false` otherwise
    fn is_aligned(ptr: *const f32) -> bool {
        let ptr = ptr as usize;

        ptr % core::mem::align_of::<__m256>() == 0
    }
}

impl SimdLoad<f32> for F32x8 {
    type Output = Self;

    /// Loads data from a slice into the SIMD vector.
    ///
    /// This is the high-level interface for loading f32 data. It automatically
    /// handles partial loads for slices smaller than 8 elements and uses the
    /// most appropriate loading strategy based on slice size.
    ///
    /// # Arguments
    ///
    /// * `slice` - Input slice of f32 values
    ///
    /// # Returns
    ///
    /// F32x8 instance with loaded data
    ///
    /// # Behavior
    ///
    /// - For slices < 8 elements: Uses masked partial loading
    /// - For slices >= 8 elements: Loads the first 8 elements using full loading
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the slice is empty.
    fn from_slice(slice: &[f32]) -> Self::Output {
        debug_assert!(!slice.is_empty(), "data pointer can't be NULL");

        let size = slice.len();

        match slice.len().cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { Self::load_partial(slice.as_ptr(), size) },
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => unsafe {
                Self::load(slice.as_ptr(), LANE_COUNT)
            },
        }
    }

    /// Loads exactly 8 elements from memory.
    ///
    /// Automatically chooses between aligned and unaligned load based on pointer alignment.
    /// This function is optimized for loading complete vectors.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data
    /// * `size` - Must be exactly 8
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to at least 8 valid f32 values.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size != 8 or if pointer is null.
    unsafe fn load(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(size == LANE_COUNT, "Size must be == {LANE_COUNT}");
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        match F32x8::is_aligned(ptr) {
            true => unsafe { Self::load_aligned(ptr) },
            false => unsafe { Self::load_unaligned(ptr) },
        }
    }

    /// Loads 8 elements from 32-byte aligned memory.
    ///
    /// This is the fastest loading method when alignment is guaranteed.
    /// Uses the optimized `_mm256_load_ps` intrinsic.
    ///
    /// # Arguments
    ///
    /// * `ptr` - 32-byte aligned pointer to f32 data
    ///
    /// # Safety
    ///
    /// Pointer must be 32-byte aligned and point to at least 8 valid f32 values.
    unsafe fn load_aligned(ptr: *const f32) -> Self::Output {
        Self {
            elements: _mm256_load_ps(ptr),
            size: LANE_COUNT,
        }
    }

    /// Loads 8 elements from unaligned memory.
    ///
    /// Slightly slower than aligned load but works with any memory alignment.
    /// Uses the `_mm256_loadu_ps` intrinsic.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data (no alignment requirement)
    ///
    /// # Safety
    ///
    /// Pointer must point to at least 8 valid f32 values.
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Output {
        Self {
            elements: _mm256_loadu_ps(ptr),
            size: LANE_COUNT,
        }
    }

    /// Loads fewer than 8 elements using masked loading operations.
    ///
    /// This function safely loads partial data when the source contains fewer than
    /// 8 elements. It uses AVX2 masked load operations to prevent reading beyond
    /// the valid data range.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to f32 data
    /// * `size` - Number of elements to load (1-7)
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to at least `size` valid f32 values.
    ///
    /// # Implementation
    ///
    /// Uses `_mm256_maskload_ps` with dynamically generated masks based on the
    /// size parameter. Unloaded lanes contain undefined values.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size >= 8 or if pointer is null.
    unsafe fn load_partial(ptr: *const f32, size: usize) -> Self::Output {
        debug_assert!(
            size < LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );

        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mask = match size {
            1 => _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0),
            2 => _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0),
            3 => _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0),
            4 => _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0),
            5 => _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0),
            6 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0),
            7 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0),
            _ => unreachable!(),
        };

        Self {
            elements: _mm256_maskload_ps(ptr, mask),
            size,
        }
    }
}

impl SimdStore<f32> for F32x8 {
    type Output = Self;

    /// Stores vector data at the given pointer location.
    ///
    /// Automatically chooses the most appropriate store method based on:
    /// - Vector size (partial vs. full store)
    /// - Pointer alignment (aligned vs. unaligned store)
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to destination memory
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to sufficient writable memory
    /// for `self.size` elements.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size > 8 or if pointer is null.
    fn store_at(&self, ptr: *const f32) {
        debug_assert!(
            self.size <= LANE_COUNT,
            "{}",
            format!("Size must be <= {LANE_COUNT}")
        );
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        // Cast to mutable pointer for store operations
        let mut_ptr = ptr as *mut f32;

        match self.size.cmp(&LANE_COUNT) {
            std::cmp::Ordering::Less => unsafe { self.store_at_partial(mut_ptr) },
            std::cmp::Ordering::Equal => match F32x8::is_aligned(ptr) {
                true => unsafe { self.store_aligned_at(mut_ptr) },
                false => unsafe { self.store_unaligned_at(mut_ptr) },
            },
            std::cmp::Ordering::Greater => unreachable!("Size cannot exceed LANE_COUNT"),
        }
    }

    /// Non-temporal store that bypasses cache.
    ///
    /// Stores all 8 vector elements to memory without polluting the cache.
    /// This is optimal for large datasets where data won't be reused soon.
    ///
    /// # Arguments
    ///
    /// * `ptr` - **32-byte aligned** mutable pointer to destination memory
    ///
    /// # Safety
    ///
    /// - Pointer must be 32-byte aligned for optimal performance and to avoid undefined behavior
    /// - Pointer must point to at least 8 valid f32 memory locations
    /// - Use `Alignment::is_aligned()` to verify alignment before calling
    ///
    /// # Performance Notes
    ///
    /// - **Alignment Required**: The underlying `_mm256_stream_ps` intrinsic requires 32-byte alignment
    /// - **Cache Bypass**: Data is written directly to memory, bypassing CPU cache
    /// - **Memory Ordering**: May have weaker memory ordering guarantees than regular stores
    ///
    /// # Use Case
    ///
    /// Best for write-once scenarios with large datasets that won't be
    /// accessed again soon, such as:
    /// - Large array initialization
    /// - Data export operations  
    /// - Streaming computations with sequential access patterns
    unsafe fn stream_at(&self, ptr: *mut f32) {
        _mm256_stream_ps(ptr, self.elements)
    }

    /// Stores 8 elements to 32-byte aligned memory.
    ///
    /// This is the fastest store method when alignment is guaranteed.
    /// Uses the optimized `_mm256_store_ps` intrinsic.
    ///
    /// # Arguments
    ///
    /// * `ptr` - 32-byte aligned mutable pointer to destination
    ///
    /// # Safety
    ///
    /// Pointer must be 32-byte aligned and point to at least 8 valid f32
    /// memory locations.
    unsafe fn store_aligned_at(&self, ptr: *mut f32) {
        _mm256_store_ps(ptr, self.elements)
    }

    /// Stores 8 elements to unaligned memory.
    ///
    /// Works with any memory alignment but slightly slower than aligned store.
    /// Uses the `_mm256_storeu_ps` intrinsic.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination (no alignment requirement)
    ///
    /// # Safety
    ///
    /// Pointer must point to at least 8 valid f32 memory locations.
    unsafe fn store_unaligned_at(&self, ptr: *mut f32) {
        _mm256_storeu_ps(ptr, self.elements)
    }

    /// Stores only the valid elements using masked store operations.
    ///
    /// This function safely stores partial data when the vector contains fewer
    /// than 8 valid elements. It uses AVX2 masked store operations to prevent
    /// writing beyond the intended memory range.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Mutable pointer to destination memory
    ///
    /// # Safety
    ///
    /// Pointer must not be null and must point to at least `self.size` valid
    /// f32 memory locations.
    ///
    /// # Implementation
    ///
    /// Uses `_mm256_maskstore_ps` with masks corresponding to `self.size`.
    /// Only the first `self.size` elements are written to memory.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if size >= 8 or if pointer is null.
    unsafe fn store_at_partial(&self, ptr: *mut f32) {
        debug_assert!(
            self.size < LANE_COUNT,
            "{}",
            format!("Size must be < {LANE_COUNT}")
        );
        debug_assert!(!ptr.is_null(), "Pointer must not be null");

        let mask: __m256i = match self.size {
            1 => _mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0),
            2 => _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0),
            3 => _mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0),
            4 => _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0),
            5 => _mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0),
            6 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0),
            7 => _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0),
            _ => unreachable!("Size must be < LANE_COUNT"),
        };

        _mm256_maskstore_ps(ptr, mask, self.elements);
    }
}

impl SimdMath<f32> for F32x8 {
    type Output = Self;

    fn abs(&self) -> Self::Output {
        todo!()
    }

    fn acos(&self) -> Self::Output {
        todo!()
    }

    fn asin(&self) -> Self::Output {
        todo!()
    }

    fn atan(&self) -> Self::Output {
        todo!()
    }

    fn atan2(&self) -> Self::Output {
        todo!()
    }

    fn cbrt(&self) -> Self::Output {
        todo!()
    }

    fn floor(&self) -> Self::Output {
        todo!()
    }

    fn exp(&self) -> Self::Output {
        todo!()
    }

    fn ln(&self) -> Self::Output {
        todo!()
    }

    fn hypot(&self) -> Self::Output {
        todo!()
    }

    fn pow(&self) -> Self::Output {
        todo!()
    }

    fn sin(&self) -> Self::Output {
        todo!()
    }

    fn cos(&self) -> Self::Output {
        todo!()
    }

    fn tan(&self) -> Self::Output {
        todo!()
    }

    fn sqrt(&self) -> Self::Output {
        todo!()
    }

    fn ceil(&self) -> Self::Output {
        todo!()
    }

    fn hypot3(&self) -> Self::Output {
        todo!()
    }

    fn hypot4(&self) -> Self::Output {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    // Helper function to create aligned memory
    fn alloc_aligned(size: usize, align: usize) -> *mut f32 {
        let layout = Layout::from_size_align(size * std::mem::size_of::<f32>(), align).unwrap();
        unsafe { alloc(layout) as *mut f32 }
    }

    // Helper function to deallocate aligned memory
    fn dealloc_aligned(ptr: *mut f32, size: usize, align: usize) {
        let layout = Layout::from_size_align(size * std::mem::size_of::<f32>(), align).unwrap();
        unsafe { dealloc(ptr as *mut u8, layout) };
    }

    // Helper function to extract vector elements for comparison
    fn extract_elements(vec: &F32x8) -> [f32; 8] {
        let mut result = [0.0f32; 8];
        unsafe {
            std::arch::x86_64::_mm256_storeu_ps(result.as_mut_ptr(), vec.elements);
        }
        result
    }

    mod alignment_tests {
        use super::*;

        #[test]
        fn test_is_aligned_32_byte_boundary() {
            let aligned_ptr = alloc_aligned(8, 32);
            assert!(F32x8::is_aligned(aligned_ptr));
            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_is_not_aligned() {
            let data = [1.0f32; 16];
            let unaligned_ptr = unsafe { data.as_ptr().add(1) }; // Offset by 1 element (4 bytes)
            assert!(!F32x8::is_aligned(unaligned_ptr));
        }

        #[test]
        fn test_stack_array_alignment() {
            // Stack arrays may or may not be aligned depending on platform
            let data = [1.0f32; 8];
            let is_aligned = F32x8::is_aligned(data.as_ptr());
            // We don't assert a specific result since stack alignment varies
            // but the function should not panic
            println!("Stack array aligned: {is_aligned}");
        }
    }

    mod simd_load_tests {
        use super::*;

        #[test]
        fn test_from_slice_full() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from_slice(&data);

            assert_eq!(vec.size, 8);
            let elements = extract_elements(&vec);
            assert_eq!(elements, data);
        }

        #[test]
        fn test_from_slice_oversized() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            let vec = F32x8::from_slice(&data);

            assert_eq!(vec.size, 8); // Should still be 8, not 10
            let elements = extract_elements(&vec);
            assert_eq!(elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }

        #[test]
        fn test_from_slice_partial() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0];
            let vec = F32x8::from_slice(&data);

            assert_eq!(vec.size, 5);
            let elements = extract_elements(&vec);
            // First 5 elements should match, rest are undefined but we don't test them
            assert_eq!(&elements[..5], &data);
        }

        #[test]
        fn test_load_aligned() {
            let aligned_ptr = alloc_aligned(8, 32);
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            unsafe {
                std::ptr::copy_nonoverlapping(test_data.as_ptr(), aligned_ptr, 8);
            }

            let vec = unsafe { F32x8::load_aligned(aligned_ptr) };
            assert_eq!(vec.size, 8);

            let elements = extract_elements(&vec);
            assert_eq!(elements, test_data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_load_unaligned() {
            let data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let unaligned_ptr = unsafe { data.as_ptr().add(1) }; // Skip first element

            let vec = unsafe { F32x8::load_unaligned(unaligned_ptr) };
            assert_eq!(vec.size, 8);

            let elements = extract_elements(&vec);
            assert_eq!(elements, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }

        #[test]
        fn test_load_with_aligned_detection() {
            let aligned_ptr = alloc_aligned(8, 32);
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            unsafe {
                std::ptr::copy_nonoverlapping(test_data.as_ptr(), aligned_ptr, 8);
            }

            let vec = unsafe { F32x8::load(aligned_ptr, 8) };
            assert_eq!(vec.size, 8);

            let elements = extract_elements(&vec);
            assert_eq!(elements, test_data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_load_partial_single_element() {
            let data = [42.0f32];
            let vec = unsafe { F32x8::load_partial(data.as_ptr(), 1) };

            assert_eq!(vec.size, 1);
            let elements = extract_elements(&vec);
            assert_eq!(elements[0], 42.0);
        }

        #[test]
        fn test_load_partial_multiple_elements() {
            for size in 1..8 {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                let vec = unsafe { F32x8::load_partial(data.as_ptr(), size) };

                assert_eq!(vec.size, size);
                let elements = extract_elements(&vec);

                // for i in 0..size {
                for (i, e) in elements.iter().enumerate().take(size) {
                    assert_eq!(*e, i as f32, "Mismatch at index {i} for size {size}");
                }
            }
        }

        #[test]
        fn test_load_partial_seven_elements() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            let vec = unsafe { F32x8::load_partial(data.as_ptr(), 7) };

            assert_eq!(vec.size, 7);
            let elements = extract_elements(&vec);
            assert_eq!(&elements[..7], &data);
        }
    }

    mod simd_store_tests {
        use super::*;

        #[test]
        fn test_store_aligned() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from_slice(&test_data);

            let aligned_ptr = alloc_aligned(8, 32);
            unsafe { vec.store_aligned_at(aligned_ptr) };

            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result, &test_data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_store_unaligned() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from_slice(&test_data);

            let mut buffer = [0.0f32; 10];
            let unaligned_ptr = unsafe { buffer.as_mut_ptr().add(1) };

            unsafe { vec.store_unaligned_at(unaligned_ptr) };

            assert_eq!(&buffer[1..9], &test_data);
            assert_eq!(buffer[0], 0.0); // Should be unchanged
            assert_eq!(buffer[9], 0.0); // Should be unchanged
        }

        #[test]
        fn test_stream_at_aligned() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from_slice(&test_data);

            // Use properly aligned memory for streaming store
            let aligned_ptr = alloc_aligned(8, 32);
            unsafe { vec.stream_at(aligned_ptr) };

            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result, &test_data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_stream_at_alignment_check() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from_slice(&test_data);

            // Test with properly aligned memory
            let aligned_ptr = alloc_aligned(8, 32);
            assert!(
                F32x8::is_aligned(aligned_ptr),
                "Allocated memory should be aligned"
            );

            unsafe { vec.stream_at(aligned_ptr) };
            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result, &test_data);

            dealloc_aligned(aligned_ptr, 8, 32);

            // Test alignment detection with unaligned memory
            let mut buffer = [0.0f32; 10];
            let unaligned_ptr = unsafe { buffer.as_mut_ptr().add(1) }; // Offset by 4 bytes
            assert!(
                !F32x8::is_aligned(unaligned_ptr),
                "Offset pointer should not be aligned"
            );
        }

        #[test]
        fn test_store_partial_single_element() {
            let test_data = [42.0];
            let vec = unsafe { F32x8::load_partial(test_data.as_ptr(), 1) };

            let mut buffer = [0.0f32; 8];
            unsafe { vec.store_at_partial(buffer.as_mut_ptr()) };

            assert_eq!(buffer[0], 42.0);
            // Rest should remain zero
            for (i, e) in buffer.iter().enumerate().skip(1) {
                assert_eq!(*e, 0.0, "Non-zero value at index {i}");
            }
        }

        #[test]
        fn test_store_partial_multiple_elements() {
            for size in 1..8 {
                let test_data: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
                let vec = unsafe { F32x8::load_partial(test_data.as_ptr(), size) };

                let mut buffer = [0.0f32; 8];
                unsafe { vec.store_at_partial(buffer.as_mut_ptr()) };

                // Check stored elements
                for (i, e) in buffer.iter().enumerate().take(size) {
                    assert_eq!(*e, (i + 1) as f32, "Mismatch at index {i} for size {size}");
                }

                // Check remaining elements are zero
                for (i, e) in buffer.iter().enumerate().skip(size) {
                    assert_eq!(*e, 0.0, "Non-zero value at index {i} for size {size}");
                }
            }
        }

        #[test]
        fn test_store_partial_seven_elements() {
            let test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            let vec = unsafe { F32x8::load_partial(test_data.as_ptr(), 7) };

            let mut buffer = [0.0f32; 8];
            unsafe { vec.store_at_partial(buffer.as_mut_ptr()) };

            assert_eq!(&buffer[..7], &test_data);
            assert_eq!(buffer[7], 0.0);
        }
    }

    mod roundtrip_tests {
        use super::*;

        #[test]
        fn test_load_store_roundtrip_full() {
            let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from_slice(&original);

            let mut result = [0.0f32; 8];
            unsafe { vec.store_unaligned_at(result.as_mut_ptr()) };

            assert_eq!(result, original);
        }

        #[test]
        fn test_load_store_roundtrip_partial() {
            for size in 1..8 {
                let original: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
                let vec = unsafe { F32x8::load_partial(original.as_ptr(), size) };

                let mut result = [0.0f32; 8];
                unsafe { vec.store_at_partial(result.as_mut_ptr()) };

                assert_eq!(
                    &result[..size],
                    &original[..],
                    "Roundtrip failed for size {size}"
                );
            }
        }

        #[test]
        fn test_aligned_roundtrip() {
            let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            let aligned_src = alloc_aligned(8, 32);
            let aligned_dst = alloc_aligned(8, 32);

            unsafe {
                std::ptr::copy_nonoverlapping(original.as_ptr(), aligned_src, 8);
            }

            let vec = unsafe { F32x8::load_aligned(aligned_src) };
            unsafe { vec.store_aligned_at(aligned_dst) };

            let result = unsafe { std::slice::from_raw_parts(aligned_dst, 8) };
            assert_eq!(result, &original);

            dealloc_aligned(aligned_src, 8, 32);
            dealloc_aligned(aligned_dst, 8, 32);
        }

        #[test]
        fn test_stream_roundtrip() {
            let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let vec = F32x8::from_slice(&original);

            // Stream to aligned memory
            let aligned_dst = alloc_aligned(8, 32);
            unsafe { vec.stream_at(aligned_dst) };

            let result = unsafe { std::slice::from_raw_parts(aligned_dst, 8) };
            assert_eq!(result, &original);

            dealloc_aligned(aligned_dst, 8, 32);
        }
    }

    mod edge_cases {
        use super::*;

        #[test]
        fn test_zero_values() {
            let zeros = [0.0f32; 8];
            let vec = F32x8::from_slice(&zeros);

            let mut result = [1.0f32; 8]; // Initialize with non-zero
            unsafe { vec.store_unaligned_at(result.as_mut_ptr()) };

            assert_eq!(result, zeros);
        }

        #[test]
        fn test_negative_values() {
            let negatives = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
            let vec = F32x8::from_slice(&negatives);

            let mut result = [0.0f32; 8];
            unsafe { vec.store_unaligned_at(result.as_mut_ptr()) };

            assert_eq!(result, negatives);
        }

        #[test]
        fn test_special_float_values() {
            let special = [
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                0.0,
                -0.0,
                f32::MIN,
                f32::MAX,
                f32::EPSILON,
            ];

            let vec = F32x8::from_slice(&special);
            let mut result = [0.0f32; 8];
            unsafe { vec.store_unaligned_at(result.as_mut_ptr()) };

            assert_eq!(result[0], f32::INFINITY);
            assert_eq!(result[1], f32::NEG_INFINITY);
            assert!(result[2].is_nan());
            assert_eq!(result[3], 0.0);
            assert_eq!(result[4], -0.0);
            assert_eq!(result[5], f32::MIN);
            assert_eq!(result[6], f32::MAX);
            assert_eq!(result[7], f32::EPSILON);
        }

        #[test]
        fn test_very_small_partial_load() {
            let single = [std::f32::consts::PI];
            let vec = unsafe { F32x8::load_partial(single.as_ptr(), 1) };

            assert_eq!(vec.size, 1);
            let elements = extract_elements(&vec);
            assert_eq!(elements[0], std::f32::consts::PI);
        }

        #[test]
        fn test_mixed_aligned_unaligned_operations() {
            let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            // Load unaligned
            let vec = unsafe { F32x8::load_unaligned(data.as_ptr()) };

            // Store aligned
            let aligned_ptr = alloc_aligned(8, 32);
            unsafe { vec.store_aligned_at(aligned_ptr) };

            // Verify
            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result, &data);

            dealloc_aligned(aligned_ptr, 8, 32);
        }

        #[test]
        fn test_stream_with_special_values() {
            let special = [
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NAN,
                0.0,
                -0.0,
                f32::MIN,
                f32::MAX,
                f32::EPSILON,
            ];

            let vec = F32x8::from_slice(&special);

            // Stream to aligned memory
            let aligned_ptr = alloc_aligned(8, 32);
            unsafe { vec.stream_at(aligned_ptr) };

            let result = unsafe { std::slice::from_raw_parts(aligned_ptr, 8) };
            assert_eq!(result[0], f32::INFINITY);
            assert_eq!(result[1], f32::NEG_INFINITY);
            assert!(result[2].is_nan());
            assert_eq!(result[3], 0.0);
            assert_eq!(result[4], -0.0);
            assert_eq!(result[5], f32::MIN);
            assert_eq!(result[6], f32::MAX);
            assert_eq!(result[7], f32::EPSILON);

            dealloc_aligned(aligned_ptr, 8, 32);
        }
    }

    #[cfg(debug_assertions)]
    mod debug_assertion_tests {
        use super::*;

        #[test]
        #[should_panic(expected = "data pointer can't be NULL")]
        fn test_from_slice_empty_panic() {
            let empty: &[f32] = &[];
            F32x8::from_slice(empty);
        }

        #[test]
        #[should_panic(expected = "Size must be == 8")]
        fn test_load_wrong_size_panic() {
            let data = [1.0f32; 4];
            unsafe { F32x8::load(data.as_ptr(), 4) }; // Wrong size
        }

        #[test]
        #[should_panic(expected = "Size must be < 8")]
        fn test_load_partial_full_size_panic() {
            let data = [1.0f32; 8];
            unsafe { F32x8::load_partial(data.as_ptr(), 8) }; // Should be < 8
        }
    }
}
