pub trait SimdVec<T> {
    fn new(slice: &[T]) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn splat(value: T) -> Self;

    fn is_aligned(ptr: *const T) -> bool;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn load(ptr: *const T, size: usize) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn load_aligned(ptr: *const T, size: usize) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn load_unaligned(ptr: *const T, size: usize) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn load_partial(ptr: *const T, size: usize) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn store_in_vec(&self) -> Vec<T>;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn store_in_vec_partial(&self) -> Vec<T>;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn store_at(&self, ptr: *mut T);

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn store_at_partial(&self, ptr: *mut T);

    fn to_vec(self) -> Vec<T>;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn eq_elements(&self, rhs: Self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn lt_elements(&self, rhs: Self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn le_elements(&self, rhs: Self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn gt_elements(&self, rhs: Self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn ge_elements(&self, rhs: Self) -> Self;
}

pub trait SimdAdd<Rhs = Self> {
    type Output;

    fn simd_add(self, rhs: Rhs) -> Self::Output;
    fn par_simd_add(self, rhs: Rhs) -> Self::Output;
    fn scalar_add(self, rhs: Rhs) -> Self::Output;
}
