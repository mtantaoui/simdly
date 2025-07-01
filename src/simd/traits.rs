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
    /// # Safety
    /// .
    unsafe fn permute<const IMM8: i32>(&self) -> Self;

    /// .
    /// # Safety
    /// .
    unsafe fn permute2f128<const IMM8: i32>(&self, other: Self) -> Self;

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
    unsafe fn load_aligned(ptr: *const T) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn load_unaligned(ptr: *const T) -> Self;

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

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn abs(&self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn acos(&self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn asin(&self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn cos(&self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn fmadd(&self, a: Self, b: Self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn unpackhi(&self, other: Self) -> Self;

    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn unpacklo(&self, other: Self) -> Self;
}

pub trait SimdAdd<Rhs = Self> {
    type Output;

    fn simd_add(self, rhs: Rhs) -> Self::Output;
    fn par_simd_add(self, rhs: Rhs) -> Self::Output;
    fn scalar_add(self, rhs: Rhs) -> Self::Output;
}

pub trait SimdCos<Rhs = Self> {
    type Output;

    fn simd_cos(self) -> Self::Output;
    fn par_simd_cos(self) -> Self::Output;
    fn scalar_cos(self) -> Self::Output;
}

pub trait SimdAbs<Rhs = Self> {
    type Output;

    fn simd_abs(self) -> Self::Output;
    fn par_simd_abs(self) -> Self::Output;
    fn scalar_abs(self) -> Self::Output;
}

pub trait SimdAsin<Rhs = Self> {
    type Output;

    fn simd_asin(self) -> Self::Output;
    fn par_simd_asin(self) -> Self::Output;
    fn scalar_asin(self) -> Self::Output;
}

pub trait SimdAcos<Rhs = Self> {
    type Output;

    fn simd_acos(self) -> Self::Output;
    fn par_simd_acos(self) -> Self::Output;
    fn scalar_acos(self) -> Self::Output;
}
