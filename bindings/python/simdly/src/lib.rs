use ::simdly::simd::{
    avx2::outer::{outer, par_outer},
    SimdMath,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

/// Element-wise cosine computation on a float32 array
///
/// Computes the cosine of each element using SIMD-accelerated operations.
/// Provides significant performance improvements over scalar operations.
///
/// Args:
///     a: Input numpy array of f32 values (angles in radians)
///
/// Returns:
///     A new numpy array containing the cosine of each input element
///
/// Raises:
///     ValueError: If input array is empty
///
/// Example:
///     >>> import simdly
///     >>> import numpy as np
///     >>> import math
///     >>> a = np.array([0.0, math.pi/4, math.pi/2, math.pi], dtype=np.float32)
///     >>> result = simdly.cos(a)
///     >>> print(result)  # [1.0, 0.707..., 0.0, -1.0]
#[pyfunction]
fn cos<'py>(py: Python<'py>, a: PyReadonlyArray1<f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a_slice = a.as_slice()?;

    if a_slice.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input array cannot be empty",
        ));
    }

    let result: Vec<f32> = a_slice.cos();
    Ok(PyArray1::from_vec_bound(py, result))
}

/// Computes the outer product of two 1D float32 arrays
///
/// Computes the outer product using SIMD-accelerated operations for optimal performance.
/// For two 1D arrays `a` and `b`, returns a 2D array where `result[i,j] = a[i] * b[j]`.
///
/// Args:
///     a: First input numpy array of f32 values
///     b: Second input numpy array of f32 values
///
/// Returns:
///     A new 2D numpy array containing the outer product
///
/// Raises:
///     ValueError: If either input array is empty
///
/// Example:
///     >>> import simdly
///     >>> import numpy as np
///     >>> a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
///     >>> b = np.array([4.0, 5.0], dtype=np.float32)
///     >>> result = simdly.outer(a, b)
///     >>> print(result)  # [[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]]
#[pyfunction]
fn outer_product<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
) -> PyResult<
    // Bound<'py, PyArray2<f32>>
    (),
> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;

    if a_slice.is_empty() || b_slice.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let result_flat = outer(a_slice, b_slice);

    // let result_2d: Vec<Vec<f32>> = result_flat
    //     .chunks(b_slice.len())
    //     .map(|chunk| chunk.to_vec())
    //     .collect();

    Ok(
        // PyArray2::from_vec2_bound(py, &result_2d).unwrap()
        (),
    )
}

/// Computes the outer product of two 1D float32 arrays using parallel processing
///
/// Computes the outer product using parallel SIMD-accelerated operations for optimal
/// performance on multi-core systems. For two 1D arrays `a` and `b`, returns a 2D
/// array where `result[i,j] = a[i] * b[j]`.
///
/// Args:
///     a: First input numpy array of f32 values
///     b: Second input numpy array of f32 values
///
/// Returns:
///     A new 2D numpy array containing the outer product
///
/// Raises:
///     ValueError: If either input array is empty
///
/// Example:
///     >>> import simdly
///     >>> import numpy as np
///     >>> a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
///     >>> b = np.array([4.0, 5.0], dtype=np.float32)
///     >>> result = simdly.par_outer_product(a, b)
///     >>> print(result)  # [[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]]
#[pyfunction]
fn par_outer_product<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f32>,
    b: PyReadonlyArray1<f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;

    if a_slice.is_empty() || b_slice.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    // Release GIL to allow parallel execution
    let result_flat = py.allow_threads(|| par_outer(a_slice, b_slice));

    let result_2d: Vec<Vec<f32>> = result_flat
        .chunks(b_slice.len())
        .map(|chunk| chunk.to_vec())
        .collect();

    Ok(PyArray2::from_vec2_bound(py, &result_2d).unwrap())
}

/// A Python module for high-performance SIMD mathematical operations
#[pymodule]
fn simdly(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(outer_product, m)?)?;
    m.add_function(wrap_pyfunction!(par_outer_product, m)?)?;
    Ok(())
}
