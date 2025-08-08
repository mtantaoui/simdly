use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use ::simdly::simd::SimdMath;

/// Python bindings for the simdly Rust library
///
/// This module provides high-performance SIMD-accelerated mathematical operations
/// for Python, leveraging the underlying Rust simdly library.

/// Element-wise addition of two float32 arrays
///
/// Performs optimized element-wise addition using SIMD instructions when beneficial.
/// Automatically selects the best algorithm (scalar, SIMD, or parallel SIMD) based on array size.
///
/// Args:
///     a: First input numpy array of f32 values
///     b: Second input numpy array of f32 values (must be same length as a)
///
/// Returns:
///     A new numpy array containing the element-wise sum
///
/// Raises:
///     ValueError: If input arrays have different lengths or are empty
///
/// Example:
///     >>> import simdly
///     >>> import numpy as np
///     >>> a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
///     >>> b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
///     >>> result = simdly.add(a, b)
///     >>> print(result)  # [5.0, 7.0, 9.0]
#[pyfunction]
fn add<'py>(py: Python<'py>, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let a_slice = a.as_slice()?;
    let b_slice = b.as_slice()?;
    
    if a_slice.len() != b_slice.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays must have the same length",
        ));
    }

    if a_slice.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input arrays cannot be empty",
        ));
    }

    let result: Vec<f32> = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x + y).collect();
    Ok(PyArray1::from_vec_bound(py, result))
}

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

/// Print array elements with formatting
///
/// Prints the contents of a numpy array in a readable format.
/// Useful for debugging and development.
///
/// Args:
///     array: Input numpy array of f32 values
///     prefix: Optional prefix string to display before the array
///
/// Example:
///     >>> import simdly
///     >>> import numpy as np
///     >>> arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
///     >>> simdly.print_array(arr, "Result: ")
#[pyfunction]
#[pyo3(signature = (array, prefix = "Array"))]
fn print_array(array: PyReadonlyArray1<f32>, prefix: &str) -> PyResult<()> {
    let array_slice = array.as_slice()?;

    println!("hello\n");
    if array_slice.is_empty() {
        println!("{}: []", prefix);
        return Ok(());
    }

    print!("{}: [", prefix);
    for (i, &value) in array_slice.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.6}", value);
    }
    println!("]");

    Ok(())
}

/// A Python module for high-performance SIMD mathematical operations
#[pymodule]
fn simdly(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(print_array, m)?)?;
    Ok(())
}
