//! Error types for simdly operations.
//!
//! This module defines custom error types that provide better error handling
//! than panicking, allowing applications to gracefully handle failures.

use std::fmt;

/// Errors that can occur during simdly operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimdlyError {
    /// Memory allocation failed.
    AllocationError {
        /// The size that was requested to be allocated.
        requested_size: usize,
        /// The alignment that was requested.
        requested_alignment: usize,
        /// Human-readable error message.
        message: String,
    },
    /// Invalid layout parameters were provided.
    LayoutError {
        /// The size parameter that caused the error.
        size: usize,
        /// The alignment parameter that caused the error.
        alignment: usize,
        /// Human-readable error message.
        message: String,
    },
    /// Input validation error.
    ValidationError {
        /// Human-readable error message.
        message: String,
    },
}

impl fmt::Display for SimdlyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimdlyError::AllocationError {
                requested_size,
                requested_alignment,
                message,
            } => write!(
                f,
                "Memory allocation failed: {} (requested {} bytes with {} byte alignment)",
                message, requested_size, requested_alignment
            ),
            SimdlyError::LayoutError {
                size,
                alignment,
                message,
            } => write!(
                f,
                "Invalid memory layout: {} (size: {}, alignment: {})",
                message, size, alignment
            ),
            SimdlyError::ValidationError { message } => {
                write!(f, "Validation error: {}", message)
            }
        }
    }
}

impl std::error::Error for SimdlyError {}

/// Result type alias for simdly operations.
pub type Result<T> = std::result::Result<T, SimdlyError>;

/// Creates an allocation error.
pub fn allocation_error(size: usize, alignment: usize, message: impl Into<String>) -> SimdlyError {
    SimdlyError::AllocationError {
        requested_size: size,
        requested_alignment: alignment,
        message: message.into(),
    }
}

/// Creates a layout error.
pub fn layout_error(size: usize, alignment: usize, message: impl Into<String>) -> SimdlyError {
    SimdlyError::LayoutError {
        size,
        alignment,
        message: message.into(),
    }
}

/// Creates a validation error.
pub fn validation_error(message: impl Into<String>) -> SimdlyError {
    SimdlyError::ValidationError {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_error_display() {
        let error = allocation_error(1024, 32, "out of memory");
        let display = format!("{}", error);
        assert!(display.contains("Memory allocation failed"));
        assert!(display.contains("1024 bytes"));
        assert!(display.contains("32 byte alignment"));
        assert!(display.contains("out of memory"));
    }

    #[test]
    fn test_layout_error_display() {
        let error = layout_error(1000, 31, "alignment must be power of two");
        let display = format!("{}", error);
        assert!(display.contains("Invalid memory layout"));
        assert!(display.contains("size: 1000"));
        assert!(display.contains("alignment: 31"));
        assert!(display.contains("alignment must be power of two"));
    }

    #[test]
    fn test_validation_error_display() {
        let error = validation_error("input slices must have same length");
        let display = format!("{}", error);
        assert!(display.contains("Validation error"));
        assert!(display.contains("input slices must have same length"));
    }

    #[test]
    fn test_error_equality() {
        let error1 = allocation_error(1024, 32, "test");
        let error2 = allocation_error(1024, 32, "test");
        let error3 = allocation_error(2048, 32, "test");

        assert_eq!(error1, error2);
        assert_ne!(error1, error3);
    }

    #[test]
    fn test_error_trait_implementation() {
        let error = allocation_error(1024, 32, "test error");
        
        // Should implement Error trait
        let _: &dyn std::error::Error = &error;
        
        // Should have source method (returns None for our simple errors)
        assert!(std::error::Error::source(&error).is_none());
    }
}