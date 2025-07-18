//! Dimension system for tensors.
//!
//! This module provides types and traits for handling tensor dimensions,
//! including both static (compile-time) and dynamic (run-time) dimensions.
//! It supports operations like broadcasting, shape validation, and dimension manipulation.

use crate::error::{Result, TensorustError};
use std::fmt::{Debug, Display};

pub mod dynamic;
pub mod shape;
pub mod stride;
pub mod static_;

pub use shape::Shape;
pub use stride::Stride;
pub use static_::StaticDim;

/// A trait for types that can represent tensor dimensions.
///
/// This trait provides an abstraction over different dimension representations,
/// such as static (compile-time) and dynamic (run-time) dimensions.
pub trait Dimension: Clone + Debug + Display + Send + Sync + 'static {
    /// Returns the shape of the dimension.
    fn shape(&self) -> &[usize];

    /// Returns the strides of the dimension.
    fn strides(&self) -> &[usize];

    /// Returns the number of dimensions.
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Returns the total number of elements in the dimension.
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns `true` if the dimension represents a contiguous memory layout.
    fn is_contiguous(&self) -> bool;

    /// Creates a new dimension from the given shape.
    fn from_shape(shape: &[usize]) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::dynamic::DynamicDim;
    
    #[test]
    fn test_static_dim() {
        let dim = StaticDim::new([2, 3, 4]);
        assert_eq!(dim.ndim(), 3);
        assert_eq!(dim.size(), 24);
        assert_eq!(dim.shape(), &[2, 3, 4]);
    }
    
    #[test]
    fn test_dynamic_dim() {
        let dim = DynamicDim::new(vec![2, 3, 4]);
        assert_eq!(dim.ndim(), 3);
        assert_eq!(dim.size(), 24);
        assert_eq!(dim.shape(), &vec![2, 3, 4]);
    }
    
    #[test]
    fn test_broadcast_static() {
        let dim = StaticDim::new([1, 3, 1]);
        let target = [2, 3, 4];
        let broadcasted = dim.broadcast_to(&target).unwrap();
        assert_eq!(broadcasted.shape(), &[2, 3, 4]);
    }
    
    #[test]
    fn test_broadcast_dynamic() {
        let dim = DynamicDim::new(vec![1, 3, 1]);
        let target = vec![2, 3, 4];
        let broadcasted = dim.broadcast_to(&target).unwrap();
        assert_eq!(broadcasted.shape(), &vec![2, 3, 4]);
    }
}
