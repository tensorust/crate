//! Dimension system for tensors.
//!
//! This module provides types and traits for handling tensor dimensions,
//! including both static (compile-time) and dynamic (run-time) dimensions.
//! It supports operations like broadcasting, shape validation, and dimension manipulation.

use std::fmt;
use std::ops::{Add, Mul, Sub};
use crate::error::{Result, TensorustError};

pub mod dynamic;
pub mod shape;
pub mod stride;

pub use dynamic::DynamicDim;
pub use shape::Shape;
pub use stride::Stride;

/// Trait for types that can represent tensor dimensions.
pub trait Dimension: fmt::Debug + Clone + Send + Sync + 'static {
    /// The shape type for this dimension.
    type Shape: Shape;
    
    /// Returns the shape of the dimension.
    fn shape(&self) -> &Self::Shape;
    
    /// Returns the number of dimensions.
    fn ndim(&self) -> usize;
    
    /// Returns the total number of elements in the dimension.
    fn size(&self) -> usize;
    
    /// Validates if this dimension is compatible with another.
    fn is_compatible_with(&self, other: &Self) -> bool;
    
    /// Broadcasts this dimension to match the target shape if possible.
    fn broadcast_to(&self, target: &Self::Shape) -> Result<Self>;
    
    /// Computes the strides for this dimension assuming row-major order.
    fn compute_strides(&self) -> Stride;
}

/// A statically-sized dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticDim<const N: usize> {
    shape: [usize; N],
    strides: Option<[usize; N]>,
}

impl<const N: usize> StaticDim<N> {
    /// Creates a new static dimension with the given shape.
    pub fn new(shape: [usize; N]) -> Self {
        Self {
            shape,
            strides: None,
        }
    }
    
    /// Creates a new static dimension with custom strides.
    pub fn with_strides(shape: [usize; N], strides: [usize; N]) -> Self {
        Self {
            shape,
            strides: Some(strides),
        }
    }
    
    /// Returns the shape as a slice.
    pub fn as_slice(&self) -> &[usize] {
        &self.shape
    }
}

impl<const N: usize> Dimension for StaticDim<N> {
    type Shape = [usize; N];
    
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }
    
    fn ndim(&self) -> usize {
        N
    }
    
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
    
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
    
    fn broadcast_to(&self, target: &Self::Shape) -> Result<Self> {
        if self.shape.len() != target.len() {
            return Err(TensorustError::shape_mismatch(
                vec![self.shape.to_vec()],
                vec![target.to_vec()],
            ));
        }
        
        let mut new_shape = *target;
        for (i, (&s, &t)) in self.shape.iter().zip(target.iter()).enumerate() {
            if s != 1 && s != t {
                return Err(TensorustError::broadcast_error(
                    self.shape.to_vec(),
                    target.to_vec(),
                ));
            }
            new_shape[i] = t;
        }
        
        Ok(Self::new(new_shape))
    }
    
    fn compute_strides(&self) -> Stride {
        if let Some(strides) = &self.strides {
            return Stride::from_slice(strides);
        }
        
        let mut strides = vec![0; N];
        if N > 0 {
            strides[N - 1] = 1;
            for i in (0..N - 1).rev() {
                strides[i] = strides[i + 1] * self.shape[i + 1];
            }
        }
        
        Stride::from_vec(strides)
    }
}

/// A dynamically-sized dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicDim {
    shape: Vec<usize>,
    strides: Option<Vec<usize>>,
}

impl DynamicDim {
    /// Creates a new dynamic dimension with the given shape.
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            strides: None,
        }
    }
    
    /// Creates a new dynamic dimension with custom strides.
    pub fn with_strides(shape: Vec<usize>, strides: Vec<usize>) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(TensorustError::invalid_input(
                "Shape and strides must have the same length",
            ));
        }
        
        Ok(Self {
            shape,
            strides: Some(strides),
        })
    }
    
    /// Returns the shape as a slice.
    pub fn as_slice(&self) -> &[usize] {
        &self.shape
    }
}

impl Dimension for DynamicDim {
    type Shape = Vec<usize>;
    
    fn shape(&self) -> &Self::Shape {
        &self.shape
    }
    
    fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    fn size(&self) -> usize {
        self.shape.iter().product()
    }
    
    fn is_compatible_with(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
    
    fn broadcast_to(&self, target: &Self::Shape) -> Result<Self> {
        if self.shape.len() > target.len() {
            return Err(TensorustError::shape_mismatch(
                self.shape.clone(),
                target.clone(),
            ));
        }
        
        let mut new_shape = target.clone();
        let offset = target.len() - self.shape.len();
        
        for (i, &s) in self.shape.iter().enumerate() {
            let t = target[offset + i];
            if s != 1 && s != t {
                return Err(TensorustError::broadcast_error(
                    self.shape.clone(),
                    target.clone(),
                ));
            }
            new_shape[offset + i] = t;
        }
        
        Ok(Self::new(new_shape))
    }
    
    fn compute_strides(&self) -> Stride {
        if let Some(strides) = &self.strides {
            return Stride::from_vec(strides.clone());
        }
        
        let ndim = self.shape.len();
        let mut strides = vec![0; ndim];
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * self.shape[i + 1];
            }
        }
        
        Stride::from_vec(strides)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_static_dim() {
        let dim = StaticDim::new([2, 3, 4]);
        assert_eq!(dim.ndim(), 3);
        assert_eq!(dim.size(), 24);
        assert_eq!(dim.shape(), &[2, 3, 4]);
        
        let strides = dim.compute_strides();
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }
    
    #[test]
    fn test_dynamic_dim() {
        let dim = DynamicDim::new(vec![2, 3, 4]);
        assert_eq!(dim.ndim(), 3);
        assert_eq!(dim.size(), 24);
        assert_eq!(dim.shape(), &vec![2, 3, 4]);
        
        let strides = dim.compute_strides();
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
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
