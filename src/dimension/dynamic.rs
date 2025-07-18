//! Dynamic dimension handling for tensors.
//!
//! This module provides types and utilities for working with dynamically-sized
//! tensor dimensions, which are determined at runtime rather than compile time.

use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use crate::error::{Result, TensorustError};
use super::{Dimension, Stride};

/// A dynamically-sized dimension for tensors.
///
/// This type allows for tensor shapes and dimensions to be determined at runtime,
/// providing flexibility at the cost of some compile-time safety.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynamicDim {
    shape: Vec<usize>,
    strides: Option<Vec<usize>>,
}

impl DynamicDim {
    /// Creates a new dynamic dimension with the given shape.
    pub fn new(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        Self {
            shape,
            strides: None,
        }
    }
    
    /// Creates a new dynamic dimension with custom strides.
    pub fn with_strides(
        shape: impl Into<Vec<usize>>,
        strides: impl Into<Vec<usize>>,
    ) -> Result<Self> {
        let shape = shape.into();
        let strides = strides.into();
        
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
    
    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Returns the total number of elements in the dimension.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Validates if this dimension is compatible with another.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
    
    /// Broadcasts this dimension to match the target shape if possible.
    pub fn broadcast_to(&self, target: &[usize]) -> Result<Self> {
        if self.shape.len() > target.len() {
            return Err(TensorustError::broadcast_error(
                self.shape.clone(),
                target.to_vec(),
            ));
        }
        
        let mut new_shape = target.to_vec();
        let offset = target.len() - self.shape.len();
        
        for (i, &dim) in self.shape.iter().enumerate() {
            let target_dim = target[offset + i];
            
            if dim != 1 && dim != target_dim {
                return Err(TensorustError::broadcast_error(
                    self.shape.clone(),
                    target.to_vec(),
                ));
            }
            
            new_shape[offset + i] = target_dim;
        }
        
        Ok(Self::new(new_shape))
    }
    
    /// Computes the strides for this dimension assuming row-major order.
    pub fn compute_strides(&self) -> Stride {
        if let Some(strides) = &self.strides {
            return Stride::from_slice(strides);
        }
        
        let ndim = self.shape.len();
        let mut strides = vec![0; ndim];
        
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1].wrapping_mul(self.shape[i + 1]);
            }
        }
        
        Stride::from_vec(strides)
    }
    
    /// Permutes the dimensions according to the given permutation.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self> {
        if permutation.len() != self.shape.len() {
            return Err(TensorustError::invalid_input(
                "Permutation length must match number of dimensions",
            ));
        }
        
        let mut new_shape = vec![0; self.shape.len()];
        let mut new_strides = if let Some(strides) = &self.strides {
            Some(vec![0; strides.len()])
        } else {
            None
        };
        
        for (i, &p) in permutation.iter().enumerate() {
            if p >= self.shape.len() {
                return Err(TensorustError::invalid_input("Invalid permutation index"));
            }
            
            new_shape[i] = self.shape[p];
            if let (Some(strides), Some(ref mut new_strides)) = (&self.strides, &mut new_strides) {
                new_strides[i] = strides[p];
            }
        }
        
        if let Some(strides) = new_strides {
            Self::with_strides(new_shape, strides)
        } else {
            Ok(Self::new(new_shape))
        }
    }
    
    /// Reshapes the dimension to the new shape if possible.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let size = self.size();
        let new_size: usize = new_shape.iter().product();
        
        if size != new_size {
            return Err(TensorustError::reshape_error(
                self.shape.clone(),
                new_shape.to_vec(),
            ));
        }
        
        Ok(Self::new(new_shape.to_vec()))
    }
}

impl Dimension for DynamicDim {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn from_shape(shape: &[usize]) -> Result<Self> {
        Ok(Self::new(shape.to_vec()))
    }
}

impl fmt::Display for DynamicDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.shape)
    }
}

impl<T: Into<Vec<usize>>> From<T> for DynamicDim {
    fn from(shape: T) -> Self {
        Self::new(shape.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dynamic_dim_creation() {
        let dim = DynamicDim::new(vec![2, 3, 4]);
        assert_eq!(dim.ndim(), 3);
        assert_eq!(dim.size(), 24);
        assert_eq!(dim.as_slice(), &[2, 3, 4]);
    }
    
    #[test]
    fn test_compute_strides() {
        let dim = DynamicDim::new(vec![2, 3, 4]);
        let strides = dim.compute_strides();
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
        
        let dim = DynamicDim::with_strides(vec![2, 3, 4], vec![24, 8, 2]).unwrap();
        let strides = dim.compute_strides();
        assert_eq!(strides.as_slice(), &[24, 8, 2]);
    }
    
    #[test]
    fn test_permute() {
        let dim = DynamicDim::new(vec![2, 3, 4]);
        let permuted = dim.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.as_slice(), &[4, 2, 3]);
        
        let dim = DynamicDim::with_strides(vec![2, 3, 4], vec![12, 4, 1]).unwrap();
        let permuted = dim.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.compute_strides().as_slice(), &[1, 12, 4]);
    }
    
    #[test]
    fn test_reshape() {
        let dim = DynamicDim::new(vec![2, 3, 4]);
        let reshaped = dim.reshape(&[6, 4]).unwrap();
        assert_eq!(reshaped.as_slice(), &[6, 4]);
        
        assert!(dim.reshape(&[5, 5]).is_err());
    }
    
    #[test]
    fn test_broadcast() {
        let dim = DynamicDim::new(vec![1, 3, 1]);
        let broadcasted = dim.broadcast_to(&[2, 3, 4]).unwrap();
        assert_eq!(broadcasted.as_slice(), &[2, 3, 4]);
        
        let dim = DynamicDim::new(vec![3, 1]);
        let broadcasted = dim.broadcast_to(&[2, 3, 4]).unwrap();
        assert_eq!(broadcasted.as_slice(), &[2, 3, 4]);
        
        let dim = DynamicDim::new(vec![3, 4]);
        assert!(dim.broadcast_to(&[2, 3, 5]).is_err());
    }
}
