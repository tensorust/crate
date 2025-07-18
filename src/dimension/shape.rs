//! Shape handling for tensors.
//!
//! This module provides types and traits for working with tensor shapes,
//! including shape manipulation, validation, and broadcasting.

use std::fmt;
use std::ops::{Deref, DerefMut};
use crate::error::{Result, TensorustError};

/// A trait for types that can represent tensor shapes.
pub trait Shape: fmt::Debug + Clone + Send + Sync + 'static {
    /// Returns the number of dimensions.
    fn ndim(&self) -> usize;
    
    /// Returns the shape as a slice.
    fn as_slice(&self) -> &[usize];
    
    /// Validates if the shape is valid.
    fn validate(&self) -> Result<()> {
        for &dim in self.as_slice() {
            if dim == 0 {
                return Err(TensorustError::invalid_shape("Shape cannot have zero dimensions"));
            }
        }
        Ok(())
    }
    
    /// Computes the total number of elements in the shape.
    fn size(&self) -> usize {
        self.as_slice().iter().product()
    }
    
    /// Checks if this shape is compatible with another shape for operations like addition.
    fn is_compatible_with(&self, other: &dyn Shape) -> bool {
        let s1 = self.as_slice();
        let s2 = other.as_slice();
        
        if s1.len() != s2.len() {
            return false;
        }
        
        s1.iter().zip(s2.iter()).all(|(&a, &b)| a == b)
    }
    
    /// Broadcasts this shape to match the target shape if possible.
    fn broadcast_to(&self, target: &dyn Shape) -> Result<Vec<usize>> {
        let s1 = self.as_slice();
        let s2 = target.as_slice();
        
        if s1.len() > s2.len() {
            return Err(TensorustError::broadcast_error(
                s1.to_vec(),
                s2.to_vec(),
            ));
        }
        
        let mut result = s2.to_vec();
        let offset = s2.len() - s1.len();
        
        for (i, &dim) in s1.iter().enumerate() {
            let target_dim = s2[offset + i];
            
            if dim != 1 && dim != target_dim {
                return Err(TensorustError::broadcast_error(
                    s1.to_vec(),
                    s2.to_vec(),
                ));
            }
            
            result[offset + i] = target_dim;
        }
        
        Ok(result)
    }
}


impl<const N: usize> Shape for [usize; N] {
    fn ndim(&self) -> usize {
        N
    }
    
    fn as_slice(&self) -> &[usize] {
        self
    }
}

impl Shape for Vec<usize> {
    fn ndim(&self) -> usize {
        self.len()
    }
    
    fn as_slice(&self) -> &[usize] {
        self
    }
}

/// A wrapper type for shapes that enforces validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidShape<T: Shape> {
    shape: T,
}

impl<T: Shape> ValidShape<T> {
    /// Creates a new validated shape.
    pub fn new(shape: T) -> Result<Self> {
        shape.validate()?;
        Ok(Self { shape })
    }
    
    /// Returns a reference to the inner shape.
    pub fn inner(&self) -> &T {
        &self.shape
    }
    
    /// Converts back to the inner shape.
    pub fn into_inner(self) -> T {
        self.shape
    }
}

impl<T: Shape> Deref for ValidShape<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}

impl<T: Shape> AsRef<T> for ValidShape<T> {
    fn as_ref(&self) -> &T {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shape_validation() {
        assert!(ValidShape::new(vec![2, 3, 4]).is_ok());
        assert!(ValidShape::new(vec![0, 3, 4]).is_err());
    }
    
    #[test]
    fn test_broadcasting() {
        let shape1 = [1, 3, 1];
        let shape2 = [2, 3, 4];
        let result = shape1.broadcast_to(&shape2).unwrap();
        assert_eq!(result, vec![2, 3, 4]);
        
        let shape3 = [3, 1];
        let result = shape3.broadcast_to(&shape2).unwrap();
        assert_eq!(result, vec![2, 3, 4]);
        
        let shape4 = [3, 5];
        assert!(shape4.broadcast_to(&shape2).is_err());
    }
    
    #[test]
    fn test_valid_shape() {
        let shape = ValidShape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(shape.as_slice(), &[2, 3, 4]);
        assert_eq!(shape.size(), 24);
        
        assert!(ValidShape::new(vec![0, 3, 4]).is_err());
    }
}
