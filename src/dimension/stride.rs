//! Stride handling for tensors.
//!
//! This module provides types and utilities for working with tensor strides,
//! which define how to traverse memory when accessing tensor elements.

use std::fmt;
use std::ops::{Deref, Index};
use crate::error::{Result, TensorustError};

/// Represents the stride of a tensor.
///
/// Strides are the number of elements to skip in memory to move to the next
/// element along each dimension. For row-major order, the last dimension
/// has a stride of 1, and the second-to-last has a stride equal to the size
/// of the last dimension, and so on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Stride {
    strides: Vec<usize>,
}

impl Stride {
    /// Creates a new stride from a vector of strides.
    pub fn new(strides: Vec<usize>) -> Self {
        Self { strides }
    }
    
    /// Creates a new stride from a slice.
    pub fn from_slice(strides: &[usize]) -> Self {
        Self {
            strides: strides.to_vec(),
        }
    }
    
    /// Creates a new stride from a vector.
    pub fn from_vec(strides: Vec<usize>) -> Self {
        Self { strides }
    }
    
    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.strides.len()
    }
    
    /// Returns the stride values as a slice.
    pub fn as_slice(&self) -> &[usize] {
        &self.strides
    }
    
    /// Computes the offset for the given indices.
    pub fn offset<I: AsRef<[usize]>>(&self, indices: I) -> Result<usize> {
        let indices = indices.as_ref();
        if indices.len() != self.strides.len() {
            return Err(TensorustError::invalid_input(
                "Number of indices must match number of dimensions",
            ));
        }
        
        let mut offset = 0;
        for (&stride, &index) in self.strides.iter().zip(indices) {
            offset = offset.wrapping_add(stride.wrapping_mul(index));
        }
        
        Ok(offset)
    }
    
    /// Computes the default row-major strides for a given shape.
    pub fn row_major(shape: &[usize]) -> Self {
        let ndim = shape.len();
        let mut strides = vec![0; ndim];
        
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1].wrapping_mul(shape[i + 1]);
            }
        }
        
        Self { strides }
    }
    
    /// Computes the default column-major strides for a given shape.
    pub fn column_major(shape: &[usize]) -> Self {
        let ndim = shape.len();
        let mut strides = vec![0; ndim];
        
        if ndim > 0 {
            strides[0] = 1;
            for i in 1..ndim {
                strides[i] = strides[i - 1].wrapping_mul(shape[i - 1]);
            }
        }
        
        Self { strides }
    }
    
    /// Permutes the dimensions according to the given permutation.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self> {
        if permutation.len() != self.strides.len() {
            return Err(TensorustError::invalid_input(
                "Permutation length must match number of dimensions",
            ));
        }
        
        let mut new_strides = vec![0; self.strides.len()];
        for (i, &p) in permutation.iter().enumerate() {
            if p >= self.strides.len() {
                return Err(TensorustError::invalid_input("Invalid permutation index"));
            }
            new_strides[i] = self.strides[p];
        }
        
        Ok(Self { strides: new_strides })
    }
    
    /// Broadcasts the stride to a new shape.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        if shape.len() < self.strides.len() {
            return Err(TensorustError::broadcast_error(
                self.strides.clone(),
                shape.to_vec(),
            ));
        }
        
        let mut new_strides = vec![0; shape.len()];
        let offset = shape.len() - self.strides.len();
        
        for i in 0..shape.len() {
            if i < offset {
                new_strides[i] = 0;
            } else if shape[i] == 1 && self.strides[i - offset] != 0 {
                new_strides[i] = 0;
            } else {
                new_strides[i] = self.strides[i - offset];
            }
        }
        
        Ok(Self { strides: new_strides })
    }
}

impl Index<usize> for Stride {
    type Output = usize;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.strides[index]
    }
}

impl AsRef<[usize]> for Stride {
    fn as_ref(&self) -> &[usize] {
        &self.strides
    }
}

impl From<Vec<usize>> for Stride {
    fn from(strides: Vec<usize>) -> Self {
        Self { strides }
    }
}

impl From<&[usize]> for Stride {
    fn from(strides: &[usize]) -> Self {
        Self {
            strides: strides.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_row_major_strides() {
        let shape = [2, 3, 4];
        let strides = Stride::row_major(&shape);
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }
    
    #[test]
    fn test_column_major_strides() {
        let shape = [2, 3, 4];
        let strides = Stride::column_major(&shape);
        assert_eq!(strides.as_slice(), &[1, 2, 6]);
    }
    
    #[test]
    fn test_offset_calculation() {
        let strides = Stride::new(vec![12, 4, 1]);
        assert_eq!(strides.offset(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(strides.offset(&[1, 0, 0]).unwrap(), 12);
        assert_eq!(strides.offset(&[0, 1, 0]).unwrap(), 4);
        assert_eq!(strides.offset(&[0, 0, 1]).unwrap(), 1);
        assert_eq!(strides.offset(&[1, 2, 3]).unwrap(), 23);
    }
    
    #[test]
    fn test_permute() {
        let strides = Stride::new(vec![12, 4, 1]);
        let permuted = strides.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.as_slice(), &[1, 12, 4]);
    }
    
    #[test]
    fn test_broadcast() {
        let strides = Stride::new(vec![4, 1]);
        let broadcasted = strides.broadcast_to(&[2, 3, 4]).unwrap();
        assert_eq!(broadcasted.as_slice(), &[0, 4, 1]);
    }
}
