//! Static dimension handling for tensors.
//!
//! This module provides types and utilities for working with statically-sized
//! tensor dimensions, which are determined at compile time.

use std::fmt;
use crate::error::{Result, TensorustError};
use super::{Dimension, Stride};

/// A statically-sized dimension for tensors.
///
/// This type allows for tensor shapes and dimensions to be determined at compile time,
/// providing greater performance and safety.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
    pub fn with_strides(
        shape: [usize; N],
        strides: [usize; N],
    ) -> Result<Self> {
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
        N
    }

    /// Returns the total number of elements in the dimension.
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<const N: usize> Dimension for StaticDim<N> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn from_shape(shape: &[usize]) -> Result<Self> {
        if shape.len() != N {
            return Err(TensorustError::invalid_shape(
                "Shape length does not match static dimension",
            ));
        }
        let mut new_shape = [0; N];
        new_shape.copy_from_slice(shape);
        Ok(Self::new(new_shape))
    }
}

impl<const N: usize> fmt::Display for StaticDim<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.shape)
    }
}

impl<const N: usize> From<[usize; N]> for StaticDim<N> {
    fn from(shape: [usize; N]) -> Self {
        Self::new(shape)
    }
}
