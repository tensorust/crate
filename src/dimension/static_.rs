//! Statically-sized dimensions.

use super::{Dimension, Shape, Stride};
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

/// A dimension with a size known at compile time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StaticDim<const N: usize> {
    shape: [usize; N],
    strides: Stride,
}

impl<const N: usize> StaticDim<N> {
    /// Creates a new `StaticDim` from the given shape.
    pub fn new(shape: [usize; N]) -> Self {
        let strides = Stride::from_shape(&shape);
        Self { shape, strides }
    }
}

impl<const N: usize> Dimension for StaticDim<N> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[usize] {
        &self.strides
    }

    fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    fn from_shape(shape: &[usize]) -> Self {
        assert_eq!(shape.len(), N, "Incorrect number of dimensions");
        let mut new_shape = [0; N];
        new_shape.copy_from_slice(shape);
        Self::new(new_shape)
    }
}

impl<const N: usize> fmt::Display for StaticDim<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:?})", self.shape)
    }
}

impl<const N: usize> From<[usize; N]> for StaticDim<N> {
    fn from(shape: [usize; N]) -> Self {
        Self::new(shape)
    }
}

impl<const N: usize> Deref for StaticDim<N> {
    type Target = [usize; N];

    fn deref(&self) -> &Self::Target {
        &self.shape
    }
}

impl<const N: usize> DerefMut for StaticDim<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shape
    }
}
