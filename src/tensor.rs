//! Core tensor type that integrates the dimension system, view system, and expression graph.
//!
//! This module provides the main `Tensor` type that serves as the primary interface
//! for tensor operations in Tensorust.

use crate::{
    dimension::{dynamic::DynamicDim, Dimension},
    error::{Result, TensorustError},
    expression::{Expr, Evaluate, Optimize},
    storage::{CpuStorage, Storage},
    view::TensorView,
};
use std::{
    fmt,
    marker::PhantomData,
    ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub},
    sync::Arc,
};

/// A generic n-dimensional array.
///
/// `Tensor` is the central data structure in `tensorust`. It is a generic struct that
/// can be used to represent tensors of any data type, dimension, and storage backend.
///
/// # Type Parameters
///
/// * `T`: The data type of the tensor elements.
/// * `D`: The dimension of the tensor.
/// * `S`: The storage backend for the tensor data.
///
#[derive(Debug, Clone)]
pub struct Tensor<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    pub(crate) storage: Arc<S>,
    pub(crate) dim: D,
    pub(crate) marker: PhantomData<T>,
}

impl<T, D, S> Tensor<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Creates a new tensor from the given storage and dimension.
    ///
    /// # Arguments
    ///
    /// * `storage`: The storage backend for the tensor data.
    /// * `dim`: The dimension of the tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage length does not match the dimension size.
    pub fn new(storage: S, dim: D) -> Result<Self, TensorustError> {
        if storage.len() != dim.size() {
            return Err(TensorustError::ShapeMismatch{
                expected: vec![storage.len()],
                actual: vec![dim.size()],
            });
        }
        Ok(Self {
            storage: Arc::new(storage),
            dim,
            marker: PhantomData,
        })
    }
}
