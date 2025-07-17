//! Mathematical operations for tensors.
//! This module provides various mathematical operations that can be performed on tensors.

mod arithmetic;
mod math;
mod reduction;
mod elementwise;

pub use arithmetic::*;
pub use math::*;
pub use reduction::*;
pub use elementwise::*;

use crate::{
    dimension::Dimension,
    error::Result,
    tensor::Tensor,
    storage::Storage,
};

/// Trait for tensor operations that can be performed in-place.
pub trait InplaceOp<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Perform the operation in-place.
    fn run_inplace(&self, tensor: &mut Tensor<T, D, S>) -> Result<()>;
}

/// Trait for tensor operations that produce a new tensor.
pub trait MapOp<T, D, S, R = T>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    R: Clone + Send + Sync + 'static,
{
    /// Perform the operation and return a new tensor.
    fn run(&self, tensor: &Tensor<T, D, S>) -> Result<Tensor<R, D, S>>;
}

/// Trait for binary tensor operations.
pub trait BinaryOp<T, D, S, R = T>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    R: Clone + Send + Sync + 'static,
{
    /// Perform the binary operation and return a new tensor.
    fn run(
        &self,
        lhs: &Tensor<T, D, S>,
        rhs: &Tensor<T, D, S>,
    ) -> Result<Tensor<R, D, S>>;
}

/// Trait for reduction operations.
pub trait ReduceOp<T, D, S, R = T>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    R: Clone + Send + Sync + 'static,
{
    /// Perform the reduction operation and return a new tensor.
    fn run(
        &self,
        tensor: &Tensor<T, D, S>,
        dim: Option<usize>,
        keep_dim: bool,
    ) -> Result<Tensor<R, D, S>>;
}
