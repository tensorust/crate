//! Core tensor type that integrates the dimension system, view system, and expression graph.
//!
//! This module provides the main `Tensor` type that serves as the primary interface
//! for tensor operations in Tensorust.

use crate::{
    dimension::{dynamic::DynamicDim, static_dim::StaticDim, Dimension},
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
    pub fn new(storage: S, dim: D) -> Result<Self> {
        if storage.len() != dim.size() {
            return Err(TensorustError::ShapeMismatch {
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

    /// Create a new tensor from a vector
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let dim = D::from_shape(shape)?;
        let storage = S::from_vec(data);
        Self::new(storage, dim)
    }

    /// Create a view of this tensor with the given shape and stride
    pub fn view<D2>(&self, shape: D2::Shape) -> Result<Tensor<T, D2, S>>
    where
        D2: Dimension,
    {
        let dim = D2::from_shape(shape)?;
        
        if dim.size() != self.dim.size() {
            return Err(TensorustError::shape_mismatch(
                vec![self.dim.size()],
                vec![dim.size()],
            ));
        }

        Ok(Tensor {
            storage: self.storage.clone(),
            dim,
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
            grad: self.grad.clone(),
            is_view: true,
            expr: self.expr.clone(),
            _marker: PhantomData,
        })
    }

    /// Convert this tensor into a view
    pub fn into_view<D2>(self) -> Result<Tensor<T, D2, S>>
    where
        D2: Dimension,
    {
        self.view(self.dim.shape().to_vec())
    }

    /// Get a reference to the underlying storage
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.dim.shape()
    }

    /// Get the strides of the tensor
    pub fn strides(&self) -> &[usize] {
        self.dim.strides()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.dim.size()
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        self.dim.is_contiguous()
    }

    /// Convert the tensor to a contiguous tensor if it's not already
    pub fn contiguous(self) -> Self {
        if self.is_contiguous() {
            return self;
        }
        // TODO: Implement actual copying to make it contiguous
        self
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: D::Shape) -> Result<Self> 
    where
        T: num_traits::Zero + Clone,
    {
        let dim = D::from_shape(shape)?;
        let size = dim.size();
        let storage = S::from_vec(vec![T::zero(); size]);
        Self::new(storage, dim)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: D::Shape) -> Result<Self> 
    where
        T: num_traits::One + Clone,
    {
        let dim = D::from_shape(shape)?;
        let size = dim.size();
        let storage = S::from_vec(vec![T::one(); size]);
        Self::new(storage, dim)
    }

    /// Create a tensor with uninitialized values
    pub fn empty(shape: D::Shape) -> Result<Self> {
        let dim = D::from_shape(shape)?;
        let size = dim.size();
        let storage = S::from_vec(Vec::with_capacity(size));
        Self::new(storage, dim)
    }
    
    /// Create a tensor with values from a function
    pub fn from_fn<F>(shape: D::Shape, mut f: F) -> Result<Self>
    where
        F: FnMut() -> T,
    {
        let dim = D::from_shape(shape)?;
        let size = dim.size();
        let data: Vec<T> = (0..size).map(|_| f()).collect();
        let storage = S::from_vec(data);
        Self::new(storage, dim)
    }
    
    /// Create a tensor with a constant value
    pub fn full(shape: D::Shape, value: T) -> Result<Self>
    where
        T: Clone,
    {
        let dim = D::from_shape(shape)?;
        let size = dim.size();
        let storage = S::from_vec(vec![value; size]);
        Self::new(storage, dim)
    }
    
    /// Create an identity matrix
    pub fn eye(n: usize) -> Result<Self>
    where
        T: num_traits::Zero + num_traits::One + Clone,
    {
        let dim = D::from_shape(vec![n, n])?;
        let size = dim.size();
        let mut data = vec![T::zero(); size];
        
        for i in 0..n {
            data[i * n + i] = T::one();
        }
        
        let storage = S::from_vec(data);
        Self::new(storage, dim)
    }

    /// Get the shape of the tensor as a slice
    pub fn shape(&self) -> &[usize] {
        self.dim.shape()
    }
    
    /// Get the strides of the tensor as a slice
    pub fn strides(&self) -> &[usize] {
        self.dim.strides()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.dim.size()
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Check if the tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        self.dim.is_contiguous()
    }
    
    /// Convert the tensor to a contiguous tensor if it's not already
    pub fn contiguous(self) -> Self {
        if self.is_contiguous() {
            return self;
        }
        // TODO: Implement actual copying to make it contiguous
        self
    }

    /// Reshape the tensor
    pub fn reshape<D2>(self, new_shape: D2::Shape) -> Result<Tensor<T, D2, S>>
    where
        D2: Dimension,
    {
        let new_dim = D2::from_shape(new_shape)?;
        
        if new_dim.size() != self.dim.size() {
            return Err(TensorustError::shape_mismatch(
                vec![self.dim.size()],
                vec![new_dim.size()],
            ));
        }

        Ok(Tensor {
            storage: self.storage,
            dim: new_dim,
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn,
            grad: self.grad,
            is_view: true, // Reshape creates a view
            expr: self.expr,
            _marker: PhantomData,
        })
    }
    
    /// Transpose the tensor by reversing the dimensions
    pub fn t(self) -> Self {
        let new_dim = self.dim.transpose();
        Tensor {
            storage: self.storage,
            dim: new_dim,
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn,
            grad: self.grad,
            is_view: true, // Transpose creates a view
            expr: self.expr,
            _marker: PhantomData,
        }
    }
    
    /// Permute the dimensions of the tensor
    pub fn permute(self, dims: &[usize]) -> Result<Self> {
        let new_dim = self.dim.permute(dims)?;
        Ok(Tensor {
            storage: self.storage,
            dim: new_dim,
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn,
            grad: self.grad,
            is_view: true, // Permute creates a view
            expr: self.expr,
            _marker: PhantomData,
        })
    }
    
    /// Squeeze the tensor by removing dimensions of size 1
    pub fn squeeze(self, dim: Option<usize>) -> Self {
        let new_dim = self.dim.squeeze(dim);
        Tensor {
            storage: self.storage,
            dim: new_dim,
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn,
            grad: self.grad,
            is_view: true, // Squeeze creates a view
            expr: self.expr,
            _marker: PhantomData,
        }
    }
    
    /// Unsqueeze the tensor by adding a dimension of size 1
    pub fn unsqueeze(self, dim: usize) -> Result<Self> {
        let new_dim = self.dim.unsqueeze(dim)?;
        Ok(Tensor {
            storage: self.storage,
            dim: new_dim,
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn,
            grad: self.grad,
            is_view: true, // Unsqueeze creates a view
            expr: self.expr,
            _marker: PhantomData,
        })
    }

    /// Transpose the tensor by reversing the dimensions
    pub fn t(self) -> Self {
        let new_dim = self.dim.transpose();
        Tensor {
            storage: self.storage,
            dim: new_dim,
            offset: self.offset,
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn,
            grad: self.grad,
            is_view: true, // Transpose creates a view
            _marker: PhantomData,
        }
    }

    /// Set whether this tensor requires gradient computation
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        if requires_grad && self.grad.is_none() {
            self.grad = Some(Arc::new(std::sync::RwLock::new(None)));
        }
        self
    }

    /// Get a reference to the gradient if it exists
    pub fn grad(&self) -> Option<std::sync::RwLockReadGuard<Option<Tensor<T, D, CpuStorage<T>>>>> {
        self.grad.as_ref().map(|grad| grad.read().unwrap())
    }

    /// Get a mutable reference to the gradient if it exists
    pub fn grad_mut(&self) -> Option<std::sync::RwLockWriteGuard<Option<Tensor<T, D, CpuStorage<T>>>>> {
        self.grad.as_ref().map(|grad| grad.write().unwrap())
    }

    /// Backward pass for automatic differentiation
    pub fn backward(&self) -> Result<()> {
        if !self.requires_grad {
            return Err(TensorustError::gradient_error(
                "Cannot call backward on a tensor that doesn't require gradient",
            ));
        }

        // Initialize gradient if needed
        if self.grad.is_none() {
            return Err(TensorustError::gradient_error(
                "No gradient tensor allocated",
            ));
        }

        // Initialize gradient to ones if it's a scalar
        if self.len() == 1 {
            let mut grad = self.grad_mut().unwrap();
            if grad.is_none() {
                *grad = Some(Tensor::ones(self.dim.shape().to_vec())?);
            }
        }

        // Call the gradient function if it exists
        if let Some(grad_fn) = &self.grad_fn {
            grad_fn()?;
        }

        Ok(())
    }
    
    /// Get a reference to the underlying data as a slice
    pub fn data(&self) -> &[T] {
        &self.storage.as_slice()[self.offset..self.offset + self.len()]
    }
    
    /// Get a mutable reference to the underlying data as a slice
    pub fn data_mut(&mut self) -> &mut [T] {
        // This is unsafe because we need to ensure no other references exist
        // and that the tensor owns its data (not a view)
        assert!(!self.is_view, "Cannot get mutable data from a view");
        let slice = Arc::get_mut(&mut self.storage)
            .expect("Cannot get mutable reference to storage")
            .as_mut_slice();
        &mut slice[self.offset..self.offset + self.len()]
    }
    
    /// Convert the tensor to a vector, consuming it
    pub fn into_vec(self) -> Vec<T> {
        if self.is_contiguous() && !self.is_view && Arc::strong_count(&self.storage) == 1 {
            // If we own the storage and it's contiguous, we can take it directly
            Arc::try_unwrap(self.storage)
                .map(|s| s.into_vec())
                .unwrap_or_else(|_| self.data().to_vec())
        } else {
            // Otherwise, we need to copy the data
            self.data().to_vec()
        }
    }
    
    /// Get the value at a specific index
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        let flat_idx = self.dim.flat_index(index)?;
        self.storage.as_slice().get(self.offset + flat_idx)
    }
    
    /// Get a mutable reference to the value at a specific index
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        let flat_idx = self.dim.flat_index(index)?;
        Arc::get_mut(&mut self.storage)
            .and_then(|s| s.as_mut_slice().get_mut(self.offset + flat_idx))
    }
    
    /// Set the value at a specific index
    pub fn set(&mut self, index: &[usize], value: T) -> Result<()> {
        if let Some(ptr) = self.get_mut(index) {
            *ptr = value;
            Ok(())
        } else {
            Err(TensorustError::index_error("Index out of bounds"))
        }
    }
}

// Implement Display for Tensor
impl<T, D, S> fmt::Display for Tensor<T, D, S>
where
    T: fmt::Display + Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Simple display for now, can be enhanced for better formatting
        write!(f, "Tensor(shape={:?})", self.shape())
    }
}

// Implement Index for Tensor
impl<T, D, S> Index<usize> for Tensor<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.storage.as_slice()[self.offset + index]
    }
}

// Implement IndexMut for Tensor
impl<T, D, S> IndexMut<usize> for Tensor<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(!self.is_view, "Cannot index mutably into a view");
        &mut Arc::get_mut(&mut self.storage).unwrap().as_mut_slice()[self.offset + index]
    }
}

// Implement arithmetic operations with broadcasting
macro_rules! impl_binary_op {
    ($trait:ident, $op:ident, $method:ident) => {
        impl<T, D1, D2, S1, S2> $trait<&Tensor<T, D2, S2>> for &Tensor<T, D1, S1>
        where
            T: Clone + Default + std::ops::$trait<Output = T> + Send + Sync + 'static,
            D1: Dimension,
            D2: Dimension,
            S1: Storage<T>,
            S2: Storage<T>,
        {
            type Output = Tensor<T, DynamicDim, CpuStorage<T>>;
            
            fn $method(self, rhs: &Tensor<T, D2, S2>) -> Self::Output {
                // For now, implement a simple version without broadcasting
                // A full implementation would handle broadcasting and different storage types
                assert_eq!(self.shape(), rhs.shape(), "Shapes must match for operation");
                
                let result_data: Vec<T> = self.data()
                    .iter()
                    .zip(rhs.data().iter())
                    .map(|(a, b)| a.clone().$method(b.clone()))
                    .collect();
                
                Tensor::from_vec(result_data, self.shape().to_vec()).unwrap()
            }
        }
    };
}

// Generate implementations for Add, Sub, Mul, Div
impl_binary_op!(Add, add, add);
impl_binary_op!(Sub, sub, sub);
impl_binary_op!(Mul, mul, mul);
impl_binary_op!(Div, div, div);

// Implement unary operations
impl<T, D, S> Neg for &Tensor<T, D, S>
where
    T: Clone + Default + std::ops::Neg<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    type Output = Tensor<T, D, CpuStorage<T>>;
    
    fn neg(self) -> Self::Output {
        let result_data: Vec<T> = self.data()
            .iter()
            .map(|x| -x.clone())
            .collect();
        
        Tensor::from_vec(result_data, self.shape().to_vec()).unwrap()
    }
}

// Implement in-place operations
impl<T, D, S> Tensor<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Add another tensor to this one in-place
    pub fn add_(&mut self, other: &Self) -> Result<()>
    where
        T: std::ops::Add<Output = T>,
    {
        assert_eq!(self.shape(), other.shape(), "Shapes must match for add_");
        
        for (a, b) in self.data_mut().iter_mut().zip(other.data().iter()) {
            *a = a.clone() + b.clone();
        }
        
        Ok(())
    }
    
    /// Multiply this tensor by another tensor in-place
    pub fn mul_(&mut self, other: &Self) -> Result<()>
    where
        T: std::ops::Mul<Output = T>,
    {
        assert_eq!(self.shape(), other.shape(), "Shapes must match for mul_");
        
        for (a, b) in self.data_mut().iter_mut().zip(other.data().iter()) {
            *a = a.clone() * b.clone();
        }
        
        Ok(())
    }
    
    /// Apply a function element-wise to the tensor
    pub fn apply<F>(&mut self, f: F) -> &mut Self
  where
      F: Fn(T) -> T,
  {
      for x in self.data_mut().iter_mut() {
          *x = f(x.clone());
      }
      self
  }
  
  /// Map a function over the tensor, producing a new tensor
  pub fn map<F, U>(&self, f: F) -> Tensor<U, D, CpuStorage<U>>
  where
      F: Fn(T) -> U,
      U: Clone + Default + Send + Sync + 'static,
  {
      let result_data: Vec<U> = self.data().iter().map(|x| f(x.clone())).collect();
      Tensor::from_vec(result_data, self.shape().to_vec()).unwrap()
  }
  
  /// Reduce the tensor using a function
  pub fn reduce<F, U>(&self, init: U, f: F) -> U
  where
      F: Fn(U, &T) -> U,
      U: Clone,
  {
      self.data().iter().fold(init, f)
  }
  
  /// Sum all elements in the tensor
  pub fn sum_elements(&self) -> T
  where
      T: std::iter::Sum,
  {
      self.data().iter().cloned().sum()
  }
  
  /// Compute the mean of all elements in the tensor
  pub fn mean_elements(&self) -> T
  where
      T: num_traits::Float,
  {
      let sum = self.sum_elements();
      let count = T::from(self.len()).unwrap();
      sum / count
  }
  
  /// Compute the maximum element in the tensor
  pub fn max_element(&self) -> Option<T>
  where
      T: PartialOrd,
  {
      self.data().iter().max_by(|a, b| a.partial_cmp(b).unwrap()).cloned()
  }
  
  /// Compute the minimum element in the tensor
  pub fn min_element(&self) -> Option<T>
  where
      T: PartialOrd,
  {
      self.data().iter().min_by(|a, b| a.partial_cmp(b).unwrap()).cloned()
  }
}

// Implement expression graph integration
impl<T, D, S> Tensor<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Convert this tensor to an expression node
    pub fn to_expr(&self) -> Expr {
        if let Some(expr) = &self.expr {
            return expr.clone();
        }
        
        // Create a new input node for this tensor
        let node = crate::expression::nodes::InputNode::new(self.shape().to_vec());
        Arc::new(node)
    }
    
    /// Evaluate an expression and store the result in this tensor
    pub fn eval_expr(&mut self, expr: Expr) -> Result<()> {
        // Evaluate the expression
        let result = expr.eval()?;
        
        // Copy the result into this tensor
        // This assumes the shapes match and the types are compatible
        // In a real implementation, you'd want to handle these cases properly
        self.data_mut().copy_from_slice(&result.data());
        
        Ok(())
    }
}

// Implement From for common types
impl<T, S> From<Vec<T>> for Tensor<T, DynamicDim, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
{
    fn from(data: Vec<T>) -> Self {
        let shape = vec![data.len()];
        let storage = S::from_vec(data);
        Tensor::new(storage, DynamicDim::from_shape(shape).unwrap()).unwrap()
    }
}

impl<const N: usize, T, S> From<[T; N]> for Tensor<T, StaticDim<1>, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
{
    fn from(data: [T; N]) -> Self {
        let shape = [N];
        let storage = S::from_vec(data.to_vec());
        Tensor::new(storage, StaticDim::from_shape(shape).unwrap()).unwrap()
    }
}

// Implement IntoIterator for Tensor
impl<T, D, S> IntoIterator for Tensor<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

// Implement IntoIterator for &Tensor
impl<'a, T, D, S> IntoIterator for &'a Tensor<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.data().to_vec().into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{DynamicDim, StaticDim};
    use crate::storage::CpuStorage;
    
    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t[0], 1.0);
        assert_eq!(t[1], 2.0);
        assert_eq!(t[2], 3.0);
    }
    
    #[test]
    fn test_tensor_operations() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        
        // Test addition
        let c = &a + &b;
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c[0], 5.0);
        assert_eq!(c[1], 7.0);
        assert_eq!(c[2], 9.0);
        
        // Test subtraction
        let d = &a - &b;
        assert_eq!(d[0], -3.0);
        assert_eq!(d[1], -3.0);
        assert_eq!(d[2], -3.0);
        
        // Test multiplication
        let e = &a * &b;
        assert_eq!(e[0], 4.0);
        assert_eq!(e[1], 10.0);
        assert_eq!(e[2], 18.0);
        
        // Test negation
        let f = -&a;
        assert_eq!(f[0], -1.0);
        assert_eq!(f[1], -2.0);
        assert_eq!(f[2], -3.0);
    }
    
    #[test]
    fn test_tensor_views() {
        let t = Tensor::from_vec((0..8).map(|x| x as f32).collect(), vec![2, 2, 2]).unwrap();
        
        // Test reshaping
        let v = t.reshape(vec![4, 2]).unwrap();
        assert_eq!(v.shape(), &[4, 2]);
        assert_eq!(v[[0, 0]], 0.0);
        assert_eq!(v[[3, 1]], 7.0);
        
        // Test transposing
        let t = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(t.shape(), &[2, 2, 2]);
        
        // Test squeezing/unsqueezing
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let t = t.squeeze(Some(1));
        assert_eq!(t.shape(), &[3]);
        
        let t = t.unsqueeze(0).unwrap();
        assert_eq!(t.shape(), &[1, 3]);
    }
    
    #[test]
    fn test_tensor_expression() {
        // Create input tensors
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        
        // Convert to expressions
        let a_expr = a.to_expr();
        let b_expr = b.to_expr();
        
        // Build expression graph
        let expr = a_expr.add(b_expr).sigmoid();
        
        // Evaluate the expression
        let mut result = Tensor::zeros(&[3]).unwrap();
        result.eval_expr(expr).unwrap();
        
        // Verify the result
        assert!((result[0] - 0.993307).abs() < 1e-6);
        assert!((result[1] - 0.999089).abs() < 1e-6);
        assert!((result[2] - 0.999877).abs() < 1e-6);
    }
}


// Implement other arithmetic operations (Sub, Mul, Div, Neg) similarly
// ...




