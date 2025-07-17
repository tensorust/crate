//! Automatic differentiation module for Tensorust.
//! This module implements reverse-mode automatic differentiation (backpropagation)
//! for computing gradients of scalar-valued functions with respect to their inputs.

use crate::{
    dimension::Dimension,
    error::{Result, TensorustError},
    expression::{ExpressionGraph, Node},
    storage::Storage,
    tensor::Tensor,
};
use std::sync::Arc;

/// A differentiable variable that tracks operations for automatic differentiation.
#[derive(Debug, Clone)]
pub struct Variable<T, D, S = crate::storage::CpuStorage<T>>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    /// The underlying tensor data
    tensor: Tensor<T, D, S>,
    /// Whether this variable requires gradient computation
    requires_grad: bool,
    /// The computation graph node
    node: Arc<Node<T, D, S>>,
}

impl<T, D, S> Variable<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    /// Create a new variable from a tensor.
    pub fn new(tensor: Tensor<T, D, S>) -> Self {
        let node = Node::new(tensor.clone());
        Self {
            tensor,
            requires_grad: false,
            node: Arc::new(node),
        }
    }

    /// Set whether this variable requires gradient computation.
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Get a reference to the underlying tensor.
    pub fn tensor(&self) -> &Tensor<T, D, S> {
        &self.tensor
    }

    /// Get a mutable reference to the underlying tensor.
    /// This will detach the variable from the computation graph.
    pub fn tensor_mut(&mut self) -> &mut Tensor<T, D, S> {
        self.requires_grad = false;
        &mut self.tensor
    }

    /// Get a reference to the computation graph node.
    pub fn node(&self) -> &Arc<Node<T, D, S>> {
        &self.node
    }
}

/// A scalar value that tracks operations for automatic differentiation.
pub type Scalar = Variable<f32, crate::dimension::StaticDim<0>>;

/// Trait for types that can compute gradients.
pub trait Differentiable<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    /// Compute the gradient of this value with respect to all variables that require gradients.
    fn backward(&self) -> Result<()>;

    /// Get the gradient of this value.
    fn grad(&self) -> Option<Tensor<T, D, S>>;
}

impl<T, D, S> Differentiable<T, D, S> for Variable<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    fn backward(&self) -> Result<()> {
        if !self.requires_grad {
            return Err(TensorustError::gradient_error(
                "Cannot compute gradient for variable that doesn't require gradient",
            ));
        }

        // Initialize gradient if needed
        if self.node.grad().is_none() {
            // Create a tensor of ones with the same shape as the variable
            let ones = Tensor::ones(self.tensor.shape().clone())?;
            *self.node.grad_mut() = Some(ones);
        }

        // Perform backward pass
        self.node.backward(None)
    }

    fn grad(&self) -> Option<Tensor<T, D, S>> {
        self.node.grad().cloned()
    }
}

/// Context for automatic differentiation.
pub struct AutogradContext {
    graph: ExpressionGraph,
}

impl Default for AutogradContext {
    fn default() -> Self {
        Self::new()
    }
}

impl AutogradContext {
    /// Create a new autograd context.
    pub fn new() -> Self {
        Self {
            graph: ExpressionGraph::new(),
        }
    }

    /// Create a new variable in this context.
    pub fn variable<T, D, S>(&self, tensor: Tensor<T, D, S>) -> Variable<T, D, S>
    where
        T: Clone + Default + Send + Sync + 'static,
        D: Dimension + 'static,
        S: Storage<T> + 'static,
    {
        Variable::new(tensor)
    }

    /// Run a closure with gradient computation enabled.
    pub fn with_grad<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // In a real implementation, this would set up the gradient tape
        f()
    }

    /// Clear the computation graph.
    pub fn clear(&self) {
        self.graph.clear();
    }
}

/// Gradient functions for common operations
pub mod grad_fns {
    use super::*;

    /// Gradient function for addition.
    pub fn add_grad<T, D, S>(
        grad: Tensor<T, D, S>,
        _left: &Tensor<T, D, S>,
        _right: &Tensor<T, D, S>,
    ) -> Result<(Tensor<T, D, S>, Tensor<T, D, S>)>
    where
        T: Clone + Default + Send + Sync + 'static,
        D: Dimension + 'static,
        S: Storage<T> + 'static,
    {
        // For addition, the gradient is passed through to both operands
        Ok((grad.clone(), grad))
    }

    /// Gradient function for multiplication.
    pub fn mul_grad<T, D, S>(
        grad: Tensor<T, D, S>,
        left: &Tensor<T, D, S>,
        right: &Tensor<T, D, S>,
    ) -> Result<(Tensor<T, D, S>, Tensor<T, D, S>)>
    where
        T: Clone + Default + std::ops::Mul<Output = T> + Send + Sync + 'static,
        D: Dimension + 'static,
        S: Storage<T> + 'static,
    {
        // ∂(a*b)/∂a = b * grad
        // ∂(a*b)/∂b = a * grad
        let grad_a = right * &grad;
        let grad_b = left * &grad;
        Ok((grad_a, grad_b))
    }

    // Add more gradient functions for other operations as needed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dimension::DynamicDim,
        storage::CpuStorage,
        tensor::Tensor,
    };

    #[test]
    fn test_variable_creation() {
        let ctx = AutogradContext::new();
        let t = Tensor::<f32, DynamicDim, _>::from(vec![1.0, 2.0, 3.0]);
        let var = ctx.variable(t);
        assert_eq!(var.tensor().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_backward() {
        let ctx = AutogradContext::new();
        
        // Create variables
        let x = ctx.variable(Tensor::from(2.0f32)).requires_grad(true);
        let y = ctx.variable(Tensor::from(3.0f32)).requires_grad(true);
        
        // Perform computation
        let z = {
            let x_tensor = x.tensor().clone();
            let y_tensor = y.tensor().clone();
            ctx.variable((&x_tensor * &y_tensor).unwrap())
        };
        
        // Compute gradients
        z.backward().unwrap();
        
        // Check gradients
        assert_eq!(x.grad().unwrap().to_scalar(), 3.0); // dz/dx = y = 3.0
        assert_eq!(y.grad().unwrap().to_scalar(), 2.0); // dz/dy = x = 2.0
    }
    
    #[test]
    fn test_computation_graph() {
        let ctx = AutogradContext::new();
        
        // Create variables
        let a = ctx.variable(Tensor::from(2.0f32)).requires_grad(true);
        let b = ctx.variable(Tensor::from(3.0f32)).requires_grad(true);
        
        // Build computation: c = a * b + a
        let c = {
            let a_tensor = a.tensor().clone();
            let b_tensor = b.tensor().clone();
            let ab = (&a_tensor * &b_tensor).unwrap();
            let aba = (&ab + &a_tensor).unwrap();
            ctx.variable(aba)
        };
        
        // Compute gradients
        c.backward().unwrap();
        
        // Check gradients
        // c = a*b + a
        // dc/da = b + 1 = 3 + 1 = 4
        // dc/db = a = 2
        assert_eq!(a.grad().unwrap().to_scalar(), 4.0);
        assert_eq!(b.grad().unwrap().to_scalar(), 2.0);
    }
}
