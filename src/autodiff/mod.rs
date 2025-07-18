//! Automatic differentiation module for Tensorust.
//!
//! This module provides the building blocks for automatic differentiation,
//! including the computation graph, gradient computation, and backpropagation.

mod graph;
mod ops;
mod tensor;

pub use graph::*;
pub use ops::*;
pub use tensor::*;

use std::sync::{Arc, Mutex, Weak};
use uuid::Uuid;

/// Represents a differentiable operation in the computation graph.
pub trait DifferentiableOp: std::fmt::Debug + Send + Sync {
    /// Computes the forward pass of the operation.
    fn forward(&self, inputs: &[&crate::tensor::Tensor]) -> Result<crate::tensor::Tensor, crate::error::TensorustError>;
    
    /// Computes the backward pass (gradient) of the operation.
    fn backward(&self, grad: &crate::tensor::Tensor, inputs: &[&crate::tensor::Tensor], output: &crate::tensor::Tensor) -> Vec<crate::tensor::Tensor>;
}

/// Error type for automatic differentiation operations.
#[derive(Debug, thiserror::Error)]
pub enum AutodiffError {
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    
    #[error("Operation failed: {0}")]
    OperationFailed(String),
    
    #[error("Gradient computation failed: {0}")]
    GradientError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// A node in the computation graph.
#[derive(Debug)]
pub struct Node {
    /// The unique ID of this node.
    pub id: Uuid,
    /// The tensor associated with this node.
    pub tensor: crate::tensor::Tensor,
    /// The operation that produced this node.
    pub op: Option<Arc<dyn DifferentiableOp>>,
    /// The inputs to the operation.
    pub inputs: Vec<Arc<Node>>,
    /// The gradient of this node.
    pub gradient: Option<crate::tensor::Tensor>,
    /// Whether this node requires a gradient.
    pub requires_grad: bool,
    /// A weak reference to this node's gradient function.
    grad_fn: Option<Weak<Mutex<Box<dyn FnMut() -> Result<(), crate::error::TensorustError>>>>>,
}

impl Node {
    /// Creates a new leaf node (input tensor).
    pub fn new_leaf(tensor: crate::tensor::Tensor) -> Arc<Self> {
        Arc::new(Self {
            id: Uuid::new_v4(),
            tensor,
            op: None,
            inputs: Vec::new(),
            gradient: None,
            requires_grad: true,
            grad_fn: None,
        })
    }

    /// Creates a new internal node (the result of an operation).
    pub fn new_internal(
        tensor: crate::tensor::Tensor,
        op: Arc<dyn DifferentiableOp>,
        inputs: Vec<Arc<Node>>,
    ) -> Arc<Self> {
        let requires_grad = inputs.iter().any(|i| i.requires_grad);
        Arc::new(Self {
            id: Uuid::new_v4(),
            tensor,
            op: Some(op),
            inputs,
            gradient: None,
            requires_grad,
            grad_fn: None,
        })
    }

    /// Backpropagates the gradient from this node.
    pub fn backward(&self, grad: Option<&crate::tensor::Tensor>) -> Result<(), crate::error::TensorustError> {
        if !self.requires_grad {
            return Ok(());
        }

        let grad = if let Some(grad) = grad {
            grad.clone()
        } else {
            // If no gradient is provided, assume it's 1.
            crate::tensor::Tensor::ones_like(&self.tensor).map_err(|e| {
                crate::error::TensorustError::Other(format!("Failed to create ones tensor: {}", e))
            })?
        };

        if let Some(grad_fn_weak) = &self.grad_fn {
            if let Some(grad_fn_arc) = grad_fn_weak.upgrade() {
                let mut grad_fn = grad_fn_arc.lock().unwrap();
                (*grad_fn)().map_err(|e| {
                    crate::error::TensorustError::Other(format!("Gradient computation failed: {}", e))
                })?;
            }
        }

        if let Some(op) = &self.op {
            let input_tensors: Vec<_> = self.inputs.iter().map(|i| &i.tensor).collect();
            let input_grads = op.backward(&grad, &input_tensors, &self.tensor);

            for (input_node, input_grad) in self.inputs.iter().zip(input_grads) {
                input_node.backward(Some(&input_grad))?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    
    #[test]
    fn test_computation_graph() {
        // Create a simple computation graph: z = x * y + 2
        let mut graph = ComputationGraph::new();
        
        // Input tensors
        let x = graph.add_tensor(Tensor::scalar(2.0));
        let y = graph.add_tensor(Tensor::scalar(3.0));
        
        // Operations
        let mul = graph.add_op(
            Arc::new(MultiplyOp),
            &[x.clone(), y.clone()],
        ).unwrap();
        
        let two = graph.add_tensor(Tensor::scalar(2.0));
        let z = graph.add_op(
            Arc::new(AddOp),
            &[mul, two],
        ).unwrap();
        
        // Forward pass
        assert_eq!(z.tensor.to_scalar::<f32>().unwrap(), 8.0);
        
        // Backward pass
        graph.backward(&z).unwrap();
        
        // Check gradients
        assert_eq!(x.gradient.as_ref().unwrap().to_scalar::<f32>().unwrap(), 3.0); // dz/dx = y = 3
        assert_eq!(y.gradient.as_ref().unwrap().to_scalar::<f32>().unwrap(), 2.0); // dz/dy = x = 2
    }
}
