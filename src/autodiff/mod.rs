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

use std::sync::Arc;

/// Represents a differentiable operation in the computation graph.
pub trait DifferentiableOp: std::fmt::Debug + Send + Sync {
    /// Computes the forward pass of the operation.
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, AutodiffError>;
    
    /// Computes the backward pass (gradient) of the operation.
    fn backward(&self, grad: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Tensor>;
    
    /// Returns the names of the input tensors.
    fn input_names(&self) -> Vec<&'static str>;
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
    /// The tensor value at this node.
    pub tensor: Tensor,
    
    /// The operation that produced this tensor (None for leaf nodes).
    pub op: Option<Arc<dyn DifferentiableOp>>,
    
    /// The input nodes to this operation.
    pub inputs: Vec<Arc<Node>>,
    
    /// The gradient of the loss with respect to this tensor.
    pub gradient: Option<Tensor>,
}

impl Node {
    /// Creates a new leaf node (input tensor).
    pub fn new_leaf(tensor: Tensor) -> Arc<Self> {
        Arc::new(Self {
            tensor,
            op: None,
            inputs: Vec::new(),
            gradient: None,
        })
    }
    
    /// Creates a new operation node.
    pub fn new_op(
        tensor: Tensor,
        op: Arc<dyn DifferentiableOp>,
        inputs: Vec<Arc<Node>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            tensor,
            op: Some(op),
            inputs,
            gradient: None,
        })
    }
    
    /// Computes the gradient of the loss with respect to this node.
    pub fn backward(&self, grad: Option<&Tensor>) -> Result<(), AutodiffError> {
        // Initialize gradient if not already set
        let grad = match (grad, &self.gradient) {
            (Some(g), None) => g.clone(),
            (Some(g), Some(existing)) => g.add(&existing).map_err(|e| {
                AutodiffError::GradientError(format!("Failed to accumulate gradient: {}", e))
            })?,
            (None, None) => {
                // If this is the output node, create a ones tensor with the same shape
                    Tensor::ones_like(&self.tensor).map_err(|e| {
                        AutodiffError::GradientError(format!("Failed to create ones tensor: {}", e))
                    })?
            }
            (None, Some(_)) => return Ok(()), // Already computed
        };
        
        // Store the gradient
        self.gradient = Some(grad.clone());
        
        // If this is a leaf node, we're done
        let op = match &self.op {
            Some(op) => op,
            None => return Ok(()),
        };
        
        // Compute gradients for inputs
        let input_tensors: Vec<_> = self.inputs.iter().map(|node| &node.tensor).collect();
        let input_grads = op.backward(&grad, &input_tensors, &self.tensor);
        
        // Propagate gradients to input nodes
        for (i, (node, grad)) in self.inputs.iter().zip(input_grads.into_iter()).enumerate() {
            node.backward(Some(&grad)).map_err(|e| {
                AutodiffError::GradientError(format!("Failed to propagate gradient to input {}: {}", i, e))
            })?;
        }
        
        Ok(())
    }
}

/// A computation graph for automatic differentiation.
#[derive(Debug, Default)]
pub struct ComputationGraph {
    nodes: Vec<Arc<Node>>,
}

impl ComputationGraph {
    /// Creates a new empty computation graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }
    
    /// Adds a new tensor to the graph.
    pub fn add_tensor(&mut self, tensor: Tensor) -> Arc<Node> {
        let node = Node::new_leaf(tensor);
        self.nodes.push(node.clone());
        node
    }
    
    /// Adds a new operation to the graph.
    pub fn add_op(
        &mut self,
        op: Arc<dyn DifferentiableOp>,
        inputs: &[Arc<Node>],
    ) -> Result<Arc<Node>, AutodiffError> {
        let input_tensors: Vec<_> = inputs.iter().map(|node| &node.tensor).collect();
        let output = op.forward(&input_tensors)?;
        
        let node = Node::new_op(output, op, inputs.to_vec());
        self.nodes.push(node.clone());
        
        Ok(node)
    }
    
    /// Computes the gradients of the loss with respect to all tensors in the graph.
    pub fn backward(&self, output: &Node) -> Result<(), AutodiffError> {
        output.backward(None)
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
