//! Computation graph for automatic differentiation.

use super::*;
use std::sync::{Arc, RwLock};

/// A node in the computation graph.
#[derive(Debug)]
pub struct Node {
    /// The tensor value at this node.
    pub tensor: BaseTensor,
    
    /// The operation that produced this tensor (None for leaf nodes).
    pub op: Option<Arc<dyn DifferentiableOp>>,
    
    /// The input nodes to this operation.
    pub inputs: Vec<Arc<Node>>,
    
    /// The gradient of the loss with respect to this tensor.
    pub gradient: Option<BaseTensor>,
    
    /// Whether this node requires gradient computation.
    requires_grad: bool,
    
    /// Whether this node has been visited during backward pass.
    visited: RwLock<bool>,
}

impl Node {
    /// Creates a new leaf node (input tensor).
    pub fn new_leaf(tensor: BaseTensor, requires_grad: bool) -> Arc<Self> {
        Arc::new(Self {
            tensor,
            op: None,
            inputs: Vec::new(),
            gradient: None,
            requires_grad,
            visited: RwLock::new(false),
        })
    }
    
    /// Creates a new operation node.
    pub fn new_op(
        tensor: BaseTensor,
        op: Arc<dyn DifferentiableOp>,
        inputs: Vec<Arc<Node>>,
        requires_grad: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            tensor,
            op: Some(op),
            inputs,
            gradient: None,
            requires_grad,
            visited: RwLock::new(false),
        })
    }
    
    /// Returns whether this node requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    /// Marks this node as visited during the backward pass.
    pub fn mark_visited(&self) -> bool {
        let mut visited = self.visited.write().unwrap();
        if *visited {
            false
        } else {
            *visited = true;
            true
        }
    }
    
    /// Resets the visited flag for this node.
    pub fn reset_visited(&self) {
        *self.visited.write().unwrap() = false;
    }
    
    /// Computes the gradient of the loss with respect to this node.
    pub fn backward(&self, grad: Option<&BaseTensor>) -> Result<(), AutodiffError> {
        // Skip if gradient is not needed
        if !self.requires_grad {
            return Ok(());
        }
        
        // Initialize gradient if not already set
        let grad = match (grad, &self.gradient) {
            (Some(g), None) => g.clone(),
            (Some(g), Some(existing)) => {
                g.add(existing).map_err(|e| {
                    AutodiffError::GradientError(format!("Failed to accumulate gradient: {}", e))
                })?
            }
            (None, None) => {
                // If this is the output node, create a ones tensor with the same shape
                BaseTensor::ones_like(&self.tensor).map_err(|e| {
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
        
        // Mark as visited to prevent cycles
        if !self.mark_visited() {
            return Ok(());
        }
        
        // Compute gradients for inputs
        let input_tensors: Vec<_> = self.inputs.iter().map(|node| &node.tensor).collect();
        let input_grads = op.backward(&grad, &input_tensors, &self.tensor);
        
        // Propagate gradients to input nodes
        for (i, (node, grad)) in self.inputs.iter().zip(input_grads.into_iter()).enumerate() {
            if node.requires_grad() {
                node.backward(Some(&grad)).map_err(|e| {
                    AutodiffError::GradientError(format!(
                        "Failed to propagate gradient to input {}: {}",
                        i, e
                    ))
                })?;
            }
        }
        
        Ok(())
    }
    
    /// Resets the gradients of this node and all its dependencies.
    pub fn zero_grad(&self) {
        if self.requires_grad {
            self.gradient = None;
            *self.visited.write().unwrap() = false;
            
            for input in &self.inputs {
                input.zero_grad();
            }
        }
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
    pub fn add_tensor(&mut self, tensor: BaseTensor, requires_grad: bool) -> Arc<Node> {
        let node = Node::new_leaf(tensor, requires_grad);
        self.nodes.push(node.clone());
        node
    }
    
    /// Adds a new operation to the graph.
    pub fn add_op(
        &mut self,
        op: Arc<dyn DifferentiableOp>,
        inputs: &[Arc<Node>],
        requires_grad: bool,
    ) -> Result<Arc<Node>, AutodiffError> {
        let input_tensors: Vec<_> = inputs.iter().map(|node| &node.tensor).collect();
        let output = op.forward(&input_tensors)?;
        
        let node = Node::new_op(output, op, inputs.to_vec(), requires_grad);
        self.nodes.push(node.clone());
        
        Ok(node)
    }
    
    /// Computes the gradients of the loss with respect to all tensors in the graph.
    pub fn backward(&self, output: &Node) -> Result<(), AutodiffError> {
        // Reset visited flags
        self.reset_visited();
        
        // Perform backward pass
        output.backward(None)
    }
    
    /// Resets the gradients of all tensors in the graph.
    pub fn zero_grad(&self) {
        for node in &self.nodes {
            node.zero_grad();
        }
    }
    
    /// Resets the visited flags of all nodes in the graph.
    fn reset_visited(&self) {
        for node in &self.nodes {
            node.reset_visited();
        }
    }
    
    /// Returns a reference to all nodes in the graph.
    pub fn nodes(&self) -> &[Arc<Node>] {
        &self.nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor as BaseTensor;
    
    #[test]
    fn test_computation_graph() -> Result<(), AutodiffError> {
        // Create a simple computation graph: z = x * y + 2
        let mut graph = ComputationGraph::new();
        
        // Input tensors
        let x = graph.add_tensor(BaseTensor::scalar(2.0), true);
        let y = graph.add_tensor(BaseTensor::scalar(3.0), true);
        
        // Operations
        let mul = graph.add_op(
            Arc::new(super::super::ops::MultiplyOp),
            &[x.clone(), y.clone()],
            true,
        )?;
        
        let two = graph.add_tensor(BaseTensor::scalar(2.0), false);
        let z = graph.add_op(
            Arc::new(super::super::ops::AddOp),
            &[mul, two],
            true,
        )?;
        
        // Forward pass
        assert_eq!(z.tensor.to_scalar::<f32>(), Some(8.0));
        
        // Backward pass
        graph.backward(&z)?;
        
        // Check gradients
        assert_eq!(x.gradient.as_ref().unwrap().to_scalar::<f32>(), Some(3.0)); // dz/dx = y = 3
        assert_eq!(y.gradient.as_ref().unwrap().to_scalar::<f32>(), Some(2.0)); // dz/dy = x = 2
        
        // Test zero_grad
        graph.zero_grad();
        assert!(x.gradient.is_none());
        assert!(y.gradient.is_none());
        
        Ok(())
    }
    
    #[test]
    fn test_computation_graph_with_reuse() -> Result<(), AutodiffError> {
        // Test reusing the same input in multiple operations
        // z = (x * y) + (x * 2)
        let mut graph = ComputationGraph::new();
        
        // Input tensors
        let x = graph.add_tensor(BaseTensor::scalar(2.0), true);
        let y = graph.add_tensor(BaseTensor::scalar(3.0), true);
        
        // First operation: x * y
        let mul1 = graph.add_op(
            Arc::new(super::super::ops::MultiplyOp),
            &[x.clone(), y.clone()],
            true,
        )?;
        
        // Second operation: x * 2
        let two = graph.add_tensor(BaseTensor::scalar(2.0), false);
        let mul2 = graph.add_op(
            Arc::new(super::super::ops::MultiplyOp),
            &[x.clone(), two],
            true,
        )?;
        
        // Final operation: add the results
        let z = graph.add_op(
            Arc::new(super::super::ops::AddOp),
            &[mul1, mul2],
            true,
        )?;
        
        // Forward pass
        assert_eq!(z.tensor.to_scalar::<f32>(), Some(10.0)); // (2*3) + (2*2) = 10
        
        // Backward pass
        graph.backward(&z)?;
        
        // Check gradients
        // dz/dx = y + 2 = 3 + 2 = 5
        // dz/dy = x = 2
        assert_eq!(x.gradient.as_ref().unwrap().to_scalar::<f32>(), Some(5.0));
        assert_eq!(y.gradient.as_ref().unwrap().to_scalar::<f32>(), Some(2.0));
        
        Ok(())
    }
}
