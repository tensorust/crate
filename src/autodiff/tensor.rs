//! Differentiable tensor type for automatic differentiation.

use super::*;
use crate::tensor::Tensor as BaseTensor;
use std::sync::Arc;

/// A differentiable tensor that tracks operations for automatic differentiation.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// The underlying tensor data.
    data: BaseTensor,
    
    /// The node in the computation graph.
    node: Option<Arc<Node>>,
    
    /// Whether to track operations on this tensor.
    requires_grad: bool,
}

impl Tensor {
    /// Creates a new tensor that requires gradient computation.
    pub fn new(data: BaseTensor, requires_grad: bool) -> Self {
        Self {
            data,
            node: None,
            requires_grad,
        }
    }
    
    /// Creates a new tensor from a scalar value.
    pub fn scalar<T: Into<f64>>(value: T) -> Self {
        Self {
            data: BaseTensor::scalar(value.into() as f32),
            node: None,
            requires_grad: false,
        }
    }
    
    /// Creates a new tensor filled with ones.
    pub fn ones(shape: &[usize], requires_grad: bool) -> Result<Self, AutodiffError> {
        Ok(Self {
            data: BaseTensor::ones(shape).map_err(|e| AutodiffError::OperationFailed(e.to_string()))?,
            node: None,
            requires_grad,
        })
    }
    
    /// Creates a new tensor filled with zeros.
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Result<Self, AutodiffError> {
        Ok(Self {
            data: BaseTensor::zeros(shape).map_err(|e| AutodiffError::OperationFailed(e.to_string()))?,
            node: None,
            requires_grad,
        })
    }
    
    /// Creates a tensor like the given one with the same shape and device.
    pub fn ones_like(other: &Self) -> Result<Self, AutodiffError> {
        Ok(Self {
            data: other.data.ones_like().map_err(|e| AutodiffError::OperationFailed(e.to_string()))?,
            node: None,
            requires_grad: other.requires_grad,
        })
    }
    
    /// Returns a reference to the underlying tensor data.
    pub fn data(&self) -> &BaseTensor {
        &self.data
    }
    
    /// Returns a mutable reference to the underlying tensor data.
    /// 
    /// # Safety
    /// Modifying the tensor data directly can break the computation graph.
    /// Only use this if you know what you're doing.
    pub unsafe fn data_mut(&mut self) -> &mut BaseTensor {
        &mut self.data
    }
    
    /// Returns whether this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    /// Sets whether this tensor requires gradient computation.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }
    
    /// Returns the gradient of this tensor, if it has been computed.
    pub fn grad(&self) -> Option<&BaseTensor> {
        self.node.as_ref().and_then(|n| n.gradient.as_ref())
    }
    
    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
    
    /// Converts the tensor to a scalar value.
    /// 
    /// # Panics
    /// Panics if the tensor is not a scalar.
    pub fn to_scalar<T: From<f32>>(&self) -> Option<T> {
        self.data.to_scalar()
    }
    
    /// Performs element-wise addition with another tensor.
    pub fn add(&self, other: &Self) -> Result<Self, AutodiffError> {
        let requires_grad = self.requires_grad || other.requires_grad;
        
        // Forward pass
        let output_data = self.data.add(&other.data)
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))?;
        
        // Create output tensor
        let mut output = Self {
            data: output_data,
            node: None,
            requires_grad,
        };
        
        // Build computation graph if needed
        if requires_grad {
            let op = Arc::new(AddOp);
            let inputs = if self.requires_grad && other.requires_grad {
                vec![self.clone(), other.clone()]
            } else if self.requires_grad {
                vec![self.clone()]
            } else {
                vec![other.clone()]
            };
            
            output.node = Some(Arc::new(Node {
                tensor: output.clone(),
                op: Some(op),
                inputs: inputs.into_iter().filter_map(|t| t.node).collect(),
                gradient: None,
            }));
        }
        
        Ok(output)
    }
    
    /// Performs element-wise multiplication with another tensor.
    pub fn mul(&self, other: &Self) -> Result<Self, AutodiffError> {
        let requires_grad = self.requires_grad || other.requires_grad;
        
        // Forward pass
        let output_data = self.data.mul(&other.data)
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))?;
        
        // Create output tensor
        let mut output = Self {
            data: output_data,
            node: None,
            requires_grad,
        };
        
        // Build computation graph if needed
        if requires_grad {
            let op = Arc::new(MultiplyOp);
            let inputs = if self.requires_grad && other.requires_grad {
                vec![self.clone(), other.clone()]
            } else if self.requires_grad {
                vec![self.clone()]
            } else {
                vec![other.clone()]
            };
            
            output.node = Some(Arc::new(Node {
                tensor: output.clone(),
                op: Some(op),
                inputs: inputs.into_iter().filter_map(|t| t.node).collect(),
                gradient: None,
            }));
        }
        
        Ok(output)
    }
    
    /// Computes the gradient of this tensor with respect to all input tensors.
    pub fn backward(&self) -> Result<(), AutodiffError> {
        if !self.requires_grad {
            return Err(AutodiffError::GradientError(
                "Cannot compute gradient for a tensor that doesn't require gradient".to_string(),
            ));
        }
        
        // Initialize gradient to ones if this is the output node
        if let Some(node) = &self.node {
            if node.gradient.is_none() {
                let ones = Self::ones_like(self)?;
                node.gradient = Some(ones.data);
            }
            
            // Perform backward pass
            node.backward(None)?;
        }
        
        Ok(())
    }
}

impl std::ops::Add for Tensor {
    type Output = Result<Self, AutodiffError>;
    
    fn add(self, other: Self) -> Self::Output {
        self.add(&other)
    }
}

impl std::ops::Add<&Tensor> for Tensor {
    type Output = Result<Self, AutodiffError>;
    
    fn add(self, other: &Self) -> Self::Output {
        self.add(other)
    }
}

impl std::ops::Mul for Tensor {
    type Output = Result<Self, AutodiffError>;
    
    fn mul(self, other: Self) -> Self::Output {
        self.mul(&other)
    }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Result<Self, AutodiffError>;
    
    fn mul(self, other: &Self) -> Self::Output {
        self.mul(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_creation() {
        let x = Tensor::scalar(2.0);
        assert_eq!(x.to_scalar::<f32>(), Some(2.0));
        assert!(!x.requires_grad());
        
        let y = Tensor::new(BaseTensor::scalar(3.0), true);
        assert_eq!(y.to_scalar::<f32>(), Some(3.0));
        assert!(y.requires_grad());
    }
    
    #[test]
    fn test_tensor_operations() {
        let x = Tensor::new(BaseTensor::scalar(2.0), true);
        let y = Tensor::new(BaseTensor::scalar(3.0), true);
        
        // Forward pass
        let z = (x.mul(&y).unwrap() + Tensor::scalar(1.0)).unwrap();
        assert_eq!(z.to_scalar::<f32>(), Some(7.0));
        
        // Backward pass
        z.backward().unwrap();
        
        // Check gradients
        if let (Some(node_x), Some(node_y)) = (x.node, y.node) {
            assert_eq!(node_x.gradient.as_ref().unwrap().to_scalar::<f32>(), Some(3.0)); // dz/dx = y = 3
            assert_eq!(node_y.gradient.as_ref().unwrap().to_scalar::<f32>(), Some(2.0)); // dz/dy = x = 2
        } else {
            panic!("Nodes not properly set up for gradient computation");
        }
    }
}
