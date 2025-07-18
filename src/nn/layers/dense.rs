#![cfg(feature = "autodiff")]
//! Fully connected (Dense) layer implementation.

use crate::{
    autodiff::{
        ops::{AddOp, MatMulOp},
        tensor::Tensor,
        ComputationGraph, DifferentiableOp, Node,
    },
    tensor::Tensor as BaseTensor,
};
use std::sync::Arc;

/// A fully connected (dense) layer.
///
/// This layer implements the operation:
/// `output = activation(input @ weights + bias)`
///
/// Where `@` represents matrix multiplication.
#[derive(Debug)]
pub struct DenseLayer {
    /// Weight matrix of shape [input_dim, output_dim]
    weights: Arc<Node>,
    
    /// Bias vector of shape [output_dim]
    bias: Arc<Node>,
    
    /// Activation function (e.g., ReLU, Sigmoid, etc.)
    activation: Option<Arc<dyn DifferentiableOp>>,
    
    /// Input dimension
    input_dim: usize,
    
    /// Output dimension
    output_dim: usize,
    
    /// Whether to use bias
    use_bias: bool,
}

impl DenseLayer {
    /// Creates a new dense layer with the given parameters.
    ///
    /// # Arguments
    /// * `graph` - The computation graph to add the layer to
    /// * `input_dim` - Dimensionality of the input
    /// * `output_dim` - Dimensionality of the output
    /// * `activation` - Activation function to use (None for linear)
    /// * `use_bias` - Whether to use a bias term
    /// * `weight_init` - Initializer for the weight matrix
    /// * `bias_init` - Initializer for the bias vector
    pub fn new(
        graph: &mut ComputationGraph,
        input_dim: usize,
        output_dim: usize,
        activation: Option<Arc<dyn DifferentiableOp>>,
        use_bias: bool,
        weight_init: impl Fn(&[usize]) -> BaseTensor,
        bias_init: impl Fn(&[usize]) -> BaseTensor,
    ) -> Self {
        // Initialize weights
        let weights_data = weight_init(&[input_dim, output_dim]);
        let weights = graph.add_tensor(weights_data, true);
        
        // Initialize bias if needed
        let bias = if use_bias {
            let bias_data = bias_init(&[output_dim]);
            graph.add_tensor(bias_data, true)
        } else {
            // Add a zero bias that won't be used
            graph.add_tensor(BaseTensor::zeros(&[output_dim]).unwrap(), false)
        };
        
        Self {
            weights,
            bias,
            activation,
            input_dim,
            output_dim,
            use_bias,
        }
    }
    
    /// Applies the layer to the input tensor.
    ///
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `input` - Input tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, output_dim]
    pub fn forward(
        &self,
        graph: &mut ComputationGraph,
        input: Arc<Node>,
    ) -> Result<Arc<Node>, Box<dyn std::error::Error>> {
        // Matrix multiplication: input @ weights
        let output = graph.add_op(
            Arc::new(MatMulOp),
            &[input, self.weights.clone()],
            true,
        )?;
        
        // Add bias if needed
        let output = if self.use_bias {
            graph.add_op(
                Arc::new(AddOp),
                &[output, self.bias.clone()],
                true,
            )?
        } else {
            output
        };
        
        // Apply activation if specified
        if let Some(activation) = &self.activation {
            graph.add_op(
                activation.clone(),
                &[output],
                true,
            )
        } else {
            Ok(output)
        }
    }
    
    /// Returns the weight tensor.
    pub fn weights(&self) -> &Arc<Node> {
        &self.weights
    }
    
    /// Returns the bias tensor.
    pub fn bias(&self) -> &Arc<Node> {
        &self.bias
    }
    
    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    
    /// Returns the output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
    
    /// Returns whether the layer uses a bias term.
    pub fn use_bias(&self) -> bool {
        self.use_bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::ops::{ReLUOp, AddOp, MatMulOp};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_dense_layer_forward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a dense layer with known weights
        let input_dim = 3;
        let output_dim = 2;
        
        // Initialize weights and bias with known values
        let weights_data = BaseTensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[input_dim, output_dim],
        )?;
        
        let bias_data = BaseTensor::from_slice(&[0.1, 0.2], &[output_dim])?;
        
        let layer = DenseLayer::new(
            &mut graph,
            input_dim,
            output_dim,
            Some(Arc::new(ReLUOp)),
            true,
            |_| weights_data.clone(),
            |_| bias_data.clone(),
        );
        
        // Create input tensor
        let input_data = BaseTensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, input_dim],  // batch_size=2, input_dim=3
        )?;
        
        let input = graph.add_tensor(input_data, false);
        
        // Forward pass
        let output = layer.forward(&mut graph, input)?;
        
        // Expected output:
        // For first sample: [1, 2, 3] @ [[1, 4], [2, 5], [3, 6]] + [0.1, 0.2] = [14.1, 32.2]
        // After ReLU: [14.1, 32.2] (no change)
        // For second sample: [4, 5, 6] @ [[1, 4], [2, 5], [3, 6]] + [0.1, 0.2] = [32.1, 77.2]
        // After ReLU: [32.1, 77.2] (no change)
        let output_data = output.tensor.to_vec::<f32>()?;
        
        assert_eq!(output_data.len(), 4);  // 2 samples * 2 output dims
        assert_relative_eq!(output_data[0], 14.1, epsilon = 1e-5);
        assert_relative_eq!(output_data[1], 32.2, epsilon = 1e-5);
        assert_relative_eq!(output_data[2], 32.1, epsilon = 1e-5);
        assert_relative_eq!(output_data[3], 77.2, epsilon = 1e-5);
        
        Ok(())
    }
    
    #[test]
    fn test_dense_layer_backward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a dense layer with random initialization
        let input_dim = 3;
        let output_dim = 2;
        
        let layer = DenseLayer::new(
            &mut graph,
            input_dim,
            output_dim,
            Some(Arc::new(ReLUOp)),
            true,
            |shape| BaseTensor::randn(shape, 0.0, 0.01),
            |shape| BaseTensor::zeros(shape).unwrap(),
        );
        
        // Create input tensor
        let batch_size = 4;
        let input_data = BaseTensor::randn(&[batch_size, input_dim], 0.0, 1.0);
        let input = graph.add_tensor(input_data, true);
        
        // Forward pass
        let output = layer.forward(&mut graph, input.clone())?;
        
        // Create a dummy loss (sum of outputs)
        let ones = graph.add_tensor(BaseTensor::ones(&[batch_size, output_dim])?, false);
        let loss = graph.add_op(
            Arc::new(MatMulOp),
            &[output, ones],
            true,
        )?;
        
        // Backward pass
        graph.backward(&loss)?;
        
        // Check gradients
        assert!(layer.weights().gradient.is_some(), "Weights gradient should be computed");
        assert!(layer.bias().gradient.is_some(), "Bias gradient should be computed");
        
        // The input gradient should also be computed since we set requires_grad=true
        assert!(input.gradient.is_some(), "Input gradient should be computed");
        
        Ok(())
    }
}
