//! 2D Convolutional layer implementation.

use crate::{
    autodiff::{tensor::Tensor, ComputationGraph, DifferentiableOp, Node},
    tensor::Tensor as BaseTensor,
};
use std::sync::Arc;

/// A 2D convolutional layer.
///
/// This layer applies a 2D convolution over an input signal composed of several input
/// planes. The input is expected to have shape [batch_size, in_channels, height, width].
#[derive(Debug)]
pub struct Conv2dLayer {
    /// Convolutional kernel weights of shape [out_channels, in_channels, kernel_height, kernel_width]
    weights: Arc<Node>,
    
    /// Bias terms of shape [out_channels]
    bias: Option<Arc<Node>>,
    
    /// Stride of the convolution
    stride: (usize, usize),
    
    /// Padding to add to all four sides of the input
    padding: (usize, usize),
    
    /// Spacing between kernel elements
    dilation: (usize, usize),
    
    /// Number of groups in which the input is split along the channel axis
    groups: usize,
    
    /// Input channels
    in_channels: usize,
    
    /// Output channels
    out_channels: usize,
    
    /// Kernel size (height, width)
    kernel_size: (usize, usize),
}

impl Conv2dLayer {
    /// Creates a new 2D convolutional layer.
    ///
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolving kernel (height, width)
    /// * `stride` - Stride of the convolution (default: 1)
    /// * `padding` - Padding added to all four sides of the input (default: 0)
    /// * `dilation` - Spacing between kernel elements (default: 1)
    /// * `groups` - Number of blocked connections from input to output (default: 1)
    /// * `bias` - Whether to add a learnable bias (default: true)
    /// * `weight_init` - Function to initialize weights
    /// * `bias_init` - Function to initialize bias
    pub fn new(
        graph: &mut ComputationGraph,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: Option<bool>,
        weight_init: impl Fn(&[usize]) -> BaseTensor,
        bias_init: impl Fn(&[usize]) -> BaseTensor,
    ) -> Self {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);
        
        // Initialize weights
        let weight_shape = [out_channels, in_channels / groups, kernel_size.0, kernel_size.1];
        let weights_data = weight_init(&weight_shape);
        let weights = graph.add_tensor(weights_data, true);
        
        // Initialize bias if needed
        let bias_node = if use_bias {
            let bias_data = bias_init(&[out_channels]);
            Some(graph.add_tensor(bias_data, true))
        } else {
            None
        };
        
        Self {
            weights,
            bias: bias_node,
            stride,
            padding,
            dilation,
            groups,
            in_channels,
            out_channels,
            kernel_size,
        }
    }
    
    /// Applies the 2D convolution to the input.
    ///
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `input` - Input tensor of shape [batch_size, in_channels, height, width]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, out_channels, out_height, out_width]
    pub fn forward(
        &self,
        graph: &mut ComputationGraph,
        input: Arc<Node>,
    ) -> Result<Arc<Node>, Box<dyn std::error::Error>> {
        // Add convolution operation
        let output = graph.add_op(
            Arc::new(Conv2dOp {
                stride: self.stride,
                padding: self.padding,
                dilation: self.dilation,
                groups: self.groups,
            }),
            &[input, self.weights.clone()],
            true,
        )?;
        
        // Add bias if needed
        if let Some(bias) = &self.bias {
            // Reshape bias to [1, out_channels, 1, 1] for broadcasting
            let bias_reshaped = graph.add_op(
                Arc::new(ReshapeOp::new(&[1, self.out_channels, 1, 1])),
                &[bias.clone()],
                true,
            )?;
            
            graph.add_op(
                Arc::new(AddOp),
                &[output, bias_reshaped],
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
    
    /// Returns the bias tensor if it exists.
    pub fn bias(&self) -> Option<&Arc<Node>> {
        self.bias.as_ref()
    }
    
    /// Returns the stride of the convolution.
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }
    
    /// Returns the padding of the convolution.
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }
    
    /// Returns the dilation of the convolution.
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }
    
    /// Returns the number of groups.
    pub fn groups(&self) -> usize {
        self.groups
    }
    
    /// Returns the number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
    
    /// Returns the number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
    
    /// Returns the kernel size.
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }
}

/// 2D convolution operation.
#[derive(Debug)]
struct Conv2dOp {
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
}

impl DifferentiableOp for Conv2dOp {
    fn forward(&self, inputs: &[&BaseTensor]) -> Result<BaseTensor, crate::autodiff::AutodiffError> {
        if inputs.len() != 2 {
            return Err(crate::autodiff::AutodiffError::InvalidInput(
                "Conv2d operation requires exactly 2 inputs".to_string(),
            ));
        }
        
        let input = inputs[0];
        let weight = inputs[1];
        
        // Get input dimensions
        let batch_size = input.shape()[0];
        let in_channels = input.shape()[1];
        let in_height = input.shape()[2];
        let in_width = input.shape()[3];
        
        // Get weight dimensions
        let out_channels = weight.shape()[0];
        let kernel_height = weight.shape()[2];
        let kernel_width = weight.shape()[3];
        
        // Calculate output dimensions
        let out_height = (in_height + 2 * self.padding.0 - self.dilation.0 * (kernel_height - 1) - 1) / self.stride.0 + 1;
        let out_width = (in_width + 2 * self.padding.1 - self.dilation.1 * (kernel_width - 1) - 1) / self.stride.1 + 1;
        
        // Create output tensor
        let mut output = BaseTensor::zeros(&[batch_size, out_channels, out_height, out_width])?;
        
        // TODO: Implement efficient 2D convolution
        // This is a naive implementation for demonstration
        // In practice, you'd want to use a BLAS library or specialized convolution implementation
        
        // Apply padding if needed
        let padded_input = if self.padding.0 > 0 || self.padding.1 > 0 {
            let mut padded = BaseTensor::zeros(&[
                batch_size,
                in_channels,
                in_height + 2 * self.padding.0,
                in_width + 2 * self.padding.1,
            ])?;
            
            // Copy input to padded tensor
            // This is a simplified version - in practice, you'd want to handle this more efficiently
            for b in 0..batch_size {
                for c in 0..in_channels {
                    for h in 0..in_height {
                        for w in 0..in_width {
                            let val = input.get(&[b, c, h, w])?;
                            padded.set(&[b, c, h + self.padding.0, w + self.padding.1], val)?;
                        }
                    }
                }
            }
            
            padded
        } else {
            input.clone()
        };
        
        // Perform convolution
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0;
                        
                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let h = oh * self.stride.0 + kh * self.dilation.0;
                                    let w = ow * self.stride.1 + kw * self.dilation.1;
                                    
                                    if h < padded_input.shape()[2] && w < padded_input.shape()[3] {
                                        let input_val = padded_input.get(&[b, ic, h, w])?;
                                        let weight_val = weight.get(&[oc, ic, kh, kw])?;
                                        sum += input_val * weight_val;
                                    }
                                }
                            }
                        }
                        
                        output.set(&[b, oc, oh, ow], sum)?;
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    fn backward(&self, grad: &BaseTensor, inputs: &[&BaseTensor], _output: &BaseTensor) -> Vec<BaseTensor> {
        // This is a placeholder implementation
        // In practice, you'd need to implement proper convolution gradients
        vec![
            BaseTensor::zeros_like(inputs[0]).expect("Failed to create zeros tensor"),
            BaseTensor::zeros_like(inputs[1]).expect("Failed to create zeros tensor"),
        ]
    }
    
    fn input_names(&self) -> Vec<&'static str> {
        vec!["input", "weight"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_conv2d_forward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a simple 2D convolution layer
        let in_channels = 1;
        let out_channels = 1;
        let kernel_size = (3, 3);
        
        // Initialize weights with a simple edge detection kernel
        let weight_data = BaseTensor::from_slice(
            &[
                0.0, 1.0, 0.0,
                1.0, -4.0, 1.0,
                0.0, 1.0, 0.0,
            ],
            &[out_channels, in_channels, kernel_size.0, kernel_size.1],
        )?;
        
        let bias_data = BaseTensor::zeros(&[out_channels])?;
        
        let layer = Conv2dLayer::new(
            &mut graph,
            in_channels,
            out_channels,
            kernel_size,
            Some((1, 1)),  // stride
            Some((1, 1)),  // padding
            Some((1, 1)),  // dilation
            Some(1),       // groups
            Some(true),    // bias
            |_| weight_data.clone(),
            |_| bias_data.clone(),
        );
        
        // Create input tensor (simple 5x5 image with a vertical edge)
        let input_data = BaseTensor::from_slice(
            &[
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
            ],
            &[1, 1, 5, 5],  // [batch_size, in_channels, height, width]
        )?;
        
        let input = graph.add_tensor(input_data, false);
        
        // Forward pass
        let output = layer.forward(&mut graph, input)?;
        
        // Check output shape
        let output_shape = output.tensor.shape();
        assert_eq!(output_shape, &[1, 1, 5, 5]);
        
        // Check some output values
        let output_data = output.tensor.to_vec::<f32>()?;
        
        // The edge detection kernel should highlight the vertical edge
        assert_relative_eq!(output_data[12], 2.0, epsilon = 1e-5);  // Center of the edge
        
        Ok(())
    }
    
    #[test]
    fn test_conv2d_backward() -> Result<(), Box<dyn std::error::Error>> {
        // This test verifies that gradients can be computed
        let mut graph = ComputationGraph::new();
        
        // Create a simple 2D convolution layer
        let in_channels = 1;
        let out_channels = 2;
        let kernel_size = (3, 3);
        
        let layer = Conv2dLayer::new(
            &mut graph,
            in_channels,
            out_channels,
            kernel_size,
            Some((1, 1)),  // stride
            Some((1, 1)),  // padding
            Some((1, 1)),  // dilation
            Some(1),       // groups
            Some(true),    // bias
            |shape| BaseTensor::randn(shape, 0.0, 0.1),
            |shape| BaseTensor::zeros(shape).unwrap(),
        );
        
        // Create input tensor
        let input_data = BaseTensor::randn(&[2, in_channels, 5, 5], 0.0, 1.0);
        let input = graph.add_tensor(input_data, true);
        
        // Forward pass
        let output = layer.forward(&mut graph, input.clone())?;
        
        // Create a dummy loss (sum of outputs)
        let ones = graph.add_tensor(
            BaseTensor::ones(output.tensor.shape())?,
            false
        );
        
        let loss = graph.add_op(
            Arc::new(MatMulOp),
            &[output, ones],
            true,
        )?;
        
        // Backward pass
        graph.backward(&loss)?;
        
        // Check gradients
        assert!(layer.weights().gradient.is_some(), "Weights gradient should be computed");
        if let Some(bias) = layer.bias() {
            assert!(bias.gradient.is_some(), "Bias gradient should be computed");
        }
        
        // The input gradient should also be computed since we set requires_grad=true
        assert!(input.gradient.is_some(), "Input gradient should be computed");
        
        Ok(())
    }
}
