//! 2D Convolutional layer implementation.

use crate::{
    autodiff::{Node},
    tensor::Tensor,
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
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        dilation: Option<(usize, usize)>,
        groups: Option<usize>,
        bias: Option<bool>,
        weight_init: impl Fn(&[usize]) -> Tensor,
        bias_init: impl Fn(&[usize]) -> Tensor,
    ) -> Self {
        let stride = stride.unwrap_or((1, 1));
        let padding = padding.unwrap_or((0, 0));
        let dilation = dilation.unwrap_or((1, 1));
        let groups = groups.unwrap_or(1);
        let use_bias = bias.unwrap_or(true);
        
        // Initialize weights
        let weight_shape = [out_channels, in_channels / groups, kernel_size.0, kernel_size.1];
        let weights_data = weight_init(&weight_shape);
        let weights = Arc::new(Node::new_leaf(weights_data));
        
        // Initialize bias if needed
        let bias_node = if use_bias {
            let bias_data = bias_init(&[out_channels]);
            Some(Arc::new(Node::new_leaf(bias_data)))
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
    /// * `input` - Input tensor of shape [batch_size, in_channels, height, width]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, out_channels, out_height, out_width]
    pub fn forward(
        &self,
        input: Arc<Node>,
    ) -> Result<Arc<Node>, Box<dyn std::error::Error>> {
        // Add convolution operation
        let output = input.conv2d(
            self.weights.clone(),
            self.bias.clone(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        
        Ok(output)
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


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_conv2d_forward() -> Result<(), Box<dyn std::error::Error>> {
        // Create a simple 2D convolution layer
        let in_channels = 1;
        let out_channels = 1;
        let kernel_size = (3, 3);
        
        // Initialize weights with a simple edge detection kernel
        let weight_data = Tensor::from_slice(
            &[
                0.0, 1.0, 0.0,
                1.0, -4.0, 1.0,
                0.0, 1.0, 0.0,
            ],
            vec![out_channels, in_channels, kernel_size.0, kernel_size.1],
        )?;
        
        let bias_data = Tensor::zeros(&[out_channels])?;
        
        let layer = Conv2dLayer::new(
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
        let input_data = Tensor::from_slice(
            &[
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
            ],
            vec![1, 1, 5, 5],  // [batch_size, in_channels, height, width]
        )?;
        
        let input = Arc::new(Node::new_leaf(input_data));
        
        // Forward pass
        let output = layer.forward(input)?;
        
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
        // Create a simple 2D convolution layer
        let in_channels = 1;
        let out_channels = 2;
        let kernel_size = (3, 3);
        
        let layer = Conv2dLayer::new(
            in_channels,
            out_channels,
            kernel_size,
            Some((1, 1)),  // stride
            Some((1, 1)),  // padding
            Some((1, 1)),  // dilation
            Some(1),       // groups
            Some(true),    // bias
            |shape| Tensor::randn(shape, 0.0, 0.1),
            |shape| Tensor::zeros(shape).unwrap(),
        );
        
        // Create input tensor
        let input_data = Tensor::randn(&[2, in_channels, 5, 5], 0.0, 1.0);
        let input = Arc::new(Node::new_leaf(input_data));
        
        // Forward pass
        let output = layer.forward(input.clone())?;
        
        // Create a dummy loss (sum of outputs)
        let loss = output.sum();
        
        // Backward pass
        loss.backward();
        
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
