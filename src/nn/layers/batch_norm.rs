//! Batch Normalization layer implementation.
//!
//! This module implements Batch Normalization as described in the paper:
//! "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
//! by Sergey Ioffe and Christian Szegedy (2015).

use crate::{
    autodiff::{Node},
    tensor::Tensor,
};
use std::sync::Arc;

/// Batch Normalization layer.
///
/// This layer normalizes the activations of the previous layer at each batch,
/// i.e., applies a transformation that maintains the mean activation close to 0
/// and the activation standard deviation close to 1.
#[derive(Debug)]
pub struct BatchNorm2d {
    /// Scale parameter (gamma) [num_features]
    gamma: Arc<Node>,
    
    /// Shift parameter (beta) [num_features]
    beta: Arc<Node>,
    
    /// Running mean of the features [num_features]
    running_mean: Tensor,
    
    /// Running variance of the features [num_features]
    running_var: Tensor,
    
    /// Number of features (channels for 2D input)
    num_features: usize,
    
    /// Momentum for the moving average
    momentum: f32,
    
    /// Epsilon for numerical stability
    eps: f32,
    
    /// Whether the model is in training mode
    training: bool,
}

impl BatchNorm2d {
    /// Creates a new BatchNorm2d layer.
    ///
    /// # Arguments
    /// * `graph` - The computation graph
    /// * `num_features` - Number of features/channels
    /// * `momentum` - The value used for the running_mean and running_var computation
    /// * `eps` - A value added to the denominator for numerical stability
    /// * `affine` - Whether to learn scale (gamma) and shift (beta) parameters
    /// * `gamma_init` - Function to initialize gamma
    /// * `beta_init` - Function to initialize beta
    pub fn new<F, G>(
        num_features: usize,
        momentum: Option<f32>,
        eps: Option<f32>,
        affine: bool,
        gamma_init: F,
        beta_init: G,
    ) -> Self
    where
        F: Fn(&[usize]) -> Tensor,
        G: Fn(&[usize]) -> Tensor,
    {
        let momentum = momentum.unwrap_or(0.1);
        let eps = eps.unwrap_or(1e-5);
        
        // Initialize parameters if affine is true
        let (gamma, beta) = if affine {
            let gamma_data = gamma_init(&[num_features]);
            let beta_data = beta_init(&[num_features]);
            
            let gamma = Arc::new(Node::new_leaf(gamma_data));
            let beta = Arc::new(Node::new_leaf(beta_data));
            
            (gamma, beta)
        } else {
            // Use fixed scale of 1 and shift of 0
            let ones = Arc::new(Node::new_leaf(Tensor::ones(&[num_features]).unwrap()));
            let zeros = Arc::new(Node::new_leaf(Tensor::zeros(&[num_features]).unwrap()));
            
            (ones, zeros)
        };
        
        // Initialize running statistics
        let running_mean = Tensor::zeros(&[num_features]).unwrap();
        let running_var = Tensor::ones(&[num_features]).unwrap();
        
        Self {
            gamma,
            beta,
            running_mean,
            running_var,
            num_features,
            momentum,
            eps,
            training: true,
        }
    }
    
    /// Applies batch normalization to the input.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [batch_size, num_features, height, width]
    ///
    /// # Returns
    /// Normalized output tensor of the same shape as input
    pub fn forward(
        &mut self,
        input: Arc<Node>,
    ) -> Result<Arc<Node>, Box<dyn std::error::Error>> {
        let input_shape = input.tensor.shape();
        let batch_size = input_shape[0];
        let num_features = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];
        
        assert_eq!(
            num_features, self.num_features,
            "Number of features in input ({}) does not match expected ({})",
            num_features, self.num_features
        );
        
        if self.training {
            // Training mode: use batch statistics and update running statistics
            
            // Calculate mean and variance over the batch and spatial dimensions
            let axes = &[0, 2, 3]; // Reduce over batch, height, width
            
            // Calculate mean [num_features]
            let mean = input.tensor.mean(axes, true)?;
            
            // Calculate variance [num_features]
            let squared_diff = input.tensor.sub(&mean.reshape(&[1, num_features, 1, 1])?)?.powf(2.0)?;
            let var = squared_diff.mean(axes, true)?;
            
            // Update running statistics
            self.running_mean = self.running_mean.mul_scalar(1.0 - self.momentum)?
                .add(&mean.mul_scalar(self.momentum)?)?;
                
            let unbiased_var = var.mul_scalar(batch_size as f32 * height as f32 * width as f32 / 
                ((batch_size * height * width).saturating_sub(1)) as f32)?;
                
            self.running_var = self.running_var.mul_scalar(1.0 - self.momentum)?
                .add(&unbiased_var.mul_scalar(self.momentum)?)?;
            
            // Normalize using batch statistics
            self.normalize(input, mean, var)
        } else {
            // Inference mode: use running statistics
            let mean = self.running_mean.clone();
            let var = self.running_var.clone();
            
            self.normalize(input, mean, var)
        }
    }
    
    /// Internal helper function to apply normalization
    fn normalize(
        &self,
        input: Arc<Node>,
        mean: Tensor,
        var: Tensor,
    ) -> Result<Arc<Node>, Box<dyn std::error::Error>> {
        let input_shape = input.tensor.shape();
        let num_features = input_shape[1];
        
        // Add epsilon for numerical stability and compute standard deviation
        let std = var.add_scalar(self.eps)?.sqrt()?;
        
        // Reshape parameters for broadcasting
        let mean_reshaped = mean.reshape(&[1, num_features, 1, 1])?;
        let std_reshaped = std.reshape(&[1, num_features, 1, 1])?;
        
        // Normalize: (x - mean) / sqrt(var + eps)
        let normalized = input.sub(&mean_reshaped)?.div(&std_reshaped)?;
        
        // Scale and shift: gamma * normalized + beta
        let gamma_reshaped = self.gamma.reshape(&[1, num_features, 1, 1])?;
        let beta_reshaped = self.beta.reshape(&[1, num_features, 1, 1])?;
        
        let scaled = gamma_reshaped.mul(&normalized)?;
        let output = scaled.add(&beta_reshaped)?;
        
        Ok(Arc::new(Node::new_leaf(output)))
    }
    
    /// Sets the layer to training mode.
    pub fn train(&mut self) {
        self.training = true;
    }
    
    /// Sets the layer to evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
    }
    
    /// Returns whether the layer is in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }
    
    /// Returns the scale parameter (gamma).
    pub fn gamma(&self) -> &Arc<Node> {
        &self.gamma
    }
    
    /// Returns the shift parameter (beta).
    pub fn beta(&self) -> &Arc<Node> {
        &self.beta
    }
    
    /// Returns the running mean.
    pub fn running_mean(&self) -> &BaseTensor {
        &self.running_mean
    }
    
    /// Returns the running variance.
    pub fn running_var(&self) -> &BaseTensor {
        &self.running_var
    }
    
    /// Returns the number of features.
    pub fn num_features(&self) -> usize {
        self.num_features
    }
    
    /// Returns the momentum value.
    pub fn momentum(&self) -> f32 {
        self.momentum
    }
    
    /// Returns the epsilon value.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_batchnorm2d_forward_train() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a batch norm layer
        let num_features = 2;
        let mut bn = BatchNorm2d::new(
            &mut graph,
            num_features,
            Some(0.1), // momentum
            Some(1e-5), // eps
            true,      // affine
            |shape| BaseTensor::ones(shape),
            |shape| BaseTensor::zeros(shape),
        );
        
        // Create input tensor [batch_size, num_features, height, width]
        let input_data = BaseTensor::from_slice(
            &[
                // First feature map
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                // Second feature map (shifted by 10)
                11.0, 12.0, 13.0,
                14.0, 15.0, 16.0,
                // Second sample
                -1.0, -2.0, -3.0,
                -4.0, -5.0, -6.0,
                // Second feature map (shifted by -10)
                9.0, 8.0, 7.0,
                6.0, 5.0, 4.0,
            ],
            &[2, 2, 3, 2], // [batch_size, num_features, height, width]
        )?;
        
        let input = graph.add_tensor(input_data, true);
        
        // Forward pass in training mode
        let output = bn.forward(&mut graph, input)?;
        
        // Check output shape
        assert_eq!(output.tensor.shape(), &[2, 2, 3, 2]);
        
        // For the first feature map, mean should be ~0 and std ~1 (approximately)
        let output_data = output.tensor.to_vec::<f32>()?;
        
        // Check that the output has approximately zero mean and unit variance
        // for each feature map across the batch and spatial dimensions
        for f in 0..num_features {
            let mut sum = 0.0;
            let mut count = 0;
            
            for b in 0..2 {
                for h in 0..3 {
                    for w in 0..2 {
                        let idx = b * num_features * 3 * 2 + f * 3 * 2 + h * 2 + w;
                        sum += output_data[idx];
                        count += 1;
                    }
                }
            }
            
            let mean = sum / count as f32;
            assert_relative_eq!(mean, 0.0, epsilon = 1e-5);
            
            // Calculate variance
            let mut var = 0.0;
            for b in 0..2 {
                for h in 0..3 {
                    for w in 0..2 {
                        let idx = b * num_features * 3 * 2 + f * 3 * 2 + h * 2 + w;
                        let diff = output_data[idx] - mean;
                        var += diff * diff;
                    }
                }
            }
            
            var /= (count - 1) as f32; // Unbiased estimator
            assert_relative_eq!(var, 1.0, epsilon = 1e-5);
        }
        
        // Check that running statistics were updated
        assert_ne!(bn.running_mean().to_vec::<f32>()?[0], 0.0);
        assert_ne!(bn.running_var().to_vec::<f32>()?[0], 1.0);
        
        Ok(())
    }
    
    #[test]
    fn test_batchnorm2d_forward_eval() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a batch norm layer
        let num_features = 2;
        let mut bn = BatchNorm2d::new(
            &mut graph,
            num_features,
            Some(0.1), // momentum
            Some(1e-5), // eps
            true,      // affine
            |shape| BaseTensor::ones(shape),
            |shape| BaseTensor::zeros(shape),
        );
        
        // Set to evaluation mode
        bn.eval();
        
        // Manually set running statistics
        let mean = BaseTensor::from_slice(&[2.0, 12.0], &[num_features])?;
        let var = BaseTensor::from_slice(&[1.5, 1.5], &[num_features])?;
        
        // Use reflection to set private fields for testing
        let bn_ref = &mut bn as *mut _ as *mut BatchNorm2dMut;
        unsafe {
            (*bn_ref).running_mean = mean;
            (*bn_ref).running_var = var;
        }
        
        // Create input tensor [batch_size, num_features, height, width]
        let input_data = BaseTensor::from_slice(
            &[
                // First feature map
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                // Second feature map (shifted by 10)
                11.0, 12.0, 13.0,
                14.0, 15.0, 16.0,
            ],
            &[1, 2, 3, 2], // [batch_size, num_features, height, width]
        )?;
        
        let input = graph.add_tensor(input_data, true);
        
        // Forward pass in evaluation mode
        let output = bn.forward(&mut graph, input)?;
        
        // Check output shape
        assert_eq!(output.tensor.shape(), &[1, 2, 3, 2]);
        
        // Check that the output is normalized using the running statistics
        let output_data = output.tensor.to_vec::<f32>()?;
        
        // For the first feature map: (x - 2.0) / sqrt(1.5 + 1e-5)
        // For the second feature map: (x - 12.0) / sqrt(1.5 + 1e-5)
        let expected = vec![
            // First feature map
            (1.0 - 2.0) / 1.2247449, (2.0 - 2.0) / 1.2247449,
            (3.0 - 2.0) / 1.2247449, (4.0 - 2.0) / 1.2247449,
            (5.0 - 2.0) / 1.2247449, (6.0 - 2.0) / 1.2247449,
            // Second feature map
            (11.0 - 12.0) / 1.2247449, (12.0 - 12.0) / 1.2247449,
            (13.0 - 12.0) / 1.2247449, (14.0 - 12.0) / 1.2247449,
            (15.0 - 12.0) / 1.2247449, (16.0 - 12.0) / 1.2247449,
        ];
        
        for (i, &val) in expected.iter().enumerate() {
            assert_relative_eq!(output_data[i], val, epsilon = 1e-5);
        }
        
        // Check that running statistics were not updated
        assert_eq!(bn.running_mean().to_vec::<f32>()?, vec![2.0, 12.0]);
        assert_eq!(bn.running_var().to_vec::<f32>()?, vec![1.5, 1.5]);
        
        Ok(())
    }
    
    #[test]
    fn test_batchnorm2d_backward() -> Result<(), Box<dyn std::error::Error>> {
        let mut graph = ComputationGraph::new();
        
        // Create a batch norm layer
        let num_features = 2;
        let mut bn = BatchNorm2d::new(
            &mut graph,
            num_features,
            Some(0.1), // momentum
            Some(1e-5), // eps
            true,      // affine
            |shape| BaseTensor::ones(shape),
            |shape| BaseTensor::zeros(shape),
        );
        
        // Create input tensor [batch_size, num_features, height, width]
        let input_data = BaseTensor::randn(&[2, 2, 3, 2], 0.0, 1.0);
        let input = graph.add_tensor(input_data, true);
        
        // Forward pass
        let output = bn.forward(&mut graph, input.clone())?;
        
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
        assert!(bn.gamma().gradient.is_some(), "Gamma gradient should be computed");
        assert!(bn.beta().gradient.is_some(), "Beta gradient should be computed");
        assert!(input.gradient.is_some(), "Input gradient should be computed");
        
        Ok(())
    }
}

// Helper struct to access private fields of BatchNorm2d for testing
#[allow(dead_code)]
struct BatchNorm2dMut {
    gamma: Arc<Node>,
    beta: Arc<Node>,
    running_mean: BaseTensor,
    running_var: BaseTensor,
    num_features: usize,
    momentum: f32,
    eps: f32,
    training: bool,
}
