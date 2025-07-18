//! Dropout layer implementation.
//!
//! This module implements the Dropout layer, which randomly sets a fraction of
//! input units to 0 during training, which helps prevent overfitting.

use crate::{
    autodiff::{Node},
    tensor::Tensor,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

/// Dropout layer.
///
/// During training, randomly zeroes some of the elements of the input tensor
/// with probability `p` using samples from a Bernoulli distribution.
/// The remaining elements are scaled by `1/(1-p)` to maintain the same sum.
#[derive(Debug)]
pub struct Dropout {
    /// Probability of an element to be zeroed
    p: f32,
    
    /// Whether the layer is in training mode
    training: bool,
    
    /// Random number generator
    rng: StdRng,
    
    /// Random seed for reproducibility
    seed: u64,
}

impl Dropout {
    /// Creates a new Dropout layer.
    ///
    /// # Arguments
    /// * `p` - Probability of an element to be zeroed
    /// * `seed` - Optional random seed for reproducibility
    pub fn new(p: f32, seed: Option<u64>) -> Self {
        assert!(
            p >= 0.0 && p < 1.0,
            "Dropout probability must be in range [0, 1), got {}",
            p
        );
        
        let seed = seed.unwrap_or_else(|| rand::random());
        let rng = StdRng::seed_from_u64(seed);
        
        Self {
            p,
            training: true,
            rng,
            seed,
        }
    }
    
    /// Applies dropout to the input.
    ///
    /// # Arguments
    /// * `input` - Input tensor of any shape
    ///
    /// # Returns
    /// Output tensor of the same shape as input
    pub fn forward(
        &mut self,
        input: Arc<Node>,
    ) -> Result<Arc<Node>, Box<dyn std::error::Error>> {
        if !self.training || self.p == 0.0 {
            return Ok(input);
        }
        
        // Create a binary mask with probability 1-p of being 1
        let input_shape = input.tensor.shape();
        let mask_data: Vec<f32> = (0..input_shape.iter().product())
            .map(|_| if self.rng.gen::<f32>() < self.p { 0.0 } else { 1.0 / (1.0 - self.p) })
            .collect();
        
        let mask = Tensor::from_slice(
            &mask_data,
            input_shape.to_vec(),
        )?;
        
        // Element-wise multiplication with the mask
        let output = input.mul(&mask)?;

        Ok(Arc::new(Node::new_leaf(output)))
    }
    
    /// Sets the layer to training mode.
    pub fn train(&mut self) {
        self.training = true;
        // Reset RNG to ensure consistent behavior
        self.rng = StdRng::seed_from_u64(self.seed);
    }
    
    /// Sets the layer to evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
    }
    
    /// Returns whether the layer is in training mode.
    pub fn is_training(&self) -> bool {
        self.training
    }
    
    /// Returns the dropout probability.
    pub fn probability(&self) -> f32 {
        self.p
    }
    
    /// Returns the random seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_dropout_forward_train() -> Result<(), Box<dyn std::error::Error>> {
        // Create a dropout layer with p=0.5 and fixed seed for testing
        let mut dropout = Dropout::new(0.5, Some(42));
        
        // Create input tensor
        let input_data = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],  // batch_size=2, features=3
        )?;
        
        let input = Arc::new(Node::new_leaf(input_data));
        
        // Forward pass in training mode
        let output = dropout.forward(input)?;
        
        // Check output shape
        assert_eq!(output.tensor.shape(), &[2, 3]);
        
        // With seed=42 and p=0.5, we know exactly which elements should be zeroed
        // This is deterministic because we set a fixed seed
        let output_data = output.tensor.to_vec::<f32>()?;
        let expected = vec![
            0.0,  // 1.0 * 0 (dropped)
            4.0,  // 2.0 * 2 (scaled by 1/(1-0.5) = 2)
            0.0,  // 3.0 * 0 (dropped)
            0.0,  // 4.0 * 0 (dropped)
            10.0, // 5.0 * 2 (scaled)
            0.0,  // 6.0 * 0 (dropped)
        ];
        
        assert_eq!(output_data, expected);
        
        Ok(())
    }
    
    #[test]
    fn test_dropout_forward_eval() -> Result<(), Box<dyn std::error::Error>> {
        // Create a dropout layer with p=0.5
        let mut dropout = Dropout::new(0.5, Some(42));
        dropout.eval();
        
        // Create input tensor
        let input_data = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],  // batch_size=2, features=3
        )?;
        
        let input = Arc::new(Node::new_leaf(input_data));
        
        // Forward pass in evaluation mode
        let output = dropout.forward(input)?;
        
        // In evaluation mode, the output should be identical to the input
        let output_data = output.tensor.to_vec::<f32>()?;
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        assert_eq!(output_data, expected);
        
        Ok(())
    }
    
    #[test]
    fn test_dropout_backward() -> Result<(), Box<dyn std::error::Error>> {
        // Create a dropout layer with p=0.5 and fixed seed for testing
        let mut dropout = Dropout::new(0.5, Some(42));
        
        // Create input tensor
        let input_data = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],  // batch_size=2, features=3
        )?;
        
        let input = Arc::new(Node::new_leaf(input_data));
        
        // Forward pass in training mode
        let output = dropout.forward(input.clone())?;
        
        // Create a dummy loss (sum of outputs)
        let loss = output.sum();
        
        // Backward pass
        loss.backward();
        
        // Check gradients
        let grad = input.gradient.as_ref().unwrap().to_vec::<f32>()?;
        
        // The gradient should be the same as the dropout mask (0 or 2)
        // With the fixed seed, we know exactly which elements should be zeroed
        let expected_grad = vec![
            0.0, // dropped
            2.0, // kept (scaled by 2)
            0.0, // dropped
            0.0, // dropped
            2.0, // kept (scaled by 2)
            0.0, // dropped
        ];
        
        assert_eq!(grad, expected_grad);
        
        Ok(())
    }
    
    #[test]
    fn test_dropout_zero_probability() -> Result<(), Box<dyn std::error::Error>> {
        // Create a dropout layer with p=0 (should be equivalent to identity)
        let mut dropout = Dropout::new(0.0, None);
        
        // Create input tensor
        let input_data = Tensor::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],  // batch_size=2, features=3
        )?;
        
        let input = Arc::new(Node::new_leaf(input_data));
        
        // Forward pass in training mode
        let output = dropout.forward(input.clone())?;
        
        // With p=0, the output should be identical to the input
        let output_data = output.tensor.to_vec::<f32>()?;
        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        assert_eq!(output_data, expected);
        
        // Check that input and output are the same tensor (no copy)
        assert_eq!(input.tensor.as_ptr(), output.tensor.as_ptr());
        
        Ok(())
    }
}
