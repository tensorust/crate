//! Neural network building blocks for Tensorust.
//! This module provides various neural network components like layers, activations, losses, and optimizers.

mod layers;
mod activations;
mod losses;
mod optimizers;
mod init;

pub use layers::*;
pub use activations::*;
pub use losses::*;
pub use optimizers::*;
pub use init::*;

use crate::{
    dimension::Dimension,
    tensor::Tensor,
    storage::Storage,
    error::Result,
};

/// Trait for all neural network layers.
pub trait Layer<T, D, S>: 'static + Send + Sync + std::fmt::Debug
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// The input type of the layer.
    type Input: Dimension;
    /// The output type of the layer.
    type Output: Dimension;

    /// Forward pass through the layer.
    fn forward(&self, input: &Tensor<T, Self::Input, S>) -> Result<Tensor<T, Self::Output, S>>;

    /// Backward pass through the layer.
    /// 
    /// # Arguments
    /// * `input` - The input to the forward pass.
    /// * `output` - The output from the forward pass.
    /// * `grad_output` - The gradient of the loss with respect to the output.
    /// 
    /// Returns:
    /// * The gradient of the loss with respect to the input.
    /// * A vector of parameter gradients, if any.
    fn backward(
        &self,
        input: &Tensor<T, Self::Input, S>,
        output: &Tensor<T, Self::Output, S>,
        grad_output: &Tensor<T, Self::Output, S>,
    ) -> Result<(
        Tensor<T, Self::Input, S>,
        Option<Vec<Tensor<T, crate::dimension::dynamic::DynamicDim, S>>>,
    )>;

    /// Get the trainable parameters of the layer, if any.
    fn parameters(&self) -> Vec<&dyn std::any::Any>;
}

/// Trait for activation functions.
pub trait Activation<T, D, S>: 'static + Send + Sync + std::fmt::Debug
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Apply the activation function.
    fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>;
    
    /// Compute the gradient of the activation function.
    fn backward(
        &self,
        input: &Tensor<T, D, S>,
        output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>>;
}

/// Trait for loss functions.
pub trait Loss<T, D, S>: 'static + Send + Sync + std::fmt::Debug
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Compute the loss between predictions and targets.
    fn compute(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, crate::dimension::static_::StaticDim<0>, S>>;
    
    /// Compute the gradient of the loss with respect to the predictions.
    fn gradient(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>>;
}

/// Trait for optimization algorithms.
pub trait Optimizer<T, S>: 'static + Send + Sync + std::fmt::Debug
where
    T: Clone + Send + Sync + 'static,
    S: Storage<T>,
{
    /// Update the parameters using their gradients.
    fn step(&mut self, parameters: &[&mut Tensor<T, crate::dimension::dynamic::DynamicDim, S>]);
    
    /// Zero out the gradients of the parameters.
    fn zero_grad(&self, parameters: &[&mut Tensor<T, crate::dimension::dynamic::DynamicDim, S>]);
}

/// A sequential container for layers.
#[derive(Debug)]
pub struct Sequential<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    layers: Vec<Box<dyn Layer<T, D, S, Input = D, Output = D>>>,
}

impl<T, D, S> Sequential<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Create a new sequential container.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }
    
    /// Add a layer to the container.
    pub fn add<L>(mut self, layer: L) -> Self
    where
        L: Layer<T, D, S, Input = D, Output = D> + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }
    
    /// Forward pass through all layers.
    pub fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }
    
    /// Backward pass through all layers.
    pub fn backward(
        &self,
        input: &Tensor<T, D, S>,
        output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<(Tensor<T, D, S>, Option<Vec<Vec<Tensor<T, crate::dimension::dynamic::DynamicDim, S>>>>)> {
        let mut grad_input = grad_output.clone();
        let mut param_grads = Vec::new();
        
        // We need to keep track of intermediate outputs for the backward pass
        // In a real implementation, you might want to store these during the forward pass
        let mut inputs = vec![input.clone()];
        let mut outputs = Vec::new();
        
        // Forward pass to collect intermediate outputs
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
            outputs.push(current.clone());
            inputs.push(current.clone());
        }
        
        // Backward pass
        for (i, layer) in self.layers.iter().rev().enumerate() {
            let idx = self.layers.len() - i - 1;
            let input = &inputs[idx];
            let output = &outputs[idx];
            
            let (gi, pg) = layer.backward(input, output, &grad_input)?;
            grad_input = gi;
            
            if let Some(pg) = pg {
                param_grads.push(pg);
            }
        }
        
        // Reverse to match layer order
        param_grads.reverse();
        
        Ok((grad_input, Some(param_grads)))
    }
    
    /// Get all trainable parameters from all layers.
    pub fn parameters(&self) -> Vec<&dyn std::any::Any> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
}

impl<T, D, S> Default for Sequential<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dimension::StaticDim,
        storage::CpuStorage,
        tensor,
    };
    
    #[test]
    fn test_sequential() {
        // This is a simple test to verify the sequential container works
        // In a real test, you would test with actual layers
        let model: Sequential<f32, StaticDim<2>, CpuStorage<f32>> = Sequential::new();
        // Add layers and test forward/backward passes
    }
}
