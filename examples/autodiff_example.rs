//! Example of using Tensorust's automatic differentiation.
//!
//! This example demonstrates how to use the automatic differentiation
//! system to compute gradients for a simple neural network.

use tensorust::{
    autodiff::{
        tensor::Tensor,
        ops::{MatMulOp, ReLUOp, CrossEntropyLossOp},
        ComputationGraph,
    },
    tensor::Tensor as BaseTensor,
};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple neural network with one hidden layer
    // Network architecture:
    // Input (2) -> Dense (4) -> ReLU -> Dense (2) -> Softmax
    
    // Create computation graph
    let mut graph = ComputationGraph::new();
    
    // Input data (batch_size=3, input_dim=2)
    let input_data = BaseTensor::from_slice(
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        &[3, 2],  // 3 samples, 2 features each
    )?;
    
    // Target class indices (batch_size=3)
    let targets = BaseTensor::from_slice(&[1, 0, 1], &[3])?;
    
    // Add input tensors to the graph
    let input = graph.add_tensor(input_data, false);  // Don't need gradient for input
    let targets = graph.add_tensor(targets, false);   // Don't need gradient for targets
    
    // Initialize weights and biases for the first layer (input_dim=2, hidden_dim=4)
    let w1_data = BaseTensor::from_slice(
        &[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
        &[2, 4],  // input_dim x hidden_dim
    )?;
    
    let b1_data = BaseTensor::from_slice(&[0.1, 0.2, 0.3, 0.4], &[4])?;
    
    // Add trainable parameters to the graph
    let w1 = graph.add_tensor(w1_data, true);  // Needs gradient
    let b1 = graph.add_tensor(b1_data, true);  // Needs gradient
    
    // First layer: input @ w1 + b1
    let fc1 = graph.add_op(
        Arc::new(MatMulOp),
        &[input, w1.clone()],
        true,
    )?;
    
    let fc1_bias = graph.add_op(
        Arc::new(AddOp),
        &[fc1, b1.clone()],
        true,
    )?;
    
    // ReLU activation
    let relu1 = graph.add_op(
        Arc::new(ReLUOp),
        &[fc1_bias],
        true,
    )?;
    
    // Initialize weights and biases for the second layer (hidden_dim=4, num_classes=2)
    let w2_data = BaseTensor::from_slice(
        &[0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4],
        &[4, 2],  // hidden_dim x num_classes
    )?;
    
    let b2_data = BaseTensor::from_slice(&[0.1, -0.1], &[2])?;
    
    // Add second layer parameters
    let w2 = graph.add_tensor(w2_data, true);  // Needs gradient
    let b2 = graph.add_tensor(b2_data, true);  // Needs gradient
    
    // Second layer: relu1 @ w2 + b2
    let logits = graph.add_op(
        Arc::new(MatMulOp),
        &[relu1, w2.clone()],
        true,
    )?;
    
    let logits = graph.add_op(
        Arc::new(AddOp),
        &[logits, b2.clone()],
        true,
    )?;
    
    // Compute cross-entropy loss
    let loss = graph.add_op(
        Arc::new(CrossEntropyLossOp::new("mean")),
        &[logits, targets],
        true,
    )?;
    
    // Forward pass
    let loss_value = loss.tensor.to_scalar::<f32>()?;
    println!("Initial loss: {:.4}", loss_value);
    
    // Backward pass
    graph.zero_grad();
    graph.backward(&loss)?;
    
    // Print gradients
    println!("Gradient for w1: {:?}", w1.gradient.as_ref().unwrap().to_vec::<f32>()?);
    println!("Gradient for b1: {:?}", b1.gradient.as_ref().unwrap().to_vec::<f32>()?);
    println!("Gradient for w2: {:?}", w2.gradient.as_ref().unwrap().to_vec::<f32>()?);
    println!("Gradient for b2: {:?}", b2.gradient.as_ref().unwrap().to_vec::<f32>()?);
    
    // Update parameters using gradient descent
    let learning_rate = 0.1;
    
    // Helper function to update parameters
    fn update_param(
        param: &mut BaseTensor,
        grad: &BaseTensor,
        lr: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let updated = param.sub(&grad.mul_scalar(lr)?)?;
        *param = updated;
        Ok(())
    }
    
    // Update parameters
    update_param(
        unsafe { &mut *(&w1.tensor as *const _ as *mut BaseTensor) },
        w1.gradient.as_ref().unwrap(),
        learning_rate,
    )?;
    
    update_param(
        unsafe { &mut *(&b1.tensor as *const _ as *mut BaseTensor) },
        b1.gradient.as_ref().unwrap(),
        learning_rate,
    )?;
    
    update_param(
        unsafe { &mut *(&w2.tensor as *const _ as *mut BaseTensor) },
        w2.gradient.as_ref().unwrap(),
        learning_rate,
    )?;
    
    update_param(
        unsafe { &mut *(&b2.tensor as *const _ as *mut BaseTensor) },
        b2.gradient.as_ref().unwrap(),
        learning_rate,
    )?;
    
    // Forward pass with updated parameters
    graph.zero_grad();
    let new_loss = loss.tensor.to_scalar::<f32>()?;
    println!("Loss after one update: {:.4}", new_loss);
    
    // The loss should decrease after the update
    assert!(new_loss < loss_value, "Loss should decrease after parameter update");
    
    Ok(())
}
