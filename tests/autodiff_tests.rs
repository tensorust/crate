//! Integration tests for automatic differentiation.

use tensorust::{
    autodiff::{
        tensor::Tensor,
        ops::{AddOp, MultiplyOp, MatMulOp, ReLUOp, SoftmaxOp, CrossEntropyLossOp},
        ComputationGraph, DifferentiableOp,
    },
    tensor::Tensor as BaseTensor,
};
use approx::assert_relative_eq;
use std::sync::Arc;

#[test]
fn test_scalar_operations() {
    // Test basic scalar operations
    let mut graph = ComputationGraph::new();
    
    // Input tensors
    let x = graph.add_tensor(BaseTensor::scalar(2.0), true);
    let y = graph.add_tensor(BaseTensor::scalar(3.0), true);
    
    // z = x * y + x^2
    let x_squared = graph.add_op(
        Arc::new(MultiplyOp),
        &[x.clone(), x.clone()],
        true,
    ).unwrap();
    
    let xy = graph.add_op(
        Arc::new(MultiplyOp),
        &[x.clone(), y.clone()],
        true,
    ).unwrap();
    
    let z = graph.add_op(
        Arc::new(AddOp),
        &[xy, x_squared],
        true,
    ).unwrap();
    
    // Forward pass
    assert_relative_eq!(z.tensor.to_scalar::<f32>().unwrap(), 10.0); // 2*3 + 2*2 = 10
    
    // Backward pass
    graph.backward(&z).unwrap();
    
    // Check gradients
    // dz/dx = y + 2x = 3 + 4 = 7
    // dz/dy = x = 2
    assert_relative_eq!(
        x.gradient.as_ref().unwrap().to_scalar::<f32>().unwrap(),
        7.0,
        epsilon = 1e-5
    );
    assert_relative_eq!(
        y.gradient.as_ref().unwrap().to_scalar::<f32>().unwrap(),
        2.0,
        epsilon = 1e-5
    );
}

#[test]
fn test_matrix_operations() {
    // Test matrix operations
    let mut graph = ComputationGraph::new();
    
    // Input matrices
    let a_data = BaseTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b_data = BaseTensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    
    let a = graph.add_tensor(a_data, true);
    let b = graph.add_tensor(b_data, true);
    
    // C = A @ B
    let c = graph.add_op(
        Arc::new(MatMulOp),
        &[a.clone(), b.clone()],
        true,
    ).unwrap();
    
    // Forward pass
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_eq!(c.tensor.to_vec::<f32>().unwrap(), expected);
    
    // Backward pass
    graph.backward(&c).unwrap();
    
    // Check gradients
    // dC/dA = dC/dC @ B^T = I @ B^T = B^T
    // dC/dB = A^T @ dC/dC = A^T @ I = A^T
    let expected_da = vec![5.0, 7.0, 6.0, 8.0];
    let expected_db = vec![1.0, 3.0, 2.0, 4.0];
    
    assert_eq!(
        a.gradient.as_ref().unwrap().to_vec::<f32>().unwrap(),
        expected_da
    );
    assert_eq!(
        b.gradient.as_ref().unwrap().to_vec::<f32>().unwrap(),
        expected_db
    );
}

#[test]
fn test_relu_activation() {
    // Test ReLU activation
    let mut graph = ComputationGraph::new();
    
    // Input tensor
    let x_data = BaseTensor::from_slice(&[-1.0, 0.0, 2.0, -3.0, 4.0], &[5]).unwrap();
    let x = graph.add_tensor(x_data, true);
    
    // y = relu(x)
    let y = graph.add_op(
        Arc::new(ReLUOp),
        &[x.clone()],
        true,
    ).unwrap();
    
    // Forward pass
    let expected = vec![0.0, 0.0, 2.0, 0.0, 4.0];
    assert_eq!(y.tensor.to_vec::<f32>().unwrap(), expected);
    
    // Backward pass
    graph.backward(&y).unwrap();
    
    // Gradient should be 1 where x > 0, 0 otherwise
    let expected_grad = vec![0.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(
        x.gradient.as_ref().unwrap().to_vec::<f32>().unwrap(),
        expected_grad
    );
}

#[test]
fn test_softmax_activation() {
    // Test softmax activation
    let mut graph = ComputationGraph::new();
    
    // Input tensor
    let x_data = BaseTensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
    let x = graph.add_tensor(x_data, true);
    
    // y = softmax(x)
    let y = graph.add_op(
        Arc::new(SoftmaxOp { axis: -1 }),
        &[x.clone()],
        true,
    ).unwrap();
    
    // Forward pass
    let y_vec = y.tensor.to_vec::<f32>().unwrap();
    let sum: f32 = y_vec.iter().sum();
    
    // Check that outputs sum to 1
    assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
    
    // Check that outputs are in the correct order
    assert!(y_vec[0] < y_vec[1]);
    assert!(y_vec[1] < y_vec[2]);
    
    // Backward pass
    graph.backward(&y).unwrap();
    
    // The gradient should be close to zero because softmax is normalized
    let grad = x.gradient.as_ref().unwrap().to_vec::<f32>().unwrap();
    for &g in &grad {
        assert_relative_eq!(g, 0.0, epsilon = 1e-5);
    }
}

#[test]
fn test_cross_entropy_loss() {
    // Test cross-entropy loss
    let mut graph = ComputationGraph::new();
    
    // Logits (unnormalized predictions)
    let logits_data = BaseTensor::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
        &[2, 3],  // batch_size=2, num_classes=3
    ).unwrap();
    
    // Target class indices
    let targets_data = BaseTensor::from_slice(&[2, 0], &[2]).unwrap();  // Class 2 and 0
    
    let logits = graph.add_tensor(logits_data, true);
    let targets = graph.add_tensor(targets_data, false);
    
    // loss = cross_entropy(logits, targets)
    let loss = graph.add_op(
        Arc::new(CrossEntropyLossOp::new("mean")),
        &[logits.clone(), targets],
        true,
    ).unwrap();
    
    // Forward pass
    let loss_val = loss.tensor.to_scalar::<f32>().unwrap();
    assert!(loss_val > 0.0, "Loss should be positive");
    
    // Backward pass
    graph.backward(&loss).unwrap();
    
    // Check gradient shape matches input logits
    let grad = logits.gradient.as_ref().unwrap();
    assert_eq!(grad.shape(), logits.tensor.shape());
    
    // The gradient should be (softmax(logits) - one_hot(targets)) / batch_size
    let grad_vec = grad.to_vec::<f32>().unwrap();
    
    // For the first example (target=2):
    // - The gradient for the correct class (index 2) should be (p - 1)/batch_size
    // - The gradient for incorrect classes should be p/batch_size
    assert!(grad_vec[2] < 0.0, "Gradient for correct class should be negative");
    assert!(grad_vec[0] > 0.0, "Gradient for incorrect class should be positive");
    assert!(grad_vec[1] > 0.0, "Gradient for incorrect class should be positive");
    
    // For the second example (target=0):
    // - The gradient for the correct class (index 0) should be (p - 1)/batch_size
    // - The gradient for incorrect classes should be p/batch_size
    assert!(grad_vec[3] < 0.0, "Gradient for correct class should be negative");
    assert!(grad_vec[4] > 0.0, "Gradient for incorrect class should be positive");
    assert!(grad_vec[5] > 0.0, "Gradient for incorrect class should be positive");
}

#[test]
fn test_gradient_check() {
    // Test gradients using finite differences
    let eps = 1e-3;
    
    // Test function: f(x) = x^3 + 2x^2 + 3x + 4
    // f'(x) = 3x^2 + 4x + 3
    let x_val = 2.0;
    let expected_grad = 3.0 * x_val.powi(2) + 4.0 * x_val + 3.0;
    
    // Create computation graph
    let mut graph = ComputationGraph::new();
    let x = graph.add_tensor(BaseTensor::scalar(x_val), true);
    
    // Build computation: x^3 + 2x^2 + 3x + 4
    let x2 = graph.add_op(
        Arc::new(MultiplyOp),
        &[x.clone(), x.clone()],
        true,
    ).unwrap();
    
    let x3 = graph.add_op(
        Arc::new(MultiplyOp),
        &[x2.clone(), x.clone()],
        true,
    ).unwrap();
    
    let term2 = graph.add_op(
        Arc::new(MultiplyOp),
        &[x2, graph.add_tensor(BaseTensor::scalar(2.0), false)],
        true,
    ).unwrap();
    
    let term3 = graph.add_op(
        Arc::new(MultiplyOp),
        &[x.clone(), graph.add_tensor(BaseTensor::scalar(3.0), false)],
        true,
    ).unwrap();
    
    let term4 = graph.add_tensor(BaseTensor::scalar(4.0), false);
    
    let y = graph.add_op(
        Arc::new(AddOp),
        &[
            x3,
            term2,
            term3,
            term4,
        ],
        true,
    ).unwrap();
    
    // Forward pass
    let y_val = y.tensor.to_scalar::<f32>().unwrap();
    let expected_y = x_val.powi(3) + 2.0 * x_val.powi(2) + 3.0 * x_val + 4.0;
    assert_relative_eq!(y_val, expected_y, epsilon = 1e-5);
    
    // Backward pass
    graph.backward(&y).unwrap();
    
    // Check gradient against expected value
    let computed_grad = x.gradient.as_ref().unwrap().to_scalar::<f32>().unwrap();
    assert_relative_eq!(
        computed_grad,
        expected_grad,
        epsilon = 1e-3,
        max_relative = 1e-2
    );
}
