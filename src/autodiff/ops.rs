//! Operations for automatic differentiation.

use super::*;
use crate::tensor::Tensor as BaseTensor;

/// Addition operation.
#[derive(Debug)]
pub struct AddOp;

impl DifferentiableOp for AddOp {
    fn forward(&self, inputs: &[&BaseTensor]) -> Result<BaseTensor, AutodiffError> {
        if inputs.len() != 2 {
            return Err(AutodiffError::InvalidInput(
                "Add operation requires exactly 2 inputs".to_string(),
            ));
        }
        
        inputs[0]
            .add(inputs[1])
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))
    }
    
    fn backward(&self, grad: &BaseTensor, inputs: &[&BaseTensor], _output: &BaseTensor) -> Vec<BaseTensor> {
        // For z = x + y, dz/dx = 1, dz/dy = 1
        // So the gradient is just passed through to both inputs
        vec![grad.clone(), grad.clone()]
    }
    
    fn input_names(&self) -> Vec<&'static str> {
        vec!["x", "y"]
    }
}

/// Multiplication operation.
#[derive(Debug)]
pub struct MultiplyOp;

impl DifferentiableOp for MultiplyOp {
    fn forward(&self, inputs: &[&BaseTensor]) -> Result<BaseTensor, AutodiffError> {
        if inputs.len() != 2 {
            return Err(AutodiffError::InvalidInput(
                "Multiply operation requires exactly 2 inputs".to_string(),
            ));
        }
        
        inputs[0]
            .mul(inputs[1])
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))
    }
    
    fn backward(&self, grad: &BaseTensor, inputs: &[&BaseTensor], _output: &BaseTensor) -> Vec<BaseTensor> {
        // For z = x * y, dz/dx = y, dz/dy = x
        let dx = grad.mul(inputs[1]).expect("Failed to compute gradient for x");
        let dy = grad.mul(inputs[0]).expect("Failed to compute gradient for y");
        vec![dx, dy]
    }
    
    fn input_names(&self) -> Vec<&'static str> {
        vec!["x", "y"]
    }
}

/// Matrix multiplication operation.
#[derive(Debug)]
pub struct MatMulOp;

impl DifferentiableOp for MatMulOp {
    fn forward(&self, inputs: &[&BaseTensor]) -> Result<BaseTensor, AutodiffError> {
        if inputs.len() != 2 {
            return Err(AutodiffError::InvalidInput(
                "Matrix multiplication requires exactly 2 inputs".to_string(),
            ));
        }
        
        inputs[0]
            .matmul(inputs[1])
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))
    }
    
    fn backward(&self, grad: &BaseTensor, inputs: &[&BaseTensor], _output: &BaseTensor) -> Vec<BaseTensor> {
        // For C = A @ B, dA = dC @ B^T, dB = A^T @ dC
        let a = inputs[0];
        let b = inputs[1];
        
        // Compute dA = dC @ B^T
        let b_t = b.transpose().expect("Failed to transpose B");
        let da = grad.matmul(&b_t).expect("Failed to compute dA");
        
        // Compute dB = A^T @ dC
        let a_t = a.transpose().expect("Failed to transpose A");
        let db = a_t.matmul(grad).expect("Failed to compute dB");
        
        vec![da, db]
    }
    
    fn input_names(&self) -> Vec<&'static str> {
        vec!["A", "B"]
    }
}

/// ReLU activation function.
#[derive(Debug)]
pub struct ReLUOp;

impl DifferentiableOp for ReLUOp {
    fn forward(&self, inputs: &[&BaseTensor]) -> Result<BaseTensor, AutodiffError> {
        if inputs.len() != 1 {
            return Err(AutodiffError::InvalidInput(
                "ReLU operation requires exactly 1 input".to_string(),
            ));
        }
        
        inputs[0]
            .relu()
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))
    }
    
    fn backward(&self, grad: &BaseTensor, inputs: &[&BaseTensor], output: &BaseTensor) -> Vec<BaseTensor> {
        // For y = relu(x), dy/dx = 1 if x > 0 else 0
        // We can compute this as (x > 0) * grad
        let mask = inputs[0]
            .gt_scalar(0.0)
            .expect("Failed to compute ReLU mask")
            .to_dtype::<f32>()
            .expect("Failed to convert mask to f32");
            
        let grad_input = grad
            .mul(&mask)
            .expect("Failed to compute ReLU gradient");
            
        vec![grad_input]
    }
    
    fn input_names(&self) -> Vec<&'static str> {
        vec!["x"]
    }
}

/// Softmax operation.
#[derive(Debug)]
pub struct SoftmaxOp {
    axis: isize,
}

impl SoftmaxOp {
    /// Creates a new softmax operation along the given axis.
    pub fn new(axis: isize) -> Self {
        Self { axis }
    }
}

impl DifferentiableOp for SoftmaxOp {
    fn forward(&self, inputs: &[&BaseTensor]) -> Result<BaseTensor, AutodiffError> {
        if inputs.len() != 1 {
            return Err(AutodiffError::InvalidInput(
                "Softmax operation requires exactly 1 input".to_string(),
            ));
        }
        
        inputs[0]
            .softmax(self.axis)
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))
    }
    
    fn backward(&self, grad: &BaseTensor, inputs: &[&BaseTensor], output: &BaseTensor) -> Vec<BaseTensor> {
        // For y = softmax(x), the gradient is:
        // dy/dx = diag(y) - y @ y^T
        // But we can compute it more efficiently using the output
        let y = output;
        
        // Compute (grad * y).sum(axis, keepdim=True)
        let grad_y = grad.mul(y).expect("Failed to compute grad * y");
        let sum_grad = grad_y
            .sum(Some(&[self.axis as usize]), true)
            .expect("Failed to sum gradients");
        
        // Compute grad - sum_grad * y
        let term = sum_grad.mul(y).expect("Failed to compute sum_grad * y");
        let result = grad.sub(&term).expect("Failed to compute softmax gradient");
        
        vec![result]
    }
    
    fn input_names(&self) -> Vec<&'static str> {
        vec!["x"]
    }
}

/// Cross-entropy loss operation.
#[derive(Debug)]
pub struct CrossEntropyLossOp {
    reduction: String,
}

impl CrossEntropyLossOp {
    /// Creates a new cross-entropy loss operation.
    pub fn new(reduction: &str) -> Self {
        Self {
            reduction: reduction.to_string(),
        }
    }
}

impl DifferentiableOp for CrossEntropyLossOp {
    fn forward(&self, inputs: &[&BaseTensor]) -> Result<BaseTensor, AutodiffError> {
        if inputs.len() != 2 {
            return Err(AutodiffError::InvalidInput(
                "Cross-entropy loss requires exactly 2 inputs".to_string(),
            ));
        }
        
        let logits = inputs[0];
        let targets = inputs[1];
        
        // Apply softmax to logits
        let probs = logits
            .softmax(-1)
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))?;
            
        // Compute cross-entropy: -sum(targets * log(probs)) / N
        let log_probs = probs
            .log()
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))?;
            
        let loss_elements = targets
            .mul(&log_probs)
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))?
            .sum(Some(&[1]), false)
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))?
            .neg()
            .map_err(|e| AutodiffError::OperationFailed(e.to_string()))?;
            
        // Apply reduction
        match self.reduction.as_str() {
            "none" => Ok(loss_elements),
            "mean" => {
                let n = loss_elements.numel() as f32;
                loss_elements
                    .div_scalar(n)
                    .map_err(|e| AutodiffError::OperationFailed(e.to_string()))
            }
            "sum" => loss_elements
                .sum(None, false)
                .map_err(|e| AutodiffError::OperationFailed(e.to_string())),
            _ => Err(AutodiffError::InvalidInput(
                "Invalid reduction type. Must be 'none', 'mean', or 'sum'".to_string(),
            )),
        }
    }
    
    fn backward(&self, grad: &BaseTensor, inputs: &[&BaseTensor], _output: &BaseTensor) -> Vec<BaseTensor> {
        let logits = inputs[0];
        let targets = inputs[1];
        
        // Compute softmax probabilities
        let probs = logits
            .softmax(-1)
            .expect("Failed to compute softmax in backward pass");
            
        // Compute gradient: (probs - targets) * grad
        let grad_input = probs
            .sub(targets)
            .expect("Failed to compute gradient in cross-entropy backward pass");
            
        // Apply reduction scaling
        let grad_input = match self.reduction.as_str() {
            "none" => grad_input,
            "mean" => {
                let n = logits.shape()[0] as f32;
                grad_input
                    .div_scalar(n)
                    .expect("Failed to scale gradient by batch size")
            }
            "sum" => grad_input,
            _ => unreachable!("Invalid reduction type"),
        };
        
        // Multiply by incoming gradient
        let grad_input = grad_input
            .mul(grad)
            .expect("Failed to apply incoming gradient");
            
        vec![grad_input, BaseTensor::zeros_like(targets).expect("Failed to create zeros tensor")]
    }
    
    fn input_names(&self) -> Vec<&'static str> {
        vec!["logits", "targets"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_add_op() {
        let op = AddOp;
        
        // Forward pass
        let a = BaseTensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = BaseTensor::from_slice(&[4.0, 5.0, 6.0], &[3]).unwrap();
        let c = op.forward(&[&a, &b]).unwrap();
        
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);
        
        // Backward pass
        let grad = BaseTensor::ones(&[3], crate::DType::F32).unwrap();
        let grads = op.backward(&grad, &[&a, &b], &c);
        
        assert_eq!(grads[0].to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0]);
        assert_eq!(grads[1].to_vec::<f32>().unwrap(), vec![1.0, 1.0, 1.0]);
    }
    
    #[test]
    fn test_matmul_op() {
        let op = MatMulOp;
        
        // Forward pass
        let a = BaseTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = BaseTensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = op.forward(&[&a, &b]).unwrap();
        
        assert_eq!(c.shape(), &[2, 2]);
        assert_relative_eq!(c.to_vec::<f32>().unwrap()[0], 19.0, epsilon = 1e-5);
        assert_relative_eq!(c.to_vec::<f32>().unwrap()[3], 50.0, epsilon = 1e-5);
        
        // Backward pass
        let grad = BaseTensor::ones(&[2, 2], crate::DType::F32).unwrap();
        let grads = op.backward(&grad, &[&a, &b], &c);
        
        assert_eq!(grads[0].shape(), &[2, 2]);
        assert_eq!(grads[1].shape(), &[2, 2]);
        
        // dA = dC @ B^T
        assert_relative_eq!(grads[0].to_vec::<f32>().unwrap()[0], 11.0, epsilon = 1e-5);
        assert_relative_eq!(grads[0].to_vec::<f32>().unwrap()[3], 15.0, epsilon = 1e-5);
        
        // dB = A^T @ dC
        assert_relative_eq!(grads[1].to_vec::<f32>().unwrap()[0], 5.0, epsilon = 1e-5);
        assert_relative_eq!(grads[1].to_vec::<f32>().unwrap()[3], 7.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_relu_op() {
        let op = ReLUOp;
        
        // Forward pass
        let x = BaseTensor::from_slice(&[-1.0, 0.0, 2.0, -3.0, 4.0], &[5]).unwrap();
        let y = op.forward(&[&x]).unwrap();
        
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 2.0, 0.0, 4.0]);
        
        // Backward pass
        let grad = BaseTensor::ones(&[5], crate::DType::F32).unwrap();
        let grads = op.backward(&grad, &[&x], &y);
        
        assert_eq!(grads[0].to_vec::<f32>().unwrap(), vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    }
    
    #[test]
    fn test_softmax_op() {
        let op = SoftmaxOp { axis: -1 };
        
        // Forward pass
        let x = BaseTensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let y = op.forward(&[&x]).unwrap();
        
        let y_vec = y.to_vec::<f32>().unwrap();
        let sum: f32 = y_vec.iter().sum();
        
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
        assert_relative_eq!(y_vec[0], 0.0900306, epsilon = 1e-5);
        assert_relative_eq!(y_vec[1], 0.244728, epsilon = 1e-5);
        assert_relative_eq!(y_vec[2], 0.665241, epsilon = 1e-5);
        
        // Backward pass
        let grad = BaseTensor::ones(&[3], crate::DType::F32).unwrap();
        let grads = op.backward(&grad, &[&x], &y);
        
        // The gradient should be close to zero because the output is normalized
        let grad_vec = grads[0].to_vec::<f32>().unwrap();
        assert_relative_eq!(grad_vec[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(grad_vec[1], 0.0, epsilon = 1e-5);
        assert_relative_eq!(grad_vec[2], 0.0, epsilon = 1e-5);
    }
}
