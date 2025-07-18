//! Expression graph for lazy evaluation of tensor operations.
//!
//! This module provides a system for building and evaluating computation graphs
//! that represent tensor operations. The expression graph enables lazy evaluation
//! and operation fusion for better performance.

use crate::{
    error::Result,
    tensor::{Tensor, TensorLike},
};
use std::{fmt::Debug, sync::Arc};

pub mod eval;
pub mod fusion;
pub mod nodes;

pub use eval::{Evaluator, Gradient};
pub use fusion::FusionOptimizer;
pub use nodes::{BinaryOp, Node, UnaryOp};

/// A trait for nodes in the expression graph.
pub trait Expression: Debug + Send + Sync + 'static {
    /// Returns the shape of the output tensor.
    fn shape(&self) -> &[usize];

    /// Returns the number of dimensions of the output tensor.
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Returns the total number of elements in the output tensor.
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Evaluates the expression node.
    fn eval(&self) -> Result<Tensor>;

    /// Computes the gradient of the expression node.
    fn grad(&self, grad: &Tensor) -> Result<Gradient>;
}

/// A reference-counted expression node.
pub type Expr = Arc<dyn ExprNode>;

/// A trait for types that can be converted to expressions.
pub trait IntoExpr {
    /// Converts the value into an expression.
    fn into_expr(self) -> Expr;
}

impl IntoExpr for Expr {
    fn into_expr(self) -> Expr {
        self
    }
}

impl<T: ExprNode + 'static> IntoExpr for T {
    fn into_expr(self) -> Expr {
        Arc::new(self)
    }
}

/// A trait for building expression graphs.
pub trait ExprBuilder: Sized {
    /// Creates a new expression that applies a unary operation.
    fn unary<F>(self, op: F, op_type: &'static str) -> Expr
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static;
    
    /// Creates a new expression that applies a binary operation.
    fn binary<F>(self, rhs: Expr, op: F, op_type: &'static str) -> Expr
    where
        F: Fn(f32, f32) -> f32 + Send + Sync + 'static;
    
    /// Creates a new expression that adds two tensors.
    fn add(self, rhs: Expr) -> Expr {
        self.binary(rhs, |a, b| a + b, "Add")
    }
    
    /// Creates a new expression that subtracts two tensors.
    fn sub(self, rhs: Expr) -> Expr {
        self.binary(rhs, |a, b| a - b, "Sub")
    }
    
    /// Creates a new expression that multiplies two tensors.
    fn mul(self, rhs: Expr) -> Expr {
        self.binary(rhs, |a, b| a * b, "Mul")
    }
    
    /// Creates a new expression that divides two tensors.
    fn div(self, rhs: Expr) -> Expr {
        self.binary(rhs, |a, b| a / b, "Div")
    }
    
    /// Creates a new expression that applies the ReLU activation function.
    fn relu(self) -> Expr {
        self.unary(|x| if x > 0.0 { x } else { 0.0 }, "ReLU")
    }
    
    /// Creates a new expression that applies the sigmoid activation function.
    fn sigmoid(self) -> Expr {
        self.unary(|x| 1.0 / (1.0 + (-x).exp()), "Sigmoid")
    }
    
    /// Creates a new expression that computes the mean of the tensor.
    fn mean(self, axis: Option<usize>, keepdims: bool) -> Expr {
        let node = MeanNode::new(self.into_expr(), axis, keepdims);
        Arc::new(node)
    }
    
    /// Creates a new expression that computes the sum of the tensor.
    fn sum(self, axis: Option<usize>, keepdims: bool) -> Expr {
        let node = SumNode::new(self.into_expr(), axis, keepdims);
        Arc::new(node)
    }
    
    /// Creates a new expression that computes the maximum value of the tensor.
    fn max(self, axis: Option<usize>, keepdims: bool) -> Expr {
        let node = MaxNode::new(self.into_expr(), axis, keepdims);
        Arc::new(node)
    }
    
    /// Creates a new expression that computes the minimum value of the tensor.
    fn min(self, axis: Option<usize>, keepdims: bool) -> Expr {
        let node = MinNode::new(self.into_expr(), axis, keepdims);
        Arc::new(node)
    }
    
    /// Creates a new expression that reshapes the tensor.
    fn reshape(self, shape: Vec<usize>) -> Expr {
        let node = ReshapeNode::new(self.into_expr(), shape);
        Arc::new(node)
    }
    
    /// Creates a new expression that transposes the tensor.
    fn transpose(self, axes: Option<Vec<usize>>) -> Expr {
        let node = TransposeNode::new(self.into_expr(), axes);
        Arc::new(node)
    }
    
    /// Creates a new expression that slices the tensor.
    fn slice(self, slices: Vec<(Option<usize>, Option<usize>, usize)>) -> Expr {
        let node = SliceNode::new(self.into_expr(), slices);
        Arc::new(node)
    }
    
    /// Creates a new expression that broadcasts the tensor to a new shape.
    fn broadcast_to(self, shape: Vec<usize>) -> Expr {
        let node = BroadcastNode::new(self.into_expr(), shape);
        Arc::new(node)
    }
}

impl ExprBuilder for Expr {
    fn unary<F>(self, op: F, op_type: &'static str) -> Expr
    where
        F: Fn(f32) -> f32 + Send + Sync + 'static,
    {
        let node = UnaryNode::new(self, op, op_type);
        Arc::new(node)
    }
    
    fn binary<F>(self, rhs: Expr, op: F, op_type: &'static str) -> Expr
    where
        F: Fn(f32, f32) -> f32 + Send + Sync + 'static,
    {
        let node = BinaryNode::new(self, rhs, op, op_type);
        Arc::new(node)
    }
}

/// A trait for evaluating expression graphs.
pub trait Evaluate {
    /// Evaluates the expression and returns the result as a tensor.
    fn eval(&self) -> Result<Tensor>;
}

impl Evaluate for Expr {
    fn eval(&self) -> Result<Tensor> {
        let evaluator = Evaluator::new();
        evaluator.eval(self)
    }
}

/// A trait for optimizing expression graphs.
pub trait Optimize {
    /// Optimizes the expression graph.
    fn optimize(&self) -> Expr;
}

impl Optimize for Expr {
    fn optimize(&self) -> Expr {
        let optimizer = FusionOptimizer::new();
        optimizer.optimize(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    
    #[test]
    fn test_basic_expression() {
        // Create input tensors
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        
        // Build expression graph
        let a_expr = Arc::new(InputNode::new(a.shape().to_vec())) as Expr;
        let b_expr = Arc::new(InputNode::new(b.shape().to_vec())) as Expr;
        
        let expr = a_expr.add(b_expr).sigmoid();
        
        // Evaluate expression
        let result = expr.eval().unwrap();
        
        // Verify result
        assert_eq!(result.shape(), &[3]);
        assert!((result.get(&[0]).unwrap() - 0.993307).abs() < 1e-6);
        assert!((result.get(&[1]).unwrap() - 0.999089).abs() < 1e-6);
        assert!((result.get(&[2]).unwrap() - 0.999877).abs() < 1e-6);
    }
    
    #[test]
    fn test_expression_with_reduction() {
        // Create input tensor
        let a = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ).unwrap();
        
        // Build expression graph
        let a_expr = Arc::new(InputNode::new(a.shape().to_vec())) as Expr;
        
        // Compute mean along axis 0
        let mean_expr = a_expr.mean(Some(0), false);
        
        // Evaluate expression
        let result = mean_expr.eval().unwrap();
        
        // Verify result
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.get(&[0]).unwrap(), &2.5);
        assert_eq!(result.get(&[1]).unwrap(), &3.5);
        assert_eq!(result.get(&[2]).unwrap(), &4.5);
    }
    
    #[test]
    fn test_expression_optimization() {
        // Create input tensors
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        
        // Build expression graph
        let a_expr = Arc::new(InputNode::new(a.shape().to_vec())) as Expr;
        let b_expr = Arc::new(InputNode::new(b.shape().to_vec())) as Expr;
        
        // Build expression with multiple operations
        let expr = a_expr
            .add(b_expr.clone())
            .mul(b_expr)
            .sigmoid();
        
        // Optimize expression
        let optimized = expr.optimize();
        
        // Evaluate both original and optimized expressions
        let original_result = expr.eval().unwrap();
        let optimized_result = optimized.eval().unwrap();
        
        // Verify results are the same
        assert_eq!(original_result.shape(), optimized_result.shape());
        for i in 0..original_result.size() {
            assert!(
                (original_result.data()[i] - optimized_result.data()[i]).abs() < 1e-6,
                "Results differ at index {}",
                i
            );
        }
    }
}
