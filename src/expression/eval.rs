//! Evaluation of expression graphs.
//!
//! This module provides functionality for evaluating expression graphs
//! and computing their gradients.

use std::collections::HashMap;
use std::sync::Arc;
use crate::error::Result;
use crate::tensor::Tensor;
use super::Expr;

/// A trait for evaluating expression graphs.
pub trait Evaluate {
    /// Evaluates the expression and returns the result as a tensor.
    fn eval(&self, inputs: &[&Tensor]) -> Result<Tensor>;
}

/// An evaluator for expression graphs.
#[derive(Default)]
pub struct Evaluator {
    cache: HashMap<usize, crate::tensor::Tensor>,
}

impl Evaluator {
    /// Creates a new evaluator.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }
    
    /// Evaluates an expression node.
    pub fn eval(&mut self, expr: &Expr) -> Result<crate::tensor::Tensor> {
        let expr_ptr = Arc::as_ptr(expr) as *const () as usize;
        
        // Check if the result is already in the cache
        if let Some(result) = self.cache.get(&expr_ptr) {
            return Ok(result.clone());
        }
        
        // Evaluate child nodes
        let child_results: Vec<crate::tensor::Tensor> = expr.children()
            .iter()
            .map(|child| self.eval(&child))
            .collect::<Result<_>>()?;
        
        // Convert child results to references for the eval method
        let child_refs: Vec<&crate::tensor::Tensor> = child_results.iter().collect();
        
        // Evaluate the current node
        let result = expr.eval(&child_refs)?;
        
        // Cache the result
        self.cache.insert(expr_ptr, result.clone());
        
        Ok(result)
    }
    
    /// Clears the evaluation cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Evaluate for Expr {
    fn eval(&self, inputs: &[&crate::tensor::Tensor]) -> Result<crate::tensor::Tensor> {
        let mut evaluator = Evaluator::new();
        evaluator.eval(self)
    }
}

/// A trait for computing gradients of expressions.
pub trait Differentiate {
    /// Computes the gradient of the expression with respect to the given inputs.
    fn grad(&self, inputs: &[&crate::tensor::Tensor], grad_output: &crate::tensor::Tensor) -> Result<Vec<crate::tensor::Tensor>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::expression::nodes::*;
    use crate::expression::ExprBuilder;
    
    #[test]
    fn test_evaluator() {
        // Create input tensors
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        
        // Build expression graph
        let a_expr = Arc::new(InputNode::new(a.shape().to_vec())) as Expr;
        let b_expr = Arc::new(InputNode::new(b.shape().to_vec())) as Expr;
        
        let add_expr = a_expr.add(b_expr);
        let sigmoid_expr = add_expr.sigmoid();
        
        // Evaluate the expression
        let mut evaluator = Evaluator::new();
        let result = evaluator.eval(&sigmoid_expr).unwrap();
        
        // Verify the result
        assert_eq!(result.shape(), &[3]);
        assert!((result.get(&[0]).unwrap() - 0.993307).abs() < 1e-6);
        assert!((result.get(&[1]).unwrap() - 0.999089).abs() < 1e-6);
        assert!((result.get(&[2]).unwrap() - 0.999877).abs() < 1e-6);
    }
    
    #[test]
    fn test_evaluator_with_cache() {
        // Create input tensor
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        
        // Build expression graph with shared sub-expressions
        let a_expr = Arc::new(InputNode::new(a.shape().to_vec())) as Expr;
        let square_expr = a_expr.clone().unary(|x| x * x, "Square");
        
        // This expression uses square_expr twice
        let expr = square_expr.clone().add(square_expr);
        
        // Evaluate the expression
        let mut evaluator = Evaluator::new();
        
        // First evaluation - should compute the square once
        let result1 = evaluator.eval(&expr).unwrap();
        
        // Clear the cache and evaluate again
        evaluator.clear_cache();
        let result2 = evaluator.eval(&expr).unwrap();
        
        // Results should be the same
        assert_eq!(result1.data(), result2.data());
    }
}
