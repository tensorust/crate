//! Operation fusion for expression graphs.
//!
//! This module provides functionality for optimizing expression graphs
//! by fusing multiple operations into single operations when possible.

use std::sync::Arc;
use crate::error::Result;
use super::{Expr, Evaluate, nodes::*};
use crate::expression::Expression as ExprNode;

/// A trait for optimizing expression graphs.
pub trait Optimize {
    /// Optimizes the expression graph.
    fn optimize(&self) -> Expr;
}

/// A fusion optimizer for expression graphs.
#[derive(Default)]
pub struct FusionOptimizer {
    // Cache for memoization of optimized subexpressions
    cache: std::collections::HashMap<usize, Expr>,
}

impl FusionOptimizer {
    /// Creates a new fusion optimizer.
    pub fn new() -> Self {
        Self {
            cache: std::collections::HashMap::new(),
        }
    }
    
    /// Optimizes an expression by applying fusion rules.
    pub fn optimize(&mut self, expr: &Expr) -> Expr {
        // Get a unique identifier for this expression
        let expr_ptr = expr.as_ref() as *const dyn ExprNode as *const () as usize;
        
        // Check if we've already optimized this expression
        if let Some(optimized) = self.cache.get(&expr_ptr) {
            return optimized.clone();
        }
        
        // First, optimize all child expressions
        let optimized_children: Vec<Expr> = expr.children()
            .iter()
            .map(|child| self.optimize(&Arc::from(*child)))
            .collect();
        
        // Create a new expression with optimized children
        let optimized_expr = self.apply_fusion_rules(expr, &optimized_children);
        
        // Cache the result
        self.cache.insert(expr_ptr, optimized_expr.clone());
        
        optimized_expr
    }
    
    /// Applies fusion rules to an expression with its optimized children.
    fn apply_fusion_rules(&self, expr: &Expr, children: &[Expr]) -> Expr {
        // This is a simplified version that just rebuilds the expression with optimized children
        // A full implementation would apply various fusion patterns here
        
        // For now, we'll just rebuild the expression with optimized children
        // In a real implementation, we would check for fusion patterns like:
        // - Consecutive element-wise operations (e.g., relu(sigmoid(x)) -> fused_op(x))
        // - Scale + bias fusion
        // - Batch norm fusion
        // etc.
        
        // Rebuild the expression with optimized children
        match expr.op_type() {
            "Add" => {
                if children.len() == 2 {
                    children[0].clone().add(children[1].clone())
                } else {
                    expr.clone()
                }
            }
            "Mul" => {
                if children.len() == 2 {
                    children[0].clone().mul(children[1].clone())
                } else {
                    expr.clone()
                }
            }
            "ReLU" => {
                if !children.is_empty() {
                    children[0].clone().relu()
                } else {
                    expr.clone()
                }
            }
            "Sigmoid" => {
                if !children.is_empty() {
                    children[0].clone().sigmoid()
                } else {
                    expr.clone()
                }
            }
            _ => expr.clone(),
        }
    }
    
    /// Clears the optimization cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Optimize for Expr {
    fn optimize(&self) -> Expr {
        let mut optimizer = FusionOptimizer::new();
        optimizer.optimize(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use super::super::nodes::*;
    
    #[test]
    fn test_fusion_optimizer() {
        // Create input tensor
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        
        // Build expression graph with multiple operations
        let a_expr = Arc::new(InputNode::new(a.shape().to_vec())) as Expr;
        
        // Create an expression with multiple operations
        let expr = a_expr
            .clone()
            .sigmoid()
            .relu()
            .add(a_expr.clone());
        
        // Optimize the expression
        let optimized = expr.optimize();
        
        // Verify the optimized expression has the same output as the original
        let original_result = expr.eval(&[&a]).unwrap();
        let optimized_result = optimized.eval(&[&a]).unwrap();
        
        assert_eq!(original_result.shape(), optimized_result.shape());
        for i in 0..original_result.size() {
            assert!(
                (original_result.data()[i] - optimized_result.data()[i]).abs() < 1e-6,
                "Results differ at index {}",
                i
            );
        }
    }
    
    #[test]
    fn test_fusion_with_shared_subexpressions() {
        // Create input tensor
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        
        // Build expression graph with shared sub-expressions
        let a_expr = Arc::new(InputNode::new(a.shape().to_vec())) as Expr;
        let sigmoid_expr = a_expr.clone().sigmoid();
        
        // This expression uses sigmoid_expr twice
        let expr = sigmoid_expr.clone().add(sigmoid_expr);
        
        // Optimize the expression
        let optimized = expr.optimize();
        
        // Verify the optimized expression has the same output as the original
        let original_result = expr.eval(&[&a]).unwrap();
        let optimized_result = optimized.eval(&[&a]).unwrap();
        
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
