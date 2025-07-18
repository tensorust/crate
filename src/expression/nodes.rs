//! Expression nodes for the computation graph.
//!
//! This module defines the various node types that can be used to build
//! expression graphs for lazy evaluation of tensor operations.

use std::fmt;
use std::sync::Arc;
use crate::error::Result;
use crate::tensor::Tensor;
use super::Expr;

/// A node that represents an input to the computation graph.
#[derive(Debug)]
pub struct InputNode {
    shape: Vec<usize>,
}

impl InputNode {
    /// Creates a new input node with the given shape.
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl ExprNode for InputNode {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn eval(&self, _inputs: &[&Tensor]) -> Result<Tensor> {
        Err(crate::error::TensorustError::invalid_operation(
            "Cannot evaluate an input node without providing a value"
        ))
    }
    
    fn children(&self) -> Vec<&dyn ExprNode> {
        vec![]
    }
    
    fn op_type(&self) -> &'static str {
        "Input"
    }
}

/// A node that represents a unary operation.
#[derive(Debug)]
pub struct UnaryNode<F> {
    input: Expr,
    op: F,
    op_type: &'static str,
    shape: Vec<usize>,
}

impl<F> UnaryNode<F>
where
    F: Fn(f32) -> f32 + Send + Sync + 'static,
{
    /// Creates a new unary operation node.
    pub fn new(input: Expr, op: F, op_type: &'static str) -> Self {
        let shape = input.shape().to_vec();
        Self {
            input,
            op,
            op_type,
            shape,
        }
    }
}

impl<F> ExprNode for UnaryNode<F>
where
    F: Fn(f32) -> f32 + Send + Sync + 'static,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn eval(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        let input = inputs[0];
        let mut output = Tensor::zeros_like(input)?;
        
        for (i, &x) in input.data().iter().enumerate() {
            output.data_mut()[i] = (self.op)(x);
        }
        
        Ok(output)
    }
    
    fn children(&self) -> Vec<&dyn ExprNode> {
        vec![self.input.as_ref()]
    }
    
    fn op_type(&self) -> &'static str {
        self.op_type
    }
}

/// A node that represents a binary operation.
#[derive(Debug)]
pub struct BinaryNode<F> {
    left: Expr,
    right: Expr,
    op: F,
    op_type: &'static str,
    shape: Vec<usize>,
}

impl<F> BinaryNode<F>
where
    F: Fn(f32, f32) -> f32 + Send + Sync + 'static,
{
    /// Creates a new binary operation node.
    pub fn new(left: Expr, right: Expr, op: F, op_type: &'static str) -> Self {
        // For simplicity, we assume broadcasting is handled by the tensor type
        let shape = left.shape().to_vec();
        
        Self {
            left,
            right,
            op,
            op_type,
            shape,
        }
    }
}

impl<F> ExprNode for BinaryNode<F>
where
    F: Fn(f32, f32) -> f32 + Send + Sync + 'static,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn eval(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        let left = inputs[0];
        let right = inputs[1];
        
        // For simplicity, we assume the tensors are already broadcasted
        let mut output = Tensor::zeros_like(left)?;
        
        for i in 0..output.size() {
            output.data_mut()[i] = (self.op)(left.data()[i], right.data()[i]);
        }
        
        Ok(output)
    }
    
    fn children(&self) -> Vec<&dyn ExprNode> {
        vec![self.left.as_ref(), self.right.as_ref()]
    }
    
    fn op_type(&self) -> &'static str {
        self.op_type
    }
}

/// A node that represents a reduction operation.
#[derive(Debug)]
pub struct ReduceNode<F> {
    input: Expr,
    axis: Option<usize>,
    keepdims: bool,
    op: F,
    op_type: &'static str,
    shape: Vec<usize>,
}

impl<F> ReduceNode<F>
where
    F: Fn(f32, f32) -> f32 + Send + Sync + 'static,
{
    /// Creates a new reduction node.
    pub fn new(input: Expr, axis: Option<usize>, keepdims: bool, op: F, op_type: &'static str) -> Self {
        let input_shape = input.shape();
        let mut shape = input_shape.to_vec();
        
        if let Some(axis) = axis {
            if axis < shape.len() {
                if keepdims {
                    shape[axis] = 1;
                } else {
                    shape.remove(axis);
                }
            }
        } else if !keepdims {
            shape = vec![];
        }
        
        Self {
            input,
            axis,
            keepdims,
            op,
            op_type,
            shape,
        }
    }
}

impl<F> ExprNode for ReduceNode<F>
where
    F: Fn(f32, f32) -> f32 + Send + Sync + 'static,
{
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn eval(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        let input = inputs[0];
        
        match self.axis {
            Some(axis) => {
                // Reduce along a specific axis
                let mut output_shape = input.shape().to_vec();
                if self.keepdims {
                    output_shape[axis] = 1;
                } else {
                    output_shape.remove(axis);
                }
                
                let mut output = Tensor::zeros(&output_shape)?;
                
                // TODO: Implement reduction along axis
                // This is a simplified version that only works for 1D tensors
                if input.ndim() == 1 {
                    let mut result = 0.0;
                    for &x in input.data() {
                        result = (self.op)(result, x);
                    }
                    output.data_mut()[0] = result;
                } else {
                    return Err(crate::error::TensorustError::not_implemented(
                        "Reduction along axis is not yet implemented for multi-dimensional tensors"
                    ));
                }
                
                Ok(output)
            }
            None => {
                // Reduce all dimensions
                let mut result = 0.0;
                for &x in input.data() {
                    result = (self.op)(result, x);
                }
                
                if self.keepdims {
                    Tensor::from_vec(
                        vec![result],
                        vec![1; input.ndim()],
                    )
                } else {
                    Tensor::from_vec(vec![result], vec![])
                }
            }
        }
    }
    
    fn children(&self) -> Vec<&dyn ExprNode> {
        vec![self.input.as_ref()]
    }
    
    fn op_type(&self) -> &'static str {
        self.op_type
    }
}

/// A node that represents a reshape operation.
#[derive(Debug)]
pub struct ReshapeNode {
    input: Expr,
    shape: Vec<usize>,
}

impl ReshapeNode {
    /// Creates a new reshape node.
    pub fn new(input: Expr, shape: Vec<usize>) -> Self {
        Self { input, shape }
    }
}

impl ExprNode for ReshapeNode {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn eval(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        let input = inputs[0];
        input.reshape(&self.shape)
    }
    
    fn children(&self) -> Vec<&dyn ExprNode> {
        vec![self.input.as_ref()]
    }
    
    fn op_type(&self) -> &'static str {
        "Reshape"
    }
}

/// A node that represents a transpose operation.
#[derive(Debug)]
pub struct TransposeNode {
    input: Expr,
    axes: Option<Vec<usize>>,
    shape: Vec<usize>,
}

impl TransposeNode {
    /// Creates a new transpose node.
    pub fn new(input: Expr, axes: Option<Vec<usize>>) -> Self {
        let input_shape = input.shape();
        let ndim = input_shape.len();
        
        let axes = axes.unwrap_or_else(|| {
            let mut axes: Vec<usize> = (0..ndim).rev().collect();
            axes.truncate(ndim);
            axes
        });
        
        let shape: Vec<usize> = axes.iter().map(|&i| input_shape[i]).collect();
        
        Self {
            input,
            axes: Some(axes),
            shape,
        }
    }
}

impl ExprNode for TransposeNode {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn eval(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        let input = inputs[0];
        input.transpose(self.axes.as_deref())
    }
    
    fn children(&self) -> Vec<&dyn ExprNode> {
        vec![self.input.as_ref()]
    }
    
    fn op_type(&self) -> &'static str {
        "Transpose"
    }
}

/// A node that represents a slice operation.
#[derive(Debug)]
pub struct SliceNode {
    input: Expr,
    slices: Vec<(Option<usize>, Option<usize>, usize)>,
    shape: Vec<usize>,
}

impl SliceNode {
    /// Creates a new slice node.
    pub fn new(input: Expr, slices: Vec<(Option<usize>, Option<usize>, usize)>) -> Self {
        let input_shape = input.shape();
        let mut shape = input_shape.to_vec();
        
        for (i, (start, end, step)) in slices.iter().enumerate() {
            if i >= shape.len() {
                break;
            }
            
            let dim_size = shape[i];
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(dim_size);
            
            if *step == 0 {
                panic!("Step size cannot be zero");
            }
            
            let len = if *step > 0 {
                (end - start + step - 1) / step
            } else {
                (start - end + (-step) - 1) / (-step)
            };
            
            shape[i] = len;
        }
        
        Self {
            input,
            slices,
            shape,
        }
    }
}

impl ExprNode for SliceNode {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn eval(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        let input = inputs[0];
        
        // This is a simplified version that only supports basic slicing
        // A full implementation would handle all the edge cases
        let mut result = input.clone();
        
        for (i, (start, end, step)) in self.slices.iter().enumerate() {
            if i >= result.ndim() {
                break;
            }
            
            let dim_size = result.shape()[i];
            let start = start.unwrap_or(0);
            let end = end.unwrap_or(dim_size);
            let step = *step as isize;
            
            // TODO: Implement actual slicing with step
            // For now, we'll just take a simple slice without step
            result = result.slice_axis(i, start..end)?;
        }
        
        Ok(result)
    }
    
    fn children(&self) -> Vec<&dyn ExprNode> {
        vec![self.input.as_ref()]
    }
    
    fn op_type(&self) -> &'static str {
        "Slice"
    }
}

/// A node that represents a broadcast operation.
#[derive(Debug)]
pub struct BroadcastNode {
    input: Expr,
    shape: Vec<usize>,
}

impl BroadcastNode {
    /// Creates a new broadcast node.
    pub fn new(input: Expr, shape: Vec<usize>) -> Self {
        Self { input, shape }
    }
}

impl ExprNode for BroadcastNode {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn eval(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        let input = inputs[0];
        input.broadcast_to(&self.shape)
    }
    
    fn children(&self) -> Vec<&dyn ExprNode> {
        vec![self.input.as_ref()]
    }
    
    fn op_type(&self) -> &'static str {
        "Broadcast"
    }
}

// Helper functions for creating common reduction nodes

/// Creates a sum reduction node.
pub fn sum(input: Expr, axis: Option<usize>, keepdims: bool) -> Expr {
    Arc::new(ReduceNode::new(input, axis, keepdims, |a, b| a + b, "Sum"))
}

/// Creates a mean reduction node.
pub fn mean(input: Expr, axis: Option<usize>, keepdims: bool) -> Expr {
    let sum_node = Arc::new(ReduceNode::new(
        input.clone(),
        axis,
        keepdims,
        |a, b| a + b,
        "Sum",
    ));
    
    let count = if let Some(axis) = axis {
        input.shape()[axis] as f32
    } else {
        input.size() as f32
    };
    
    let scale = 1.0 / count;
    let scale_node = Arc::new(UnaryNode::new(
        sum_node,
        move |x| x * scale,
        "Scale",
    ));
    
    scale_node
}

/// Creates a max reduction node.
pub fn max(input: Expr, axis: Option<usize>, keepdims: bool) -> Expr {
    Arc::new(ReduceNode::new(
        input,
        axis,
        keepdims,
        |a, b| if a > b { a } else { b },
        "Max",
    ))
}

/// Creates a min reduction node.
pub fn min(input: Expr, axis: Option<usize>, keepdims: bool) -> Expr {
    Arc::new(ReduceNode::new(
        input,
        axis,
        keepdims,
        |a, b| if a < b { a } else { b },
        "Min",
    ))
}
