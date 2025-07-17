use crate::{
    dimension::{Dimension, DynamicDim},
    error::{Result, TensorustError},
    storage::Storage,
    tensor::Tensor,
};
use std::any::Any;
use std::fmt;
use std::sync::{Arc, RwLock};

/// The computation graph that tracks operations
#[derive(Debug, Default)]
pub struct ExpressionGraph {
    nodes: RwLock<Vec<Arc<dyn Any + Send + Sync>>>,
}

impl ExpressionGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(Vec::new()),
        }
    }

    /// Add a node to the computation graph
    pub fn add_node<T, D, S>(&self, node: Node<T, D, S>) -> Arc<Node<T, D, S>>
    where
        T: Clone + Default + Send + Sync + 'static,
        D: Dimension + 'static,
        S: Storage<T> + 'static,
    {
        let node = Arc::new(node);
        self.nodes.write().unwrap().push(node.clone() as Arc<dyn Any + Send + Sync>);
        node
    }

    /// Clear the computation graph
    pub fn clear(&self) {
        self.nodes.write().unwrap().clear();
    }
}

/// A variable node that can hold a value
#[derive(Debug)]
pub struct Variable<T, D, S = crate::storage::CpuStorage<T>>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    value: Tensor<T, D, S>,
    requires_grad: bool,
}

impl<T, D, S> Variable<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    /// Create a new variable
    pub fn new(value: Tensor<T, D, S>) -> Self {
        Self {
            value,
            requires_grad: false,
        }
    }

    /// Set whether this variable requires gradient computation
    pub fn requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }
}

impl<T, D, S> ExpressionNode<T, D, S> for Variable<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    fn evaluate(&self) -> Result<Tensor<T, D, S>> {
        Ok(self.value.clone())
    }

    fn shape(&self) -> &D::Shape {
        self.value.shape()
    }

    fn grad(&self, _grad: Tensor<T, D, S>) -> Result<()> {
        // Variables are leaf nodes - they don't propagate gradients further
        Ok(())
    }

    fn children(&self) -> Vec<Box<dyn Any>> {
        Vec::new()
    }
}

/// A binary operation node
#[derive(Debug)]
pub struct BinaryOp<L, R, F, T, D, S = crate::storage::CpuStorage<T>>
where
    L: ExpressionNode<T, D, S>,
    R: ExpressionNode<T, D, S>,
    F: Fn(&Tensor<T, D, S>, &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> + Send + Sync + 'static,
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    left: Arc<Node<T, D, S>>,
    right: Arc<Node<T, D, S>>,
    op: F,
    _marker: std::marker::PhantomData<(L, R, T, D, S)>,
}

impl<L, R, F, T, D, S> BinaryOp<L, R, F, T, D, S>
where
    L: ExpressionNode<T, D, S>,
    R: ExpressionNode<T, D, S>,
    F: Fn(&Tensor<T, D, S>, &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> + Send + Sync + 'static,
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    pub fn new(left: Arc<Node<T, D, S>>, right: Arc<Node<T, D, S>>, op: F) -> Self {
        Self {
            left,
            right,
            op,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<L, R, F, T, D, S> ExpressionNode<T, D, S> for BinaryOp<L, R, F, T, D, S>
where
    L: ExpressionNode<T, D, S>,
    R: ExpressionNode<T, D, S>,
    F: Fn(&Tensor<T, D, S>, &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> + Send + Sync + 'static,
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    fn evaluate(&self) -> Result<Tensor<T, D, S>> {
        let left = self.left.evaluate()?;
        let right = self.right.evaluate()?;
        (self.op)(&left, &right)
    }

    fn shape(&self) -> &D::Shape {
        // For simplicity, assume output shape matches left operand
        // In a real implementation, you'd need to handle broadcasting
        self.left.shape()
    }

    fn grad(&self, grad: Tensor<T, D, S>) -> Result<()> {
        // In a real implementation, you would:
        // 1. Compute the gradient with respect to left and right operands
        // 2. Propagate gradients to child nodes
        // This is a simplified placeholder
        self.left.backward(Some(grad.clone()))?;
        self.right.backward(Some(grad))?;
        Ok(())
    }

    fn children(&self) -> Vec<Box<dyn Any>> {
        vec![
            Box::new(self.left.clone()) as Box<dyn Any>,
            Box::new(self.right.clone()) as Box<dyn Any>,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::CpuStorage;

    #[test]
    fn test_variable_evaluation() {
        let graph = ExpressionGraph::new();
        let tensor = Tensor::<f32, DynamicDim, _>::from(vec![1.0, 2.0, 3.0]);
        let var = Variable::new(tensor);
        let node = Node::new(var);
        let result = node.evaluate().unwrap();
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_binary_op() {
        let graph = ExpressionGraph::new();
        let a = graph.add_node(Node::new(Variable::new(Tensor::from(vec![1.0, 2.0]))));
        let b = graph.add_node(Node::new(Variable::new(Tensor::from(vec![3.0, 4.0]))));
        
        let add = BinaryOp::new(
            a.clone(),
            b.clone(),
            |a, b| Ok(Tensor::from(a.to_vec().iter().zip(b.to_vec()).map(|(a, b)| a + b).collect::<Vec<_>>())),
        );
        
        let add_node = graph.add_node(Node::new(add));
        let result = add_node.evaluate().unwrap();
        assert_eq!(result.to_vec(), vec![4.0, 6.0]);
    }
}

/// Trait for expression nodes in the computation graph
pub trait ExpressionNode<T, D, S>: fmt::Debug + Send + Sync + 'static
where
    T: Clone + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    /// Evaluate the expression and return a tensor
    fn evaluate(&self) -> Result<Tensor<T, D, S>>;
    
    /// Get the shape of the resulting tensor
    fn shape(&self) -> &D::Shape;
    
    /// Get the gradient with respect to this node
    fn grad(&self, grad: Tensor<T, D, S>) -> Result<()>;
    
    /// Get children nodes for gradient computation
    fn children(&self) -> Vec<Box<dyn Any>>;
}

/// A node in the computation graph
#[derive(Debug)]
pub struct Node<T, D, S = crate::storage::CpuStorage<T>>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    /// The actual expression node
    node: Box<dyn ExpressionNode<T, D, S>>,
    /// Cached value (for memoization)
    cached: RwLock<Option<Tensor<T, D, S>>>,
    /// Gradient accumulator
    grad: RwLock<Option<Tensor<T, D, S>>>>,
}

impl<T, D, S> Node<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension + 'static,
    S: Storage<T> + 'static,
{
    /// Create a new node from an expression
    pub fn new<N>(node: N) -> Self
    where
        N: ExpressionNode<T, D, S> + 'static,
    {
        Self {
            node: Box::new(node),
            cached: RwLock::new(None),
            grad: RwLock::new(None),
        }
    }

    /// Evaluate the node, using cached value if available
    pub fn evaluate(&self) -> Result<Tensor<T, D, S>> {
        {
            let cached = self.cached.read().unwrap();
            if let Some(tensor) = &*cached {
                return Ok(tensor.clone());
            }
        }

        let result = self.node.evaluate()?;
        *self.cached.write().unwrap() = Some(result.clone());
        Ok(result)
    }

    /// Get the shape of the resulting tensor
    pub fn shape(&self) -> &D::Shape {
        self.node.shape()
    }

    /// Backward pass through the computation graph
    pub fn backward(&self, grad: Option<Tensor<T, D, S>>) -> Result<()> {
        let grad = match grad {
            Some(g) => g,
            None => Tensor::ones(self.shape().clone())?,
        };

        // Accumulate gradient
        {
            let mut current_grad = self.grad.write().unwrap();
            if let Some(ref mut g) = *current_grad {
                // TODO: Implement proper gradient accumulation
                *g = &*g + &grad;
            } else {
                *current_grad = Some(grad.clone());
            }
        }

        // Propagate gradient to children
        self.node.grad(grad)?;

        Ok(())
    }
}
