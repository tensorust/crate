//! Differentiable tensor type for automatic differentiation.

use super::{
    graph,
    ops,
};
use crate::tensor::{Tensor};
use std::{
    cell::{Ref, RefCell},
    ops::{Add as TensorAdd, Mul as TensorMul},
    rc::Rc,
};

/// A tensor that supports automatic differentiation.
///
/// `ADTensor` wraps a `Tensor` and a `Graph` to enable automatic
/// differentiation. The `Graph` stores the computation graph, and the `ADTensor`
/// stores the `NodeId` of the tensor in the graph.
#[derive(Clone)]
pub struct ADTensor {
    pub tensor: Tensor,
    pub graph: Rc<RefCell<graph::Graph>>,
    pub node_id: graph::NodeId,
}

impl ADTensor {
    /// Creates a new `ADTensor` from a `Tensor`.
    pub fn new(tensor: Tensor, graph: Rc<RefCell<graph::Graph>>) -> Self {
        let node_id = graph.borrow_mut().add_variable(tensor.clone());
        Self {
            tensor,
            graph,
            node_id,
        }
    }

    /// Returns the gradient of the tensor.
    pub fn grad(&self) -> Ref<Tensor> {
        let graph = self.graph.borrow();
        let node = &graph.nodes[self.node_id];
        Ref::map(graph, |g| &g.nodes[node.id].grad)
    }

    /// Computes the gradient of the tensor with respect to all other tensors in
    /// the computation graph.
    pub fn backward(&self) {
        self.graph.borrow_mut().backward(self.node_id);
    }
}

impl<T> crate::tensor::TensorLike<T> for ADTensor {
    fn as_tensor(&self) -> &Tensor<T> {
        &self.tensor
    }
}

impl TensorAdd for ADTensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let op = ops::Add::new(self.node_id, rhs.node_id);
        let result = self.tensor.add(&rhs.tensor);
        let node_id = self.graph.borrow_mut().add_node(Box::new(op));
        Self {
            tensor: result,
            graph: self.graph.clone(),
            node_id,
        }
    }
}

impl TensorMul for ADTensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let op = ops::Mul::new(self.node_id, rhs.node_id);
        let result = self.tensor.mul(&rhs.tensor);
        let node_id = self.graph.borrow_mut().add_node(Box::new(op));
        Self {
            tensor: result,
            graph: self.graph.clone(),
            node_id,
        }
    }
}
