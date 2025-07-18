//! Differentiable tensor type for automatic differentiation.

use super::{
    graph::{Graph, Node, NodeId},
    ops::{Add, Mul},
};
use crate::tensor::{DataType, Tensor, TensorLike};
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
pub struct ADTensor<T: DataType> {
    pub tensor: Tensor<T>,
    pub graph: Rc<RefCell<Graph<T>>>,
    pub node_id: NodeId,
}

impl<T: DataType> ADTensor<T> {
    /// Creates a new `ADTensor` from a `Tensor`.
    pub fn new(tensor: Tensor<T>, graph: Rc<RefCell<Graph<T>>>) -> Self {
        let node_id = graph.borrow_mut().add_variable(tensor.clone());
        Self {
            tensor,
            graph,
            node_id,
        }
    }

    /// Returns the gradient of the tensor.
    pub fn grad(&self) -> Ref<Tensor<T>> {
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

impl<T: DataType> TensorLike<T> for ADTensor<T> {
    fn a(&self) -> &Tensor<T> {
        &self.tensor
    }
}

impl<T: DataType> TensorAdd for ADTensor<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let op = Add::new(self.node_id, rhs.node_id);
        let result = self.tensor.add(&rhs.tensor);
        let node_id = self.graph.borrow_mut().add_node(Box::new(op));
        Self {
            tensor: result,
            graph: self.graph.clone(),
            node_id,
        }
    }
}

impl<T: DataType> TensorMul for ADTensor<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let op = Mul::new(self.node_id, rhs.node_id);
        let result = self.tensor.mul(&rhs.tensor);
        let node_id = self.graph.borrow_mut().add_node(Box::new(op));
        Self {
            tensor: result,
            graph: self.graph.clone(),
            node_id,
        }
    }
}
