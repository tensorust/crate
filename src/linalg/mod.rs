//! Linear algebra operations for tensors.
//! This module provides various linear algebra operations that can be performed on tensors.

mod matmul;
mod transpose;
mod inverse;
mod decomposition;

pub use matmul::*;
pub use transpose::*;
pub use inverse::*;
pub use decomposition::*;

use crate::{
    dimension::Dimension,
    error::Result,
    tensor::Tensor,
    storage::Storage,
};

/// Trait for linear algebra operations on 2D tensors (matrices).
pub trait LinearAlgebra<T, S>: Sized
where
    T: Clone + Send + Sync + 'static,
    S: Storage<T>,
{
    /// Matrix multiplication.
    fn matmul(&self, rhs: &Self) -> Result<Self>;
    
    /// Matrix transpose.
    fn t(&self) -> Result<Self>;
    
    /// Matrix inverse.
    fn inv(&self) -> Result<Self>;
    
    /// Matrix determinant.
    fn det(&self) -> Result<T>;
    
    /// Singular Value Decomposition (SVD).
    fn svd(&self, full_matrices: bool) -> Result<(Self, Tensor<T, crate::dimension::DynamicDim, S>, Self)>;
    
    /// Eigenvalue decomposition.
    fn eig(&self) -> Result<(Tensor<T, crate::dimension::DynamicDim, S>, Self)>;
}

// Implementation for 2D tensors
impl<T, S> LinearAlgebra<T, S> for Tensor<T, crate::dimension::StaticDim<2>, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn matmul(&self, rhs: &Self) -> Result<Self> {
        matmul(self, rhs)
    }
    
    fn t(&self) -> Result<Self> {
        transpose(self)
    }
    
    fn inv(&self) -> Result<Self> {
        inverse(self)
    }
    
    fn det(&self) -> Result<T> {
        determinant(self)
    }
    
    fn svd(&self, full_matrices: bool) -> Result<(Self, Tensor<T, crate::dimension::DynamicDim, S>, Self)> {
        svd(self, full_matrices)
    }
    
    fn eig(&self) -> Result<(Tensor<T, crate::dimension::DynamicDim, S>, Self)> {
        eigenvalue_decomposition(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dimension::StaticDim,
        storage::CpuStorage,
        tensor,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_matmul() {
        let a = tensor!([
            [1.0, 2.0],
            [3.0, 4.0]
        ]);
        let b = tensor!([
            [5.0, 6.0],
            [7.0, 8.0]
        ]);
        let c = a.matmul(&b).unwrap();
        
        assert_eq!(c.shape(), &[2, 2]);
        assert_relative_eq!(c[[0, 0]], 19.0);
        assert_relative_eq!(c[[0, 1]], 22.0);
        assert_relative_eq!(c[[1, 0]], 43.0);
        assert_relative_eq!(c[[1, 1]], 50.0);
    }

    #[test]
    fn test_transpose() {
        let a = tensor!([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]);
        let b = a.t().unwrap();
        
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b[[0, 0]], 1.0);
        assert_eq!(b[[0, 1]], 4.0);
        assert_eq!(b[[1, 0]], 2.0);
        assert_eq!(b[[1, 1]], 5.0);
        assert_eq!(b[[2, 0]], 3.0);
        assert_eq!(b[[2, 1]], 6.0);
    }
}
