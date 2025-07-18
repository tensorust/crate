//! Matrix multiplication operations.

use crate::{
    dimension::{static_dim::StaticDim, Dimension},
    error::{Result, TensorustError},
    tensor::Tensor,
    storage::Storage,
};

/// Matrix multiplication between two 2D tensors.
pub fn matmul<T, S>(
    lhs: &Tensor<T, StaticDim<2>, S>,
    rhs: &Tensor<T, StaticDim<2>, S>,
) -> Result<Tensor<T, StaticDim<2>, S>>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    
    // Check matrix dimensions are compatible for multiplication
    if lhs_shape[1] != rhs_shape[0] {
        return Err(TensorustError::shape_mismatch(
            lhs_shape.as_ref().to_vec(),
            rhs_shape.as_ref().to_vec(),
        ));
    }
    
    let m = lhs_shape[0];
    let n = rhs_shape[1];
    let k = lhs_shape[1];
    
    let mut result_data = vec![T::default(); m * n];
    let lhs_data = lhs.storage().to_vec();
    let rhs_data = rhs.storage().to_vec();
    
    // Naive matrix multiplication (can be optimized with BLAS, etc.)
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default();
            for l in 0..k {
                let lhs_idx = i * k + l;
                let rhs_idx = l * n + j;
                sum = &sum + &(&lhs_data[lhs_idx] * &rhs_data[rhs_idx]);
            }
            result_data[i * n + j] = sum;
        }
    }
    
    let result_storage = S::from_vec(result_data);
    Tensor::new(result_storage, [m, n].into())
}

/// Batched matrix multiplication.
/// 
/// # Arguments
/// * `lhs` - A 3D tensor with shape [batch_size, m, k]
/// * `rhs` - A 3D tensor with shape [batch_size, k, n] or [1, k, n]
/// 
/// Returns a 3D tensor with shape [batch_size, m, n]
pub fn bmm<T, S>(
    lhs: &Tensor<T, StaticDim<3>, S>,
    rhs: &Tensor<T, StaticDim<3>, S>,
) -> Result<Tensor<T, StaticDim<3>, S>>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    
    // Check batch dimensions are compatible
    if lhs_shape[0] != rhs_shape[0] && rhs_shape[0] != 1 {
        return Err(TensorustError::shape_mismatch(
            lhs_shape.as_ref().to_vec(),
            rhs_shape.as_ref().to_vec(),
        ));
    }
    
    // Check matrix dimensions are compatible
    if lhs_shape[2] != rhs_shape[1] {
        return Err(TensorustError::shape_mismatch(
            lhs_shape[1..].to_vec(),
            rhs_shape[1..].to_vec(),
        ));
    }
    
    let batch_size = lhs_shape[0];
    let m = lhs_shape[1];
    let n = rhs_shape[2];
    let k = lhs_shape[2];
    
    let mut result_data = vec![T::default(); batch_size * m * n];
    let lhs_data = lhs.storage().to_vec();
    let rhs_data = rhs.storage().to_vec();
    
    let rhs_batch_stride = if rhs_shape[0] == 1 { 0 } else { k * n };
    
    for b in 0..batch_size {
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for l in 0..k {
                    let lhs_idx = b * m * k + i * k + l;
                    let rhs_idx = b * rhs_batch_stride + l * n + j;
                    sum = &sum + &(&lhs_data[lhs_idx] * &rhs_data[rhs_idx]);
                }
                result_data[b * m * n + i * n + j] = sum;
            }
        }
    }
    
    let result_storage = S::from_vec(result_data);
    Tensor::new(result_storage, [batch_size, m, n].into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dimension::StaticDim,
        storage::CpuStorage,
        tensor,
        tensorust,
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
        let c = matmul(&a, &b).unwrap();
        
        assert_eq!(c.shape(), &[2, 2]);
        assert_relative_eq!(c[[0, 0]], 19.0);
        assert_relative_eq!(c[[0, 1]], 22.0);
        assert_relative_eq!(c[[1, 0]], 43.0);
        assert_relative_eq!(c[[1, 1]], 50.0);
    }

    #[test]
    fn test_bmm() {
        let a = tensor!([
            [
                [1.0, 2.0],
                [3.0, 4.0]
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0]
            ]
        ]);
        let b = tensor!([
            [
                [9.0, 10.0],
                [11.0, 12.0]
            ],
            [
                [13.0, 14.0],
                [15.0, 16.0]
            ]
        ]);
        let c = bmm(&a, &b).unwrap();
        
        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_relative_eq!(c[[0, 0, 0]], 31.0);
        assert_relative_eq!(c[[0, 0, 1]], 34.0);
        assert_relative_eq!(c[[0, 1, 0]], 71.0);
        assert_relative_eq!(c[[0, 1, 1]], 78.0);
        assert_relative_eq!(c[[1, 0, 0]], 155.0);
        assert_relative_eq!(c[[1, 0, 1]], 166.0);
        assert_relative_eq!(c[[1, 1, 0]], 199.0);
        assert_relative_eq!(c[[1, 1, 1]], 214.0);
    }
}
