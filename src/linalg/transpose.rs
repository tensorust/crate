//! Matrix transpose operations.

use crate::{
    dimension::{static_dim::StaticDim, Dimension},
    error::Result,
    tensor::Tensor,
    storage::Storage,
};

/// Transpose a 2D matrix.
pub fn transpose<T, S>(
    tensor: &Tensor<T, StaticDim<2>, S>,
) -> Result<Tensor<T, StaticDim<2>, S>>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
{
    let shape = tensor.shape();
    let m = shape[0];
    let n = shape[1];
    
    let mut result_data = vec![T::default(); m * n];
    let data = tensor.storage().to_vec();
    
    for i in 0..m {
        for j in 0..n {
            result_data[j * m + i] = data[i * n + j].clone();
        }
    }
    
    let result_storage = S::from_vec(result_data);
    Tensor::new(result_storage, [n, m].into())
}

/// Permute the dimensions of a tensor.
pub fn permute<T, D, S, I>(
    tensor: &Tensor<T, D, S>,
    dims: I,
) -> Result<Tensor<T, D, S>>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    I: IntoIterator<Item = usize>,
    I::IntoIter: ExactSizeIterator,
{
    let dims: Vec<_> = dims.into_iter().collect();
    let shape = tensor.shape();
    
    if dims.len() != shape.ndims() {
        return Err(crate::error::TensorustError::invalid_dimensions(
            dims.len(),
            shape.ndims(),
        ));
    }
    
    // Validate that all dimensions are within bounds and unique
    let mut seen = std::collections::HashSet::new();
    for &dim in &dims {
        if dim >= shape.ndims() {
            return Err(crate::error::TensorustError::invalid_dimension(
                dim,
                shape.ndims(),
            ));
        }
        if !seen.insert(dim) {
            return Err(crate::error::TensorustError::invalid_argument(
                "Duplicate dimension in permute",
            ));
        }
    }
    
    // Calculate the new shape and strides
    let new_shape: Vec<_> = dims.iter().map(|&d| shape[d]).collect();
    let strides = shape.strides();
    let new_strides: Vec<_> = dims.iter().map(|&d| strides[d]).collect();
    
    // Create a new tensor with the same data but new shape and strides
    let mut result = tensor.clone();
    result.reshape_with_strides(new_shape.into(), new_strides);
    
    Ok(result)
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

    #[test]
    fn test_transpose() {
        let a = tensor!([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]);
        let b = transpose(&a).unwrap();
        
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b[[0, 0]], 1.0);
        assert_eq!(b[[0, 1]], 4.0);
        assert_eq!(b[[1, 0]], 2.0);
        assert_eq!(b[[1, 1]], 5.0);
        assert_eq!(b[[2, 0]], 3.0);
        assert_eq!(b[[2, 1]], 6.0);
    }

    #[test]
    fn test_permute() {
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
        
        // Permute dimensions (0, 1, 2) -> (1, 0, 2)
        let b = permute(&a, [1, 0, 2]).unwrap();
        assert_eq!(b.shape(), &[2, 2, 2]);
        assert_eq!(b[[0, 0, 0]], 1.0);
        assert_eq!(b[[0, 1, 0]], 5.0);
        assert_eq!(b[[1, 0, 1]], 6.0);
        
        // Permute dimensions (0, 1, 2) -> (0, 2, 1)
        let c = permute(&a, [0, 2, 1]).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
        assert_eq!(c[[0, 0, 0]], 1.0);
        assert_eq!(c[[0, 1, 0]], 2.0);
        assert_eq!(c[[0, 0, 1]], 3.0);
        assert_eq!(c[[0, 1, 1]], 4.0);
    }
}
