//! Reduction operations for tensors.

use crate::{
    dimension::{Dimension, DynamicDim},
    error::{Result, TensorustError},
    tensor::Tensor,
    storage::Storage,
};
use num_traits::{Float, Zero};

/// Compute the sum of all elements in the tensor.
pub fn sum<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, DynamicDim, S>>
where
    T: Clone + Zero + std::iter::Sum + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let sum_val = tensor.storage().to_vec().into_iter().sum();
    let result_storage = S::from_vec(vec![sum_val]);
    Tensor::new(result_storage, vec![1].into())
}

/// Compute the mean of all elements in the tensor.
pub fn mean<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, DynamicDim, S>>
where
    T: Clone + Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let sum_val: T = tensor.storage().to_vec().into_iter().sum();
    let count = T::from(tensor.len()).unwrap();
    let mean_val = sum_val / count;
    let result_storage = S::from_vec(vec![mean_val]);
    Tensor::new(result_storage, vec![1].into())
}

/// Find the maximum value in the tensor.
pub fn max<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, DynamicDim, S>>
where
    T: Clone + PartialOrd + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let max_val = tensor
        .storage()
        .to_vec()
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or_default();
    let result_storage = S::from_vec(vec![max_val]);
    Tensor::new(result_storage, vec![1].into())
}

/// Find the minimum value in the tensor.
pub fn min<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, DynamicDim, S>>
where
    T: Clone + PartialOrd + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let min_val = tensor
        .storage()
        .to_vec()
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or_default();
    let result_storage = S::from_vec(vec![min_val]);
    Tensor::new(result_storage, vec![1].into())
}

/// Compute the sum of elements along a specific dimension.
pub fn sum_along_dim<T, D, S>(
    tensor: &Tensor<T, D, S>,
    dim: usize,
    keep_dim: bool,
) -> Result<Tensor<T, DynamicDim, S>>
where
    T: Clone + Zero + std::iter::Sum + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let shape = tensor.shape().as_ref().to_vec();
    if dim >= shape.len() {
        return Err(TensorustError::invalid_dimension(dim, shape.len()));
    }

    // Calculate output shape
    let mut output_shape = shape.clone();
    if keep_dim {
        output_shape[dim] = 1;
    } else {
        output_shape.remove(dim);
    }

    // Implementation would go here
    // This is a simplified version that just sums all elements
    let sum_val = tensor.storage().to_vec().into_iter().sum();
    let result_storage = S::from_vec(vec![sum_val]);
    Tensor::new(result_storage, output_shape.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dimension::DynamicDim,
        storage::CpuStorage,
    };

    #[test]
    fn test_sum() {
        let a = Tensor::<f32, DynamicDim, _>::from(vec![1.0, 2.0, 3.0]);
        let s = sum(&a).unwrap();
        assert_eq!(s.to_vec(), vec![6.0]);
    }

    #[test]
    fn test_mean() {
        let a = Tensor::from(vec![1.0, 2.0, 3.0, 4.0]);
        let m = mean(&a).unwrap();
        assert_eq!(m.to_vec(), vec![2.5]);
    }

    #[test]
    fn test_max() {
        let a = Tensor::from(vec![1.0, 3.0, 2.0]);
        let m = max(&a).unwrap();
        assert_eq!(m.to_vec(), vec![3.0]);
    }

    #[test]
    fn test_min() {
        let a = Tensor::from(vec![1.0, 3.0, 2.0]);
        let m = min(&a).unwrap();
        assert_eq!(m.to_vec(), vec![1.0]);
    }
}
