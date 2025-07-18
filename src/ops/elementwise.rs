//! Element-wise operations for tensors.

use crate::{
    dimension::Dimension,
    error::Result,
    tensor::Tensor,
    storage::Storage,
};
use num_traits::Float;

/// Apply a function element-wise to a tensor.
pub fn map<T, D, S, F>(
    tensor: &Tensor<T, D, S>,
    f: F,
) -> Result<Tensor<T, D, S>>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    F: Fn(T) -> T + Send + Sync + 'static,
{
    let result_storage = {
        let data = tensor.storage().to_vec();
        let result_data = data.into_iter().map(f).collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, tensor.shape().clone())
}

/// Apply a function element-wise to two tensors.
pub fn zip_with<T, D, S, F>(
    lhs: &Tensor<T, D, S>,
    rhs: &Tensor<T, D, S>,
    f: F,
) -> Result<Tensor<T, D, S>>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    F: Fn(T, T) -> T + Send + Sync + 'static,
{
    // Check shapes are compatible
    if lhs.shape() != rhs.shape() {
        return Err(crate::error::TensorustError::shape_mismatch(
            lhs.shape().as_ref().to_vec(),
            rhs.shape().as_ref().to_vec(),
        ));
    }

    let result_storage = {
        let lhs_data = lhs.storage().to_vec();
        let rhs_data = rhs.storage().to_vec();
        let result_data = lhs_data
            .into_iter()
            .zip(rhs_data)
            .map(|(a, b)| f(a, b))
            .collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, lhs.shape().clone())
}

/// Element-wise comparison: greater than.
pub fn gt<T, D, S>(
    lhs: &Tensor<T, D, S>,
    rhs: &Tensor<T, D, S>,
) -> Result<Tensor<bool, D, S>>
where
    T: Clone + PartialOrd + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T> + Storage<bool>,
{
    // Check shapes are compatible
    if lhs.shape() != rhs.shape() {
        return Err(crate::error::TensorustError::shape_mismatch(
            lhs.shape().as_ref().to_vec(),
            rhs.shape().as_ref().to_vec(),
        ));
    }

    let result_storage = {
        let lhs_data = lhs.storage().to_vec();
        let rhs_data = rhs.storage().to_vec();
        let result_data = lhs_data
            .into_iter()
            .zip(rhs_data)
            .map(|(a, b)| a > b)
            .collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, lhs.shape().clone())
}

/// Element-wise comparison: less than.
pub fn lt<T, D, S>(
    lhs: &Tensor<T, D, S>,
    rhs: &Tensor<T, D, S>,
) -> Result<Tensor<bool, D, S>>
where
    T: Clone + PartialOrd + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T> + Storage<bool>,
{
    // Check shapes are compatible
    if lhs.shape() != rhs.shape() {
        return Err(crate::error::TensorustError::shape_mismatch(
            lhs.shape().as_ref().to_vec(),
            rhs.shape().as_ref().to_vec(),
        ));
    }

    let result_storage = {
        let lhs_data = lhs.storage().to_vec();
        let rhs_data = rhs.storage().to_vec();
        let result_data = lhs_data
            .into_iter()
            .zip(rhs_data)
            .map(|(a, b)| a < b)
            .collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, lhs.shape().clone())
}

/// Element-wise comparison: equal to.
pub fn eq<T, D, S>(
    lhs: &Tensor<T, D, S>,
    rhs: &Tensor<T, D, S>,
) -> Result<Tensor<bool, D, S>>
where
    T: Clone + PartialEq + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T> + Storage<bool>,
{
    // Check shapes are compatible
    if lhs.shape() != rhs.shape() {
        return Err(crate::error::TensorustError::shape_mismatch(
            lhs.shape().as_ref().to_vec(),
            rhs.shape().as_ref().to_vec(),
        ));
    }

    let result_storage = {
        let lhs_data = lhs.storage().to_vec();
        let rhs_data = rhs.storage().to_vec();
        let result_data = lhs_data
            .into_iter()
            .zip(rhs_data)
            .map(|(a, b)| a == b)
            .collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, lhs.shape().clone())
}

/// Element-wise absolute value.
pub fn abs<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Clone + num_traits::Signed + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    map(tensor, |x| x.abs())
}

/// Element-wise sign function.
pub fn signum<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Clone + num_traits::Signed + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    map(tensor, |x| x.signum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dimension::DynamicDim,
        storage::CpuStorage,
        tensorust,
    };

    #[test]
    fn test_map() {
        let a = Tensor::<f32, DynamicDim, _>::from(vec![1.0, 2.0, 3.0]);
        let b = map(&a, |x| x * 2.0).unwrap();
        assert_eq!(b.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_zip_with() {
        let a = Tensor::from(vec![1.0, 2.0, 3.0]);
        let b = Tensor::from(vec![4.0, 5.0, 6.0]);
        let c = zip_with(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_comparison_ops() {
        let a = Tensor::from(vec![1.0, 2.0, 3.0]);
        let b = Tensor::from(vec![2.0, 2.0, 2.0]);
        
        let gt_result = gt(&a, &b).unwrap();
        assert_eq!(gt_result.to_vec(), vec![false, false, true]);
        
        let lt_result = lt(&a, &b).unwrap();
        assert_eq!(lt_result.to_vec(), vec![true, false, false]);
        
        let eq_result = eq(&a, &b).unwrap();
        assert_eq!(eq_result.to_vec(), vec![false, true, false]);
    }

    #[test]
    fn test_abs() {
        let a = Tensor::from(vec![-1.0, 0.0, 1.0]);
        let b = abs(&a).unwrap();
        assert_eq!(b.to_vec(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_signum() {
        let a = Tensor::from(vec![-2.0, 0.0, 3.0]);
        let b = signum(&a).unwrap();
        assert_eq!(b.to_vec(), vec![-1.0, 0.0, 1.0]);
    }
}
