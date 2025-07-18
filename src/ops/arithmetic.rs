//! Arithmetic operations for tensors.

use crate::{
    dimension::Dimension,
    error::{Result, TensorustError},
    tensor::Tensor,
    storage::Storage,
};
use std::ops::{Add, Div, Mul, Sub};

/// Element-wise addition of two tensors.
pub fn add<T, D, S>(
    lhs: &Tensor<T, D, S>,
    rhs: &Tensor<T, D, S>,
) -> Result<Tensor<T, D, S>>
where
    T: Clone + Add<Output = T> + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    // Check shapes are compatible for broadcasting
    if lhs.shape() != rhs.shape() {
        return Err(TensorustError::shape_mismatch(
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
            .map(|(a, b)| a + b)
            .collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, lhs.shape().clone())
}

/// Element-wise subtraction of two tensors.
pub fn sub<T, D, S>(
    lhs: &Tensor<T, D, S>,
    rhs: &Tensor<T, D, S>,
) -> Result<Tensor<T, D, S>>
where
    T: Clone + Sub<Output = T> + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    // Check shapes are compatible for broadcasting
    if lhs.shape() != rhs.shape() {
        return Err(TensorustError::shape_mismatch(
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
            .map(|(a, b)| a - b)
            .collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, lhs.shape().clone())
}

/// Element-wise multiplication of two tensors.
pub fn mul<T, D, S>(
    lhs: &Tensor<T, D, S>,
    rhs: &Tensor<T, D, S>,
) -> Result<Tensor<T, D, S>>
where
    T: Clone + Mul<Output = T> + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    // Check shapes are compatible for broadcasting
    if lhs.shape() != rhs.shape() {
        return Err(TensorustError::shape_mismatch(
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
            .map(|(a, b)| a * b)
            .collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, lhs.shape().clone())
}

/// Element-wise division of two tensors.
pub fn div<T, D, S>(
    lhs: &Tensor<T, D, S>,
    rhs: &Tensor<T, D, S>,
) -> Result<Tensor<T, D, S>>
where
    T: Clone + Div<Output = T> + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    // Check shapes are compatible for broadcasting
    if lhs.shape() != rhs.shape() {
        return Err(TensorustError::shape_mismatch(
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
            .map(|(a, b)| a / b)
            .collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, lhs.shape().clone())
}

/// Element-wise negation of a tensor.
pub fn neg<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Clone + std::ops::Neg<Output = T> + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let result_storage = {
        let data = tensor.storage().to_vec();
        let result_data = data.into_iter().map(|x| -x).collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, tensor.shape().clone())
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
    fn test_add() {
        let a = Tensor::<f32, DynamicDim, _>::from(vec![1.0, 2.0, 3.0]);
        let b = Tensor::from(vec![4.0, 5.0, 6.0]);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = Tensor::from(vec![4.0, 5.0, 6.0]);
        let b = Tensor::from(vec![1.0, 2.0, 3.0]);
        let c = sub(&a, &b).unwrap();
        assert_eq!(c.to_vec(), vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from(vec![1.0, 2.0, 3.0]);
        let b = Tensor::from(vec![4.0, 5.0, 6.0]);
        let c = mul(&a, &b).unwrap();
        assert_eq!(c.to_vec(), vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_div() {
        let a = Tensor::from(vec![4.0, 10.0, 18.0]);
        let b = Tensor::from(vec![2.0, 5.0, 6.0]);
        let c = div(&a, &b).unwrap();
        assert_eq!(c.to_vec(), vec![2.0, 2.0, 3.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from(vec![1.0, -2.0, 3.0]);
        let b = neg(&a).unwrap();
        assert_eq!(b.to_vec(), vec![-1.0, 2.0, -3.0]);
    }
}
