//! Mathematical functions for tensors.

use crate::{
    dimension::Dimension,
    error::Result,
    tensor::Tensor,
    storage::Storage,
};
use num_traits::Float;

/// Element-wise exponential function.
pub fn exp<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let result_storage = {
        let data = tensor.storage().to_vec();
        let result_data = data.into_iter().map(|x| x.exp()).collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, tensor.shape().clone())
}

/// Element-wise natural logarithm.
pub fn log<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let result_storage = {
        let data = tensor.storage().to_vec();
        let result_data = data.into_iter().map(|x| x.ln()).collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, tensor.shape().clone())
}

/// Element-wise square root.
pub fn sqrt<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let result_storage = {
        let data = tensor.storage().to_vec();
        let result_data = data.into_iter().map(|x| x.sqrt()).collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, tensor.shape().clone())
}

/// Element-wise power function.
pub fn powf<T, D, S>(tensor: &Tensor<T, D, S>, exponent: T) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let result_storage = {
        let data = tensor.storage().to_vec();
        let result_data = data.into_iter().map(|x| x.powf(exponent)).collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, tensor.shape().clone())
}

/// Element-wise sigmoid function.
pub fn sigmoid<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let one = T::one();
    let result_storage = {
        let data = tensor.storage().to_vec();
        let result_data = data
            .into_iter()
            .map(|x| one / (one + (-x).exp()))
            .collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, tensor.shape().clone())
}

/// Element-wise ReLU (Rectified Linear Unit) function.
pub fn relu<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let zero = T::zero();
    let result_storage = {
        let data = tensor.storage().to_vec();
        let result_data = data.into_iter().map(|x| x.max(zero)).collect();
        S::from_vec(result_data)
    };

    Tensor::new(result_storage, tensor.shape().clone())
}

/// Element-wise hyperbolic tangent function.
pub fn tanh<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let result_storage = {
        let data = tensor.storage().to_vec();
        let result_data = data.into_iter().map(|x| x.tanh()).collect();
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
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_exp() {
        let a = Tensor::<f32, DynamicDim, _>::from(vec![0.0, 1.0, 2.0]);
        let b = exp(&a).unwrap();
        assert_relative_eq!(b[0], 1.0);
        assert_relative_eq!(b[1], 1.0f32.exp());
        assert_relative_eq!(b[2], (2.0f32).exp());
    }

    #[test]
    fn test_log() {
        let a = Tensor::from(vec![1.0, std::f32::consts::E, 2.0]);
        let b = log(&a).unwrap();
        assert_relative_eq!(b[0], 0.0);
        assert_relative_eq!(b[1], 1.0);
        assert_relative_eq!(b[2], 2.0f32.ln());
    }

    #[test]
    fn test_sqrt() {
        let a = Tensor::from(vec![1.0, 4.0, 9.0]);
        let b = sqrt(&a).unwrap();
        assert_relative_eq!(b[0], 1.0);
        assert_relative_eq!(b[1], 2.0);
        assert_relative_eq!(b[2], 3.0);
    }

    #[test]
    fn test_powf() {
        let a = Tensor::from(vec![1.0, 2.0, 3.0]);
        let b = powf(&a, 2.0).unwrap();
        assert_relative_eq!(b[0], 1.0);
        assert_relative_eq!(b[1], 4.0);
        assert_relative_eq!(b[2], 9.0);
    }

    #[test]
    fn test_sigmoid() {
        let a = Tensor::from(vec![0.0]);
        let b = sigmoid(&a).unwrap();
        assert_relative_eq!(b[0], 0.5);
    }

    #[test]
    fn test_relu() {
        let a = Tensor::from(vec![-1.0, 0.0, 1.0]);
        let b = relu(&a).unwrap();
        assert_relative_eq!(b[0], 0.0);
        assert_relative_eq!(b[1], 0.0);
        assert_relative_eq!(b[2], 1.0);
    }

    #[test]
    fn test_tanh() {
        let a = Tensor::from(vec![0.0]);
        let b = tanh(&a).unwrap();
        assert_relative_eq!(b[0], 0.0);
    }
}
