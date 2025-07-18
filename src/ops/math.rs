//! Mathematical functions for tensors.

use crate::{
    dimension::{static_dim::StaticDim, Dimension},
    error::Result,
    tensor::Tensor,
    storage::Storage,
    ops::elementwise::{map, zip_with},
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

/// In-place element-wise hyperbolic tangent function.
pub fn tanh_<T, D, S>(tensor: &mut Tensor<T, D, S>) -> Result<()>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    tensor.as_mut_slice().iter_mut().for_each(|x| *x = x.tanh());
    Ok(())
}

/// Gradient of the element-wise hyperbolic tangent function.
pub fn tanh_grad<T, D, S>(
    output: &Tensor<T, D, S>,
    grad_output: &Tensor<T, D, S>,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let one = T::one();
    let result_storage = {
        let output_data = output.storage().to_vec();
        let grad_output_data = grad_output.storage().to_vec();
        let result_data = output_data
            .into_iter()
            .zip(grad_output_data)
            .map(|(y, g)| g * (one - y * y))
            .collect();
        S::from_vec(result_data)
    };
    Tensor::new(result_storage, output.shape().clone())
}

/// In-place element-wise sigmoid function.
pub fn sigmoid_<T, D, S>(tensor: &mut Tensor<T, D, S>) -> Result<()>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let one = T::one();
    tensor.as_mut_slice().iter_mut().for_each(|x| *x = one / (one + (-*x).exp()));
    Ok(())
}

/// Gradient of the element-wise sigmoid function.
pub fn sigmoid_grad<T, D, S>(
    output: &Tensor<T, D, S>,
    grad_output: &Tensor<T, D, S>,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let one = T::one();
    let result_storage = {
        let output_data = output.storage().to_vec();
        let grad_output_data = grad_output.storage().to_vec();
        let result_data = output_data
            .into_iter()
            .zip(grad_output_data)
            .map(|(y, g)| g * y * (one - y))
            .collect();
        S::from_vec(result_data)
    };
    Tensor::new(result_storage, output.shape().clone())
}

/// In-place element-wise ReLU (Rectified Linear Unit) function.
pub fn relu_<T, D, S>(tensor: &mut Tensor<T, D, S>) -> Result<()>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let zero = T::zero();
    tensor.as_mut_slice().iter_mut().for_each(|x| *x = x.max(zero));
    Ok(())
}

/// Gradient of the element-wise ReLU function.
pub fn relu_grad<T, D, S>(
    output: &Tensor<T, D, S>,
    grad_output: &Tensor<T, D, S>,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let zero = T::zero();
    let one = T::one();
    let result_storage = {
        let output_data = output.storage().to_vec();
        let grad_output_data = grad_output.storage().to_vec();
        let result_data = output_data
            .into_iter()
            .zip(grad_output_data)
            .map(|(y, g)| if y > zero { g } else { zero })
            .collect();
        S::from_vec(result_data)
    };
    Tensor::new(result_storage, output.shape().clone())
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

pub fn softmax<T, D, S>(tensor: &Tensor<T, D, S>, dim: usize) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let max = tensor.max(Some(dim), true)?;
    let exp = (tensor - &max)?.exp()?;
    let sum = exp.sum(Some(dim), true)?;
    Ok((exp / &sum)?)
}

pub fn softmax_grad<T, D, S>(
    output: &Tensor<T, D, S>,
    grad_output: &Tensor<T, D, S>,
    dim: usize,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let sum = (grad_output * output)?.sum(Some(dim), true)?;
    Ok(((grad_output - &sum)?) * output)
}

pub fn leaky_relu<T, D, S>(tensor: &Tensor<T, D, S>, negative_slope: T) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let zero = T::zero();
    map(tensor, |x| if x >= zero { x } else { negative_slope * x })
}

pub fn leaky_relu_<T, D, S>(tensor: &mut Tensor<T, D, S>, negative_slope: T) -> Result<()>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let zero = T::zero();
    tensor.as_mut_slice().iter_mut().for_each(|x| {
        if *x < zero {
            *x = negative_slope * *x
        }
    });
    Ok(())
}

pub fn leaky_relu_grad<T, D, S>(
    input: &Tensor<T, D, S>,
    grad_output: &Tensor<T, D, S>,
    negative_slope: T,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let zero = T::zero();
    let one = T::one();
    zip_with(input, grad_output, |x, g| {
        if x >= zero {
            g
        } else {
            negative_slope * g
        }
    })
}

pub fn elu<T, D, S>(tensor: &Tensor<T, D, S>, alpha: T) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let zero = T::zero();
    map(tensor, |x| if x >= zero { x } else { alpha * (x.exp() - T::one()) })
}

pub fn elu_<T, D, S>(tensor: &mut Tensor<T, D, S>, alpha: T) -> Result<()>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let zero = T::zero();
    tensor.as_mut_slice().iter_mut().for_each(|x| {
        if *x < zero {
            *x = alpha * (x.exp() - T::one())
        }
    });
    Ok(())
}

pub fn elu_grad<T, D, S>(
    input: &Tensor<T, D, S>,
    output: &Tensor<T, D, S>,
    grad_output: &Tensor<T, D, S>,
    alpha: T,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let zero = T::zero();
    zip_with(input, grad_output, |x, g| {
        if x >= zero {
            g
        } else {
            g * (alpha + output)
        }
    })
}

pub fn gelu<T, D, S>(tensor: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let half = T::from(0.5).unwrap();
    let one = T::one();
    let two = T::from(2.0).unwrap();
    map(tensor, |x| {
        half * x * (one + (x / two.sqrt()).erf())
    })
}

pub fn gelu_grad<T, D, S>(
    input: &Tensor<T, D, S>,
    _output: &Tensor<T, D, S>,
    grad_output: &Tensor<T, D, S>,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let half = T::from(0.5).unwrap();
    let one = T::one();
    let two = T::from(2.0).unwrap();
    zip_with(input, grad_output, |x, g| {
        let cdf = half * (one + (x / two.sqrt()).erf());
        let pdf = (-half * x * x).exp() / (two * T::from(std::f64::consts::PI).unwrap()).sqrt();
        g * (cdf + x * pdf)
    })
}

pub fn mse_loss<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
) -> Result<Tensor<T, crate::dimension::StaticDim<0>, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let diff = predictions - targets;
    let squared_error = &diff * &diff;
    match reduction {
        crate::nn::losses::Reduction::Mean => squared_error.mean(),
        crate::nn::losses::Reduction::Sum => squared_error.sum(),
        crate::nn::losses::Reduction::None => Ok(squared_error),
    }
}

pub fn mse_loss_grad<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let diff = predictions - targets;
    let n = T::from(predictions.len()).unwrap();
    match reduction {
        crate::nn::losses::Reduction::Mean => Ok((diff * T::from(2.0).unwrap()) / n),
        crate::nn::losses::Reduction::Sum => Ok(diff * T::from(2.0).unwrap()),
        crate::nn::losses::Reduction::None => Ok(diff * T::from(2.0).unwrap()),
    }
}

pub fn cross_entropy_loss<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    ignore_index: Option<usize>,
    label_smoothing: T,
) -> Result<Tensor<T, crate::dimension::StaticDim<0>, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    // This is a simplified implementation. A real implementation would be more complex.
    let log_softmax = predictions.log_softmax(1)?;
    let nll_loss = nll_loss(&log_softmax, targets, reduction, ignore_index, None)?;
    Ok(nll_loss)
}

pub fn cross_entropy_loss_grad<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    ignore_index: Option<usize>,
    label_smoothing: T,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    // This is a simplified implementation. A real implementation would be more complex.
    let softmax = predictions.softmax(1)?;
    let grad = softmax - targets;
    Ok(grad)
}

pub fn binary_cross_entropy_loss<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    weight: Option<&Tensor<T, D, S>>,
) -> Result<Tensor<T, crate::dimension::StaticDim<0>, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let one = T::one();
    let loss = -(targets * &predictions.log()? + (one - targets) * &(one - predictions).log()?);
    match reduction {
        crate::nn::losses::Reduction::Mean => loss.mean(),
        crate::nn::losses::Reduction::Sum => loss.sum(),
        crate::nn::losses::Reduction::None => Ok(loss),
    }
}

pub fn binary_cross_entropy_loss_grad<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    weight: Option<&Tensor<T, D, S>>,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let one = T::one();
    let grad = (predictions - targets) / (predictions * (one - predictions));
    Ok(grad)
}

pub fn l1_loss<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
) -> Result<Tensor<T, crate::dimension::StaticDim<0>, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let diff = predictions - targets;
    let abs_diff = diff.abs();
    match reduction {
        crate::nn::losses::Reduction::Mean => abs_diff.mean(),
        crate::nn::losses::Reduction::Sum => abs_diff.sum(),
        crate::nn::losses::Reduction::None => Ok(abs_diff),
    }
}

pub fn l1_loss_grad<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let diff = predictions - targets;
    let grad = diff.signum();
    Ok(grad)
}

pub fn smooth_l1_loss<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    beta: T,
) -> Result<Tensor<T, crate::dimension::StaticDim<0>, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let diff = predictions - targets;
    let abs_diff = diff.abs();
    let half = T::from(0.5).unwrap();
    let loss = zip_with(&abs_diff, &diff, |abs_d, d| {
        if abs_d < beta {
            half * d * d / beta
        } else {
            abs_d - half * beta
        }
    })?;
    match reduction {
        crate::nn::losses::Reduction::Mean => loss.mean(),
        crate::nn::losses::Reduction::Sum => loss.sum(),
        crate::nn::losses::Reduction::None => Ok(loss),
    }
}

pub fn smooth_l1_loss_grad<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    beta: T,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let diff = predictions - targets;
    let abs_diff = diff.abs();
    let grad = zip_with(&abs_diff, &diff, |abs_d, d| {
        if abs_d < beta {
            d / beta
        } else {
            d.signum()
        }
    })?;
    Ok(grad)
}

pub fn kl_div_loss<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    log_target: bool,
) -> Result<Tensor<T, crate::dimension::StaticDim<0>, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let log_predictions = predictions.log()?;
    let loss = if log_target {
        targets.exp()? * (targets - &log_predictions)
    } else {
        targets * (targets.log()? - &log_predictions)
    };
    match reduction {
        crate::nn::losses::Reduction::Mean => loss.mean(),
        crate::nn::losses::Reduction::Sum => loss.sum(),
        crate::nn::losses::Reduction::None => Ok(loss),
    }
}

pub fn kl_div_loss_grad<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    log_target: bool,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let grad = if log_target {
        -targets.exp()? / predictions
    } else {
        -targets / predictions
    };
    Ok(grad)
}

pub fn nll_loss<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    ignore_index: Option<usize>,
    weight: Option<&Tensor<T, D, S>>,
) -> Result<Tensor<T, crate::dimension::StaticDim<0>, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    // This is a simplified implementation. A real implementation would be more complex.
    let loss = -predictions;
    match reduction {
        crate::nn::losses::Reduction::Mean => loss.mean(),
        crate::nn::losses::Reduction::Sum => loss.sum(),
        crate::nn::losses::Reduction::None => Ok(loss),
    }
}

pub fn nll_loss_grad<T, D, S>(
    predictions: &Tensor<T, D, S>,
    targets: &Tensor<T, D, S>,
    reduction: crate::nn::losses::Reduction,
    ignore_index: Option<usize>,
    weight: Option<&Tensor<T, D, S>>,
) -> Result<Tensor<T, D, S>>
where
    T: Float + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let grad = -Tensor::ones_like(predictions)?;
    Ok(grad)
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
