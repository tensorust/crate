//! Loss functions for neural networks.

use crate::{
    dimension::Dimension,
    tensor::Tensor,
    storage::Storage,
    error::Result,
    ops::{
        mse_loss_grad, cross_entropy_loss_grad,
        binary_cross_entropy_loss_grad,
        l1_loss_grad, smooth_l1_loss_grad,
        kl_div_loss_grad, nll_loss_grad,
    },
};
use std::fmt;

/// Mean Squared Error (MSE) loss function.
///
/// Computes the mean squared error between the predictions and targets.
///
/// # Formula
///
/// \[ \text{MSE}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2 \]
///
/// where \( \hat{y} \) is the prediction and \( y \) is the target.
#[derive(Debug, Clone, Copy, Default)]
pub struct MSELoss {
    /// Whether to compute the mean over the loss. If false, returns the sum of the loss.
    reduction: Reduction,
}

impl MSELoss {
    /// Create a new MSELoss with the given reduction.
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
    
    /// Create a new MSELoss that computes the mean over the loss.
    pub fn mean() -> Self {
        Self { reduction: Reduction::Mean }
    }
    
    /// Create a new MSELoss that computes the sum over the loss.
    pub fn sum() -> Self {
        Self { reduction: Reduction::Sum }
    }
}

impl<T, D, S> crate::nn::Loss<T, D, S> for MSELoss
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn compute(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, crate::dimension::static_::StaticDim<0>, S>> {
        mse_loss(predictions, targets, self.reduction)
    }
    
    fn gradient(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        mse_loss_grad(predictions, targets, self.reduction)
    }
}

/// Cross-Entropy loss function.
///
/// Computes the cross-entropy loss between the predicted class probabilities and the true class labels.
///
/// # Formula
///
/// \[ \text{CE}(p, y) = -\sum_{i=1}^n y_i \log(p_i) \]
///
/// where \( p \) is the predicted probability distribution and \( y \) is the target distribution.
#[derive(Debug, Clone, Copy)]
pub struct CrossEntropyLoss {
    /// Whether to compute the mean over the loss. If false, returns the sum of the loss.
    reduction: Reduction,
    /// The index to ignore, if any.
    ignore_index: Option<usize>,
    /// The label smoothing factor.
    label_smoothing: f32,
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            ignore_index: None,
            label_smoothing: 0.0,
        }
    }
}

impl CrossEntropyLoss {
    /// Create a new CrossEntropyLoss with the given reduction, ignore_index, and label_smoothing.
    pub fn new(reduction: Reduction, ignore_index: Option<usize>, label_smoothing: f32) -> Self {
        Self {
            reduction,
            ignore_index,
            label_smoothing,
        }
    }
    
    /// Create a new CrossEntropyLoss that computes the mean over the loss.
    pub fn mean() -> Self {
        Self {
            reduction: Reduction::Mean,
            ignore_index: None,
            label_smoothing: 0.0,
        }
    }
    
    /// Create a new CrossEntropyLoss that computes the sum over the loss.
    pub fn sum() -> Self {
        Self {
            reduction: Reduction::Sum,
            ignore_index: None,
            label_smoothing: 0.0,
        }
    }
    
    /// Set the index to ignore.
    pub fn with_ignore_index(mut self, ignore_index: usize) -> Self {
        self.ignore_index = Some(ignore_index);
        self
    }
    
    /// Set the label smoothing factor.
    pub fn with_label_smoothing(mut self, label_smoothing: f32) -> Self {
        self.label_smoothing = label_smoothing;
        self
    }
}

impl<T, D, S> crate::nn::Loss<T, D, S> for CrossEntropyLoss
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32> + std::ops::Sub<Output = T> + std::ops::Div<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    f32: From<T>,
{
    fn compute(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, crate::dimension::static_::StaticDim<0>, S>> {
        cross_entropy_loss(
            predictions,
            targets,
            self.reduction,
            self.ignore_index,
            self.label_smoothing.into(),
        )
    }
    
    fn gradient(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        cross_entropy_loss_grad(
            predictions,
            targets,
            self.reduction,
            self.ignore_index,
            self.label_smoothing.into(),
        )
    }
}

/// Binary Cross-Entropy loss function.
///
/// Computes the binary cross-entropy loss between the predicted probabilities and the true binary labels.
///
/// # Formula
///
/// \[ \text{BCE}(p, y) = -\frac{1}{n} \sum_{i=1}^n [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)] \]
///
/// where \( p \) is the predicted probability and \( y \) is the target binary label.
#[derive(Debug, Clone, Copy)]
pub struct BCELoss {
    /// Whether to compute the mean over the loss. If false, returns the sum of the loss.
    reduction: Reduction,
    /// The weight to apply to the loss of each class.
    weight: Option<f32>,
}

impl Default for BCELoss {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            weight: None,
        }
    }
}

impl BCELoss {
    /// Create a new BCELoss with the given reduction and weight.
    pub fn new(reduction: Reduction, weight: Option<f32>) -> Self {
        Self { reduction, weight }
    }
    
    /// Create a new BCELoss that computes the mean over the loss.
    pub fn mean() -> Self {
        Self {
            reduction: Reduction::Mean,
            weight: None,
        }
    }
    
    /// Create a new BCELoss that computes the sum over the loss.
    pub fn sum() -> Self {
        Self {
            reduction: Reduction::Sum,
            weight: None,
        }
    }
    
    /// Set the weight to apply to the loss of each class.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = Some(weight);
        self
    }
}

impl<T, D, S> crate::nn::Loss<T, D, S> for BCELoss
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32> + std::ops::Sub<Output = T> + std::ops::Div<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    f32: From<T>,
{
    fn compute(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, crate::dimension::static_::StaticDim<0>, S>> {
        let weight = self.weight.map(|w| T::from(w));
        binary_cross_entropy_loss(predictions, targets, self.reduction, weight.as_ref())
    }
    
    fn gradient(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        let weight = self.weight.map(|w| T::from(w));
        binary_cross_entropy_loss_grad(predictions, targets, self.reduction, weight.as_ref())
    }
}

/// L1 loss function.
///
/// Computes the mean absolute error between the predictions and targets.
///
/// # Formula
///
/// \[ \text{L1}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^n |\hat{y}_i - y_i| \]
///
/// where \( \hat{y} \) is the prediction and \( y \) is the target.
#[derive(Debug, Clone, Copy, Default)]
pub struct L1Loss {
    /// Whether to compute the mean over the loss. If false, returns the sum of the loss.
    reduction: Reduction,
}

impl L1Loss {
    /// Create a new L1Loss with the given reduction.
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
    
    /// Create a new L1Loss that computes the mean over the loss.
    pub fn mean() -> Self {
        Self { reduction: Reduction::Mean }
    }
    
    /// Create a new L1Loss that computes the sum over the loss.
    pub fn sum() -> Self {
        Self { reduction: Reduction::Sum }
    }
}

impl<T, D, S> crate::nn::Loss<T, D, S> for L1Loss
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32> + std::ops::Sub<Output = T> + std::ops::Div<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn compute(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, crate::dimension::static_::StaticDim<0>, S>> {
        l1_loss(predictions, targets, self.reduction)
    }
    
    fn gradient(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        l1_loss_grad(predictions, targets, self.reduction)
    }
}

/// Smooth L1 loss function.
///
/// Computes the smooth L1 loss between the predictions and targets.
///
/// # Formula
///
/// \[ \text{SmoothL1}(\hat{y}, y) = \begin{cases}
/// 0.5 (\hat{y}_i - y_i)^2 / \beta & \text{if } |\hat{y}_i - y_i| < \beta \\
/// |\hat{y}_i - y_i| - 0.5 * \beta & \text{otherwise}
/// \end{cases} \]
///
/// where \( \hat{y} \) is the prediction and \( y \) is the target.
#[derive(Debug, Clone, Copy)]
pub struct SmoothL1Loss {
    /// The threshold at which to change between L1 and L2 loss.
    beta: f32,
    /// Whether to compute the mean over the loss. If false, returns the sum of the loss.
    reduction: Reduction,
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self {
            beta: 1.0,
            reduction: Reduction::Mean,
        }
    }
}

impl SmoothL1Loss {
    /// Create a new SmoothL1Loss with the given beta and reduction.
    pub fn new(beta: f32, reduction: Reduction) -> Self {
        Self { beta, reduction }
    }
    
    /// Create a new SmoothL1Loss that computes the mean over the loss.
    pub fn mean(beta: f32) -> Self {
        Self {
            beta,
            reduction: Reduction::Mean,
        }
    }
    
    /// Create a new SmoothL1Loss that computes the sum over the loss.
    pub fn sum(beta: f32) -> Self {
        Self {
            beta,
            reduction: Reduction::Sum,
        }
    }
}

impl<T, D, S> crate::nn::Loss<T, D, S> for SmoothL1Loss
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32> + std::ops::Sub<Output = T> + std::ops::Div<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    f32: From<T>,
{
    fn compute(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, crate::dimension::static_::StaticDim<0>, S>> {
        smooth_l1_loss(predictions, targets, self.reduction, self.beta.into())
    }
    
    fn gradient(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        smooth_l1_loss_grad(predictions, targets, self.reduction, self.beta.into())
    }
}

/// Kullback-Leibler divergence loss function.
///
/// Computes the Kullback-Leibler divergence between the predicted distribution and the target distribution.
///
/// # Formula
///
/// \[ \text{KL}(p, q) = \sum_{i=1}^n p_i \log\left(\frac{p_i}{q_i}\right) \]
///
/// where \( p \) is the target distribution and \( q \) is the predicted distribution.
#[derive(Debug, Clone, Copy, Default)]
pub struct KLDivLoss {
    /// Whether to compute the mean over the loss. If false, returns the sum of the loss.
    reduction: Reduction,
    /// Whether to compute the true KL divergence (if true) or the KL divergence with log-targets (if false).
    log_target: bool,
}

impl KLDivLoss {
    /// Create a new KLDivLoss with the given reduction and log_target.
    pub fn new(reduction: Reduction, log_target: bool) -> Self {
        Self { reduction, log_target }
    }
    
    /// Create a new KLDivLoss that computes the mean over the loss.
    pub fn mean(log_target: bool) -> Self {
        Self {
            reduction: Reduction::Mean,
            log_target,
        }
    }
    
    /// Create a new KLDivLoss that computes the sum over the loss.
    pub fn sum(log_target: bool) -> Self {
        Self {
            reduction: Reduction::Sum,
            log_target,
        }
    }
}

impl<T, D, S> crate::nn::Loss<T, D, S> for KLDivLoss
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32> + std::ops::Sub<Output = T> + std::ops::Div<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    f32: From<T>,
{
    fn compute(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, crate::dimension::static_::StaticDim<0>, S>> {
        kl_div_loss(predictions, targets, self.reduction, self.log_target)
    }
    
    fn gradient(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        kl_div_loss_grad(predictions, targets, self.reduction, self.log_target)
    }
}

/// Negative Log Likelihood loss function.
///
/// Computes the negative log likelihood loss between the predicted log-probabilities and the target class indices.
///
/// # Formula
///
/// \[ \text{NLL}(\log(p), y) = -\log(p_y) \]
///
/// where \( p \) is the predicted probability distribution and \( y \) is the target class index.
#[derive(Debug, Clone, Copy)]
pub struct NLLLoss {
    /// Whether to compute the mean over the loss. If false, returns the sum of the loss.
    reduction: Reduction,
    /// The index to ignore, if any.
    ignore_index: Option<usize>,
    /// The weight to apply to the loss of each class.
    weight: Option<f32>,
}

impl Default for NLLLoss {
    fn default() -> Self {
        Self {
            reduction: Reduction::Mean,
            ignore_index: None,
            weight: None,
        }
    }
}

impl NLLLoss {
    /// Create a new NLLLoss with the given reduction, ignore_index, and weight.
    pub fn new(reduction: Reduction, ignore_index: Option<usize>, weight: Option<f32>) -> Self {
        Self {
            reduction,
            ignore_index,
            weight,
        }
    }
    
    /// Create a new NLLLoss that computes the mean over the loss.
    pub fn mean() -> Self {
        Self {
            reduction: Reduction::Mean,
            ignore_index: None,
            weight: None,
        }
    }
    
    /// Create a new NLLLoss that computes the sum over the loss.
    pub fn sum() -> Self {
        Self {
            reduction: Reduction::Sum,
            ignore_index: None,
            weight: None,
        }
    }
    
    /// Set the index to ignore.
    pub fn with_ignore_index(mut self, ignore_index: usize) -> Self {
        self.ignore_index = Some(ignore_index);
        self
    }
    
    /// Set the weight to apply to the loss of each class.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = Some(weight);
        self
    }
}

impl<T, D, S> crate::nn::Loss<T, D, S> for NLLLoss
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32> + std::ops::Sub<Output = T> + std::ops::Div<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    f32: From<T>,
{
    fn compute(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, crate::dimension::static_::StaticDim<0>, S>> {
        let weight = self.weight.map(|w| T::from(w));
        nll_loss(
            predictions,
            targets,
            self.reduction,
            self.ignore_index,
            weight.as_ref(),
        )
    }
    
    fn gradient(
        &self,
        predictions: &Tensor<T, D, S>,
        targets: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        let weight = self.weight.map(|w| T::from(w));
        nll_loss_grad(
            predictions,
            targets,
            self.reduction,
            self.ignore_index,
            weight.as_ref(),
        )
    }
}

/// Specifies how to reduce the loss across the batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// The output will be the mean of the loss over the batch.
    Mean,
    /// The output will be the sum of the loss over the batch.
    Sum,
    /// The output will be the loss for each sample in the batch.
    None,
}

impl Default for Reduction {
    fn default() -> Self {
        Self::Mean
    }
}

/// A module that applies a loss function to the input.
/// This is a convenience wrapper around loss functions to make them usable as layers.
#[derive(Debug, Clone)]
pub struct LossFunc<L> {
    /// The loss function to apply.
    loss: L,
}

impl<L> LossFunc<L> {
    /// Create a new loss function layer.
    pub fn new(loss: L) -> Self {
        Self { loss }
    }
}

impl<T, D, S, L> LossFunc<L>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    L: crate::nn::Loss<T, D, S> + 'static,
{
    /// Compute the loss between the predictions and targets.
    pub fn compute<'a>(
        &self,
        predictions: &'a Tensor<T, D, S>,
        targets: &'a Tensor<T, D, S>,
    ) -> Result<Tensor<T, crate::dimension::static_::StaticDim<0>, S>> {
        self.loss.compute(predictions, targets)
    }
    
    /// Compute the gradient of the loss with respect to the predictions.
    pub fn gradient<'a>(
        &self,
        predictions: &'a Tensor<T, D, S>,
        targets: &'a Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        self.loss.gradient(predictions, targets)
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
    fn test_mse_loss() {
        let mse = MSELoss::mean();
        let predictions = tensor!([1.0, 2.0, 3.0]);
        let targets = tensor!([0.0, 2.0, 6.0]);
        
        let loss = mse.compute(&predictions, &targets).unwrap();
        assert_relative_eq!(loss[[0]], (1.0 + 0.0 + 9.0) / 3.0, epsilon = 1e-6);
        
        let grad = mse.gradient(&predictions, &targets).unwrap();
        assert_eq!(grad.data(), &[2.0/3.0, 0.0, -2.0]);
    }
    
    #[test]
    fn test_cross_entropy_loss() {
        let ce = CrossEntropyLoss::mean();
        let predictions = tensor!([
            [0.1, 0.2, 0.7],
            [0.9, 0.05, 0.05],
        ]);
        let targets = tensor!([2, 0]);
        
        let loss = ce.compute(&predictions, &targets).unwrap();
        let expected_loss = -((0.7f32.ln() + 0.9f32.ln()) / 2.0);
        assert_relative_eq!(loss[[0]] as f32, expected_loss, epsilon = 1e-6);
    }
    
    #[test]
    fn test_l1_loss() {
        let l1 = L1Loss::mean();
        let predictions = tensor!([1.0, 2.0, 3.0]);
        let targets = tensor!([0.0, 2.0, 6.0]);
        
        let loss = l1.compute(&predictions, &targets).unwrap();
        assert_relative_eq!(loss[[0]], (1.0 + 0.0 + 3.0) / 3.0, epsilon = 1e-6);
        
        let grad = l1.gradient(&predictions, &targets).unwrap();
        assert_eq!(grad.data(), &[1.0/3.0, 0.0, -1.0/3.0]);
    }
}
