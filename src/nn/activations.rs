//! Activation functions for neural networks.

use crate::{
    dimension::{dynamic::DynamicDim, Dimension},
    error::Result,
    ops::{
        elu, elu_, elu_grad, gelu, gelu_grad, leaky_relu, leaky_relu_, leaky_relu_grad, relu,
        relu_, relu_grad, sigmoid, sigmoid_, sigmoid_grad, softmax, softmax_grad, tanh, tanh_,
        tanh_grad,
    },
    storage::Storage,
    tensor::Tensor,
};
use std::fmt;

/// ReLU (Rectified Linear Unit) activation function.
///
/// Formula: `f(x) = max(0, x)`
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU {
    /// Whether to perform the operation in-place.
    inplace: bool,
}

impl ReLU {
    /// Create a new ReLU activation function.
    pub fn new() -> Self {
        Self { inplace: false }
    }
    
    /// Create a new in-place ReLU activation function.
    pub fn new_inplace() -> Self {
        Self { inplace: true }
    }
}

impl<T, D, S> crate::nn::Activation<T, D, S> for ReLU
where
    T: Clone + Default + std::cmp::PartialOrd + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32>,
{
    fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> {
        if self.inplace {
            relu_(input.clone())
        } else {
            relu(input)
        }
    }
    
    fn backward(
        &self,
        _input: &Tensor<T, D, S>,
        output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        relu_grad(output, grad_output)
    }
}

/// Leaky ReLU activation function.
///
/// Formula: `f(x) = x if x >= 0 else negative_slope * x`
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    /// The slope for negative inputs.
    negative_slope: f32,
    /// Whether to perform the operation in-place.
    inplace: bool,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
            inplace: false,
        }
    }
}

impl LeakyReLU {
    /// Create a new LeakyReLU activation function with the given negative slope.
    pub fn new(negative_slope: f32) -> Self {
        Self {
            negative_slope,
            inplace: false,
        }
    }
    
    /// Create a new in-place LeakyReLU activation function with the given negative slope.
    pub fn new_inplace(negative_slope: f32) -> Self {
        Self {
            negative_slope,
            inplace: true,
        }
    }
}

impl<T, D, S> crate::nn::Activation<T, D, S> for LeakyReLU
where
    T: Clone + Default + std::cmp::PartialOrd + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32>,
{
    fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> {
        if self.inplace {
            leaky_relu_(input.clone(), self.negative_slope.into())
        } else {
            leaky_relu(input, self.negative_slope.into())
        }
    }
    
    fn backward(
        &self,
        input: &Tensor<T, D, S>,
        _output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        leaky_relu_grad(input, grad_output, self.negative_slope.into())
    }
}

/// ELU (Exponential Linear Unit) activation function.
///
/// Formula: `f(x) = x if x >= 0 else alpha * (exp(x) - 1)`
#[derive(Debug, Clone, Copy)]
pub struct ELU {
    /// The alpha value for negative inputs.
    alpha: f32,
    /// Whether to perform the operation in-place.
    inplace: bool,
}

impl Default for ELU {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            inplace: false,
        }
    }
}

impl ELU {
    /// Create a new ELU activation function with the given alpha.
    pub fn new(alpha: f32) -> Self {
        Self { alpha, inplace: false }
    }
    
    /// Create a new in-place ELU activation function with the given alpha.
    pub fn new_inplace(alpha: f32) -> Self {
        Self { alpha, inplace: true }
    }
}

impl<T, D, S> crate::nn::Activation<T, D, S> for ELU
where
    T: Clone + Default + std::cmp::PartialOrd + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32>,
{
    fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> {
        if self.inplace {
            elu_(input.clone(), self.alpha.into())
        } else {
            elu(input, self.alpha.into())
        }
    }
    
    fn backward(
        &self,
        input: &Tensor<T, D, S>,
        output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        elu_grad(input, output, grad_output, self.alpha.into())
    }
}

/// GELU (Gaussian Error Linear Unit) activation function.
///
/// Formula: `f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
#[derive(Debug, Clone, Copy, Default)]
pub struct GELU;

impl<T, D, S> crate::nn::Activation<T, D, S> for GELU
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> {
        gelu(input)
    }
    
    fn backward(
        &self,
        input: &Tensor<T, D, S>,
        output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        gelu_grad(input, output, grad_output)
    }
}

/// Sigmoid activation function.
///
/// Formula: `f(x) = 1 / (1 + exp(-x))`
#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid {
    /// Whether to perform the operation in-place.
    inplace: bool,
}

impl Sigmoid {
    /// Create a new Sigmoid activation function.
    pub fn new() -> Self {
        Self { inplace: false }
    }
    
    /// Create a new in-place Sigmoid activation function.
    pub fn new_inplace() -> Self {
        Self { inplace: true }
    }
}

impl<T, D, S> crate::nn::Activation<T, D, S> for Sigmoid
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> {
        if self.inplace {
            sigmoid_(input.clone())
        } else {
            sigmoid(input)
        }
    }
    
    fn backward(
        &self,
        _input: &Tensor<T, D, S>,
        output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        sigmoid_grad(output, grad_output)
    }
}

/// Tanh (Hyperbolic Tangent) activation function.
///
/// Formula: `f(x) = tanh(x)`
#[derive(Debug, Clone, Copy, Default)]
pub struct Tanh {
    /// Whether to perform the operation in-place.
    inplace: bool,
}

impl Tanh {
    /// Create a new Tanh activation function.
    pub fn new() -> Self {
        Self { inplace: false }
    }
    
    /// Create a new in-place Tanh activation function.
    pub fn new_inplace() -> Self {
        Self { inplace: true }
    }
}

impl<T, D, S> crate::nn::Activation<T, D, S> for Tanh
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> {
        if self.inplace {
            tanh_(input.clone())
        } else {
            tanh(input)
        }
    }
    
    fn backward(
        &self,
        _input: &Tensor<T, D, S>,
        output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        tanh_grad(output, grad_output)
    }
}

/// Softmax activation function.
///
/// Formula: `f(x_i) = exp(x_i) / sum(exp(x_j) for j in dim)`
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    /// The dimension along which to apply the softmax.
    dim: usize,
}

impl Default for Softmax {
    fn default() -> Self {
        Self { dim: 1 }
    }
}

impl Softmax {
    /// Create a new Softmax activation function that normalizes along the given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl<T, D, S> crate::nn::Activation<T, D, S> for Softmax
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> {
        softmax(input, self.dim)
    }
    
    fn backward(
        &self,
        _input: &Tensor<T, D, S>,
        output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        softmax_grad(output, grad_output, self.dim)
    }
}

/// LogSoftmax activation function.
///
/// Formula: `f(x_i) = log(exp(x_i) / sum(exp(x_j) for j in dim))`
#[derive(Debug, Clone, Copy)]
pub struct LogSoftmax {
    /// The dimension along which to apply the log-softmax.
    dim: usize,
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self { dim: 1 }
    }
}

impl LogSoftmax {
    /// Create a new LogSoftmax activation function that normalizes along the given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl<T, D, S> crate::nn::Activation<T, D, S> for LogSoftmax
where
    T: Clone + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    T: From<f32>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn forward(&self, input: &Tensor<T, D, S>) -> Result<Tensor<T, D, S>> {
        // log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x), dim=dim, keepdim=True))
        // This is numerically stable and avoids overflow
        let max_val = input.max(Some(self.dim), true)?;
        let exp_x = (input - &max_val)?.exp()?;
        let sum_exp = exp_x.sum(Some(self.dim), true)?;
        let log_sum_exp = sum_exp.ln()?;
        let result = input - &(max_val + log_sum_exp)?;
        Ok(result)
    }
    
    fn backward(
        &self,
        _input: &Tensor<T, D, S>,
        output: &Tensor<T, D, S>,
        grad_output: &Tensor<T, D, S>,
    ) -> Result<Tensor<T, D, S>> {
        // The gradient of log_softmax is grad_output - sum(grad_output, dim=dim, keepdim=True) * exp(output)
        let sum_grad = grad_output.sum(Some(self.dim), true)?;
        let grad = grad_output - &(sum_grad * &output.exp()?)?;
        Ok(grad)
    }
}

/// A module that applies an activation function to the input.
/// This is a convenience wrapper around activation functions to make them usable as layers.
#[derive(Debug, Clone)]
pub struct ActivationFunc<A> {
    /// The activation function to apply.
    func: A,
}

impl<A> ActivationFunc<A> {
    /// Create a new activation function layer.
    pub fn new(func: A) -> Self {
        Self { func }
    }
}

impl<T, D, S, A> crate::nn::Layer<T, D, S> for ActivationFunc<A>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    A: crate::nn::Activation<T, D, S> + 'static,
{
    type Input = D;
    type Output = D;

    fn forward(&self, input: &Tensor<T, Self::Input, S>) -> Result<Tensor<T, Self::Output, S>> {
        self.func.forward(input)
    }
    
    fn backward(
        &self,
        input: &Tensor<T, Self::Input, S>,
        output: &Tensor<T, Self::Output, S>,
        grad_output: &Tensor<T, Self::Output, S>,
    ) -> Result<(
        Tensor<T, Self::Input, S>,
        Option<Vec<Tensor<T, crate::dimension::DynamicDim, S>>>,
    )> {
        let grad_input = self.func.backward(input, output, grad_output)?;
        Ok((grad_input, None))
    }
    
    fn parameters(&self) -> Vec<&dyn std::any::Any> {
        // Activation functions have no trainable parameters
        Vec::new()
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
    fn test_relu() {
        let relu = ReLU::new();
        let input = tensor!([-1.0, 0.0, 1.0, 2.0]);
        let output = relu.forward(&input).unwrap();
        assert_eq!(output.data(), &[0.0, 0.0, 1.0, 2.0]);
        
        let grad_output = tensor!([1.0, 1.0, 1.0, 1.0]);
        let grad_input = relu.backward(&input, &output, &grad_output).unwrap();
        assert_eq!(grad_input.data(), &[0.0, 0.0, 1.0, 1.0]);
    }
    
    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid::new();
        let input = tensor!([0.0]);
        let output = sigmoid.forward(&input).unwrap();
        assert_relative_eq!(output[[0]], 0.5, epsilon = 1e-6);
        
        let grad_output = tensor!([1.0]);
        let grad_input = sigmoid.backward(&input, &output, &grad_output).unwrap();
        assert_relative_eq!(grad_input[[0]], 0.25, epsilon = 1e-6);
    }
    
    #[test]
    fn test_softmax() {
        let softmax = Softmax::new(0);
        let input = tensor!([1.0, 2.0, 3.0]);
        let output = softmax.forward(&input).unwrap();
        
        // Check that the output sums to 1
        let sum: f32 = output.data().iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        
        // Check that the output is normalized
        assert!(output[[0]] < output[[1]]);
        assert!(output[[1]] < output[[2]]);
    }
}
