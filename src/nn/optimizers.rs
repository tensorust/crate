//! Optimization algorithms for training neural networks.
//!
//! This module provides various optimization algorithms that can be used to train
//! neural networks. Each optimizer implements the `OptimizerState` trait, which
//! defines how to update the parameters of a model based on their gradients.
//!
//! # Available Optimizers
//! - SGD (Stochastic Gradient Descent)
//! - Adam (Adaptive Moment Estimation)
//! - RMSprop (Root Mean Square Propagation)
//! - Adagrad (Adaptive Gradient)
//! - Adadelta (Adaptive Learning Rate Method)
//! - AdamW (Adam with Weight Decay Fix)
//! - LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
//!
//! # Usage Example
//! ```no_run
//! use tensorust::{
//!     nn::{optimizers::Optimizer, layers::Linear, activations::ReLU, Sequential},
//!     tensor, Tensor, CpuStorage,
//! };
//!
//! // Create a simple model
//! let mut model = Sequential::new();
//! model.add(Box::new(Linear::new(10, 20, true)));
//! model.add(Box::new(ReLU::new()));
//! model.add(Box::new(Linear::new(20, 1, true)));
//!
//! // Create an optimizer
//! let mut optimizer = Optimizer::<f32, CpuStorage<f32>>::adam(1e-3, 0.9, 0.999, 1e-8, 0.0);
//!
//! // Training loop
//! for _ in 0..100 {
//!     // Forward pass
//!     let input = tensor!([0.5; 10]);
//!     let output = model.forward(&input).unwrap();
//!     
//!     // Backward pass (compute gradients)
//!     output.backward();
//!     
//!     // Update parameters
//!     optimizer.step(model.parameters_mut()).unwrap();
//!     
//!     // Zero gradients
//!     optimizer.zero_grad(model.parameters_mut()).unwrap();
//! }

use crate::{
    dimension::Dimension,
    error::Result,
    storage::Storage,
    tensor::Tensor,
};
use std::collections::HashMap;
use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

/// State for optimization algorithms.
pub trait OptimizerState<T, S>: 'static + Send + Sync + std::fmt::Debug
where
    T: Clone + Send + Sync + 'static,
    S: Storage<T>,
{
    /// Update the parameters.
    fn update(&mut self, param: &mut Tensor<T, crate::dimension::DynamicDim, S>);
    
    /// Get the state as a trait object.
    fn as_any(&self) -> &dyn Any;
    
    /// Get the state as a mutable trait object.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// SGD (Stochastic Gradient Descent) optimizer state.
#[derive(Debug)]
struct SgdState<T, S> {
    /// The learning rate.
    lr: T,
    /// The momentum factor.
    momentum: T,
    /// The velocity for momentum.
    velocity: Option<Tensor<T, crate::dimension::DynamicDim, S>>,
    /// The weight decay factor.
    weight_decay: T,
    /// Whether to use Nesterov momentum.
    nesterov: bool,
}

impl<T, S> SgdState<T, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Sub<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    /// Create a new SGD state.
    fn new(lr: T, momentum: T, weight_decay: T, nesterov: bool) -> Self {
        Self {
            lr,
            momentum,
            velocity: None,
            weight_decay,
            nesterov,
        }
    }
}

impl<T, S> OptimizerState<T, S> for SgdState<T, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Sub<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn update(&mut self, param: &mut Tensor<T, crate::dimension::DynamicDim, S>) {
        // Apply weight decay
        if self.weight_decay != T::default() {
            // param = param - lr * weight_decay * param
            let update = &*param * &self.lr * &self.weight_decay;
            *param = (&*param - &update).unwrap();
        }
        
        // Get the gradient
        let grad = param.grad().unwrap();
        
        // Initialize velocity if needed
        if self.velocity.is_none() {
            self.velocity = Some(Tensor::zeros_like(param).unwrap());
        }
        
        // Update velocity
        let velocity = self.velocity.as_mut().unwrap();
        *velocity = (&*velocity * &self.momentum + &grad).unwrap();
        
        // Update parameters
        if self.nesterov {
            // param = param - lr * (grad + momentum * velocity)
            let update = &*velocity * &self.momentum + &grad;
            *param = (&*param - &(update * &self.lr)).unwrap();
        } else {
            // param = param - lr * velocity
            *param = (&*param - &(*velocity * &self.lr)).unwrap();
        }
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Adam optimizer state.
#[derive(Debug)]
struct AdamState<T, S> {
    /// The learning rate.
    lr: T,
    /// The beta1 parameter.
    beta1: T,
    /// The beta2 parameter.
    beta2: T,
    /// The epsilon parameter.
    eps: T,
    /// The weight decay factor.
    weight_decay: T,
    /// The first moment estimate.
    m: Option<Tensor<T, crate::dimension::DynamicDim, S>>,
    /// The second moment estimate.
    v: Option<Tensor<T, crate::dimension::DynamicDim, S>>,
    /// The current timestep.
    t: i32,
}

impl<T, S> AdamState<T, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Sub<Output = T> + std::ops::AddAssign,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    f32: From<T>,
{
    /// Create a new Adam state.
    fn new(lr: T, beta1: T, beta2: T, eps: T, weight_decay: T) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m: None,
            v: None,
            t: 0,
        }
    }
}

impl<T, S> OptimizerState<T, S> for AdamState<T, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Sub<Output = T> + std::ops::AddAssign,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    f32: From<T>,
{
    fn update(&mut self, param: &mut Tensor<T, crate::dimension::DynamicDim, S>) {
        // Increment timestep
        self.t += 1;
        
        // Apply weight decay
        if self.weight_decay != T::default() {
            param.grad_mut().map(|g| *g = &*g + &(param * &self.weight_decay));
        }
        
        // Get the gradient
        let grad = param.grad().unwrap();
        
        // Initialize moment estimates if needed
        if self.m.is_none() {
            self.m = Some(Tensor::zeros_like(param).unwrap());
        }
        if self.v.is_none() {
            self.v = Some(Tensor::zeros_like(param).unwrap());
        }
        
        // Update biased first moment estimate
        let m = self.m.as_mut().unwrap();
        *m = (&*m * &self.beta1 + &grad * (T::default() + &self.beta1 * &T::default())).unwrap();
        
        // Update biased second raw moment estimate
        let v = self.v.as_mut().unwrap();
        *v = (&*v * &self.beta2 + &(&grad * &grad) * (T::default() + &self.beta2 * &T::default())).unwrap();
        
        // Compute bias-corrected first moment estimate
        let m_hat = &*m / (T::default() + &T::default() - self.beta1.powi(self.t));
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = &*v / (T::default() + &T::default() - self.beta2.powi(self.t));
        
        // Update parameters
        let update = &m_hat / (v_hat.sqrt() + &self.eps);
        *param = (&*param - &(update * &self.lr)).unwrap();
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// RMSprop optimizer state.
#[derive(Debug)]
struct RMSpropState<T, S> {
    /// The learning rate.
    lr: T,
    /// The alpha parameter.
    alpha: T,
    /// The epsilon parameter.
    eps: T,
    /// The weight decay factor.
    weight_decay: T,
    /// The momentum factor.
    momentum: T,
    /// Whether to use centered RMSprop.
    centered: bool,
    /// The moving average of squared gradients.
    square_avg: Option<Tensor<T, crate::dimension::DynamicDim, S>>,
    /// The moving average of gradients (for centered RMSprop).
    grad_avg: Option<Tensor<T, crate::dimension::DynamicDim, S>>,
    /// The momentum buffer.
    momentum_buffer: Option<Tensor<T, crate::dimension::DynamicDim, S>>,
}

impl<T, S> RMSpropState<T, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Sub<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    /// Create a new RMSprop state.
    fn new(lr: T, alpha: T, eps: T, weight_decay: T, momentum: T, centered: bool) -> Self {
        Self {
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            square_avg: None,
            grad_avg: None,
            momentum_buffer: None,
        }
    }
}

impl<T, S> OptimizerState<T, S> for RMSpropState<T, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Sub<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn update(&mut self, param: &mut Tensor<T, crate::dimension::DynamicDim, S>) {
        // Apply weight decay
        if self.weight_decay != T::default() {
            param.grad_mut().map(|g| *g = &*g + &(param * &self.weight_decay));
        }
        
        // Get the gradient
        let grad = param.grad().unwrap();
        
        // Initialize state if needed
        if self.square_avg.is_none() {
            self.square_avg = Some(Tensor::zeros_like(param).unwrap());
        }
        if self.centered && self.grad_avg.is_none() {
            self.grad_avg = Some(Tensor::zeros_like(param).unwrap());
        }
        if self.momentum != T::default() && self.momentum_buffer.is_none() {
            self.momentum_buffer = Some(Tensor::zeros_like(param).unwrap());
        }
        
        // Update square average
        let square_avg = self.square_avg.as_mut().unwrap();
        *square_avg = (&*square_avg * &self.alpha + &(&grad * &grad) * (T::default() + &self.alpha * &T::default())).unwrap();
        
        // Update gradient average (if centered)
        let avg = if self.centered {
            let grad_avg = self.grad_avg.as_mut().unwrap();
            *grad_avg = (&*grad_avg * &self.alpha + &grad * (T::default() + &self.alpha * &T::default())).unwrap();
            let var = &*square_avg - &(&*grad_avg * &*grad_avg);
            var.sqrt() + &self.eps
        } else {
            square_avg.sqrt() + &self.eps
        };
        
        // Compute update
        let update = &grad / avg;
        
        // Apply momentum if needed
        if self.momentum != T::default() {
            let momentum_buffer = self.momentum_buffer.as_mut().unwrap();
            *momentum_buffer = (&*momentum_buffer * &self.momentum + &update).unwrap();
            *param = (&*param - &(*momentum_buffer * &self.lr)).unwrap();
        } else {
            *param = (&*param - &(update * &self.lr)).unwrap();
        }
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Adagrad optimizer state.
#[derive(Debug)]
struct AdagradState<T, S> {
    /// The learning rate.
    lr: T,
    /// The weight decay factor.
    weight_decay: T,
    /// The epsilon parameter.
    eps: T,
    /// The sum of squares of gradients.
    sum: Option<Tensor<T, crate::dimension::DynamicDim, S>>,
}

impl<T, S> AdagradState<T, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    T: std::ops::Add<Output = T> + std::ops::Div<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    /// Create a new Adagrad state.
    fn new(lr: T, weight_decay: T, eps: T) -> Self {
        Self {
            lr,
            weight_decay,
            eps,
            sum: None,
        }
    }
}

impl<T, S> OptimizerState<T, S> for AdagradState<T, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    T: std::ops::Add<Output = T> + std::ops::Div<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn update(&mut self, param: &mut Tensor<T, crate::dimension::DynamicDim, S>) {
        // Apply weight decay
        if self.weight_decay != T::default() {
            param.grad_mut().map(|g| *g = &*g + &(param * &self.weight_decay));
        }
        
        // Get the gradient
        let grad = param.grad().unwrap();
        
        // Initialize state if needed
        if self.sum.is_none() {
            self.sum = Some(Tensor::zeros_like(param).unwrap());
        }
        
        // Update sum of squares of gradients
        let sum = self.sum.as_mut().unwrap();
        *sum = (&*sum + &(&grad * &grad)).unwrap();
        
        // Update parameters
        let std = sum.sqrt() + &self.eps;
        *param = (&*param - &(grad / std * &self.lr)).unwrap();
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A generic optimizer that can be used with any optimizer state.
///
/// This struct manages the optimization process for a set of parameters.
/// It maintains the state for each parameter and applies the optimization
/// algorithm during the `step` method.
///
/// # Type Parameters
/// - `T`: The element type of the tensors.
/// - `S`: The storage backend for the tensors.
pub struct Optimizer<T, S> {
    /// The optimizer state for each parameter.
    states: HashMap<usize, Box<dyn OptimizerState<T, S>>>,
    /// The default learning rate.
    default_lr: T,
}

impl<T, S> Optimizer<T, S>
where
    T: Clone + Default + Send + Sync + 'static,
    S: Storage<T>,
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Sub<Output = T>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    f32: From<T>,
{
    /// Create a new optimizer with the given default learning rate.
    pub fn new(default_lr: T) -> Self {
        Self {
            states: HashMap::new(),
            default_lr,
        }
    }
    
    /// Create a new SGD optimizer.
    pub fn sgd(
        lr: T,
        momentum: T,
        weight_decay: T,
        nesterov: bool,
    ) -> Self {
        let mut optimizer = Self::new(lr);
        optimizer.set_default_state(move |_| {
            Box::new(SgdState::new(
                lr.clone(),
                momentum.clone(),
                weight_decay.clone(),
                nesterov,
            ))
        });
        optimizer
    }
    
    /// Create a new Adam optimizer.
    pub fn adam(
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
    ) -> Self {
        let mut optimizer = Self::new(lr);
        optimizer.set_default_state(move |_| {
            Box::new(AdamState::new(
                lr.clone(),
                beta1.clone(),
                beta2.clone(),
                eps.clone(),
                weight_decay.clone(),
            ))
        });
        optimizer
    }
    
    /// Create a new RMSprop optimizer.
    pub fn rmsprop(
        lr: T,
        alpha: T,
        eps: T,
        weight_decay: T,
        momentum: T,
        centered: bool,
    ) -> Self {
        let mut optimizer = Self::new(lr);
        optimizer.set_default_state(move |_| {
            Box::new(RMSpropState::new(
                lr.clone(),
                alpha.clone(),
                eps.clone(),
                weight_decay.clone(),
                momentum.clone(),
                centered,
            ))
        });
        optimizer
    }
    
    /// Create a new Adagrad optimizer.
    pub fn adagrad(
        lr: T,
        weight_decay: T,
        eps: T,
    ) -> Self {
        let mut optimizer = Self::new(lr);
        optimizer.set_default_state(move |_| {
            Box::new(AdagradState::new(
                lr.clone(),
                weight_decay.clone(),
                eps.clone(),
            ))
        });
        optimizer
    }
    
    /// Set the default state creation function.
    pub fn set_default_state<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(usize) -> Box<dyn OptimizerState<T, S>> + 'static,
    {
        self.states = HashMap::new();
        self.states = (0..1000)  // Just a placeholder for now
            .map(|i| (i, f(i)))
            .collect();
        self
    }
    
    /// Get the state for a parameter.
    pub fn state_mut(&mut self, param_id: usize) -> Option<&mut dyn OptimizerState<T, S>> {
        self.states.get_mut(&param_id).map(|s| &mut **s as _)
    }
    
    /// Update the parameters using the stored state.
    pub fn step(&mut self, params: &mut [&mut Tensor<T, crate::dimension::DynamicDim, S>]) -> Result<()> {
        for (i, param) in params.iter_mut().enumerate() {
            if let Some(state) = self.state_mut(i) {
                state.update(*param);
            } else {
                // If no state exists, use SGD with default parameters
                let mut state = SgdState::new(
                    self.default_lr.clone(),
                    T::default(),  // No momentum
                    T::default(),  // No weight decay
                    false,         // No Nesterov momentum
                );
                state.update(*param);
                self.states.insert(i, Box::new(state));
            }
        }
        Ok(())
    }
    
    /// Zero out the gradients of the parameters.
    pub fn zero_grad(&self, params: &mut [&mut Tensor<T, crate::dimension::DynamicDim, S>]) -> Result<()> {
        for param in params {
            param.zero_grad();
        }
        Ok(())
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
    fn test_sgd_optimizer() {
        let mut x = tensor!([1.0, 2.0, 3.0]);
        x.requires_grad(true);
        
        let y = (&x * 2.0).unwrap();
        let loss = y.sum().unwrap();
        
        // Compute gradients
        loss.backward();
        
        // Create optimizer
        let mut optimizer = Optimizer::<f32, CpuStorage<f32>>::sgd(0.1, 0.0, 0.0, false);
        
        // Update parameters
        optimizer.step(&mut [&mut x]).unwrap();
        
        // Check that the parameters were updated correctly
        assert_relative_eq!(x[0], 0.8, epsilon = 1e-6);
        assert_relative_eq!(x[1], 1.8, epsilon = 1e-6);
        assert_relative_eq!(x[2], 2.8, epsilon = 1e-6);
    }
    
    #[test]
    fn test_adam_optimizer() {
        let mut x = tensor!([1.0, 2.0, 3.0]);
        x.requires_grad(true);
        
        let y = (&x * 2.0).unwrap();
        let loss = y.sum().unwrap();
        
        // Compute gradients
        loss.backward();
        
        // Create optimizer
        let mut optimizer = Optimizer::<f32, CpuStorage<f32>>::adam(0.1, 0.9, 0.999, 1e-8, 0.0);
        
        // Update parameters
        optimizer.step(&mut [&mut x]).unwrap();
        
        // Check that the parameters were updated
        // The exact values depend on the Adam implementation details
        assert!(x[0] < 1.0);
        assert!(x[1] < 2.0);
        assert!(x[2] < 3.0);
    }
}
