//! Weight initialization functions for neural networks.
//!
//! This module provides various initialization strategies for the weights of neural network layers.
//! Proper initialization is crucial for training deep neural networks effectively.
//!
//! # Available Initialization Strategies
//! - **Basic Initializations**
//!   - `uniform_init`: Initialize with values from a uniform distribution
//!   - `normal_init`: Initialize with values from a normal distribution
//!   - `truncated_normal_init`: Initialize with values from a truncated normal distribution
//!
//! - **Advanced Initializations**
//!   - `kaiming_uniform`/`kaiming_normal`: He initialization (good for ReLU networks)
//!   - `xavier_uniform`/`xavier_normal`: Glorot initialization (good for tanh/sigmoid networks)
//!   - `orthogonal_init`: Orthogonal matrix initialization (good for RNNs)
//!   - `lecun_normal`/`lecun_uniform`: LeCun initialization (variant of Kaiming initialization)
//!   - `variance_scaling_init`: General variance scaling initialization
//!
//! # Usage Example
//! ```no_run
//! use tensorust::{
//!     nn::{
//!         init::{kaiming_uniform, normal_init, Initializer, KaimingNormalInitializer},
//!         layers::Linear,
//!     },
//!     tensor, Tensor, CpuStorage,
//! };
//!
//! // Initialize a tensor with Kaiming uniform initialization
//! let mut weights = tensor!([0.0; 100]);
//! kaiming_uniform(&mut weights, 100, 2.0f32.sqrt()).unwrap();
//!
//! // Or use the Initializer trait for more flexibility
//! let mut linear = Linear::new(100, 50, true);
//! let initializer = KaimingNormalInitializer::new("fan_in", "relu");
//! initializer.initialize(linear.weights_mut()).unwrap();
//! ```

use crate::{
    dimension::Dimension,
    tensor::Tensor,
    storage::Storage,
    error::Result,
};
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use rand_distr::StandardNormal;
use std::f32::consts::{PI, SQRT_2};
use num_traits::{Float, FromPrimitive, ToPrimitive};

/// Initializes a tensor with values from a uniform distribution.
///
/// This function fills the tensor with values sampled from a uniform distribution
/// in the range `[low, high)`.
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize.
/// * `low` - The lower bound of the uniform distribution (inclusive).
/// * `high` - The upper bound of the uniform distribution (exclusive).
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::uniform_init;
/// let mut x = tensor!([0.0; 5]);
/// uniform_init(&mut x, -1.0, 1.0).unwrap();
/// for &val in x.data() {
///     assert!(val >= -1.0 && val < 1.0);
/// }
/// ```
pub fn uniform_init<T, D, S>(
    tensor: &mut Tensor<T, D, S>,
    low: f32,
    high: f32,
) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(low, high);
    
    tensor.map_mut(|_| {
        let val: f32 = uniform.sample(&mut rng);
        val.into()
    })?;
    
    Ok(())
}

/// Initializes a tensor with values from a normal distribution.
///
/// This function fills the tensor with values sampled from a normal (Gaussian)
/// distribution with the specified mean and standard deviation.
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize.
/// * `mean` - The mean of the normal distribution.
/// * `stddev` - The standard deviation of the normal distribution.
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::normal_init;
/// let mut x = tensor!([0.0; 1000]);
/// normal_init(&mut x, 0.0, 0.1).unwrap();
/// ```
pub fn normal_init<T, D, S>(
    tensor: &mut Tensor<T, D, S>,
    mean: f32,
    stddev: f32,
) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    let mut rng = rand::thread_rng();
    
    tensor.map_mut(|_| {
        let val: f32 = rng.sample(StandardNormal) * stddev + mean;
        val.into()
    })?;
    
    Ok(())
}

/// Initializes a tensor with values from a truncated normal distribution.
///
/// This function fills the tensor with values sampled from a normal distribution
/// with the specified mean and standard deviation, but discards and re-samples any
/// values that fall outside the range `[a, b]`.
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize.
/// * `mean` - The mean of the normal distribution.
/// * `stddev` - The standard deviation of the normal distribution.
/// * `a` - The lower bound of the truncation range (inclusive).
/// * `b` - The upper bound of the truncation range (inclusive).
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::truncated_normal_init;
/// let mut x = tensor!([0.0; 1000]);
/// truncated_normal_init(&mut x, 0.0, 1.0, -2.0, 2.0).unwrap();
/// for &val in x.data() {
///     assert!(val >= -2.0 && val <= 2.0);
/// }
/// ```
pub fn truncated_normal_init<T, D, S>(
    tensor: &mut Tensor<T, D, S>,
    mean: f32,
    stddev: f32,
    a: f32,
    b: f32,
) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    let mut rng = rand::thread_rng();
    
    tensor.map_mut(|_| {
        loop {
            let val: f32 = rng.sample(StandardNormal) * stddev + mean;
            if val >= a && val <= b {
                return val.into();
            }
        }
    })?;
    
    Ok::<(), crate::error::TensorustError>(())
}

/// Initializes a tensor with the Kaiming (He) uniform initialization.
///
/// This initialization is designed to keep the scale of the gradients roughly the same
/// in all layers. It's particularly well-suited for layers with ReLU activation functions.
///
/// The weights are sampled from a uniform distribution in the range `[-bound, bound]` where:
/// ```text
/// bound = gain * sqrt(3 / fan_in)
/// ```
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize.
/// * `fan_in` - The number of input units in the weight tensor.
/// * `gain` - A scaling factor to apply to the standard deviation.
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # References
/// - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::kaiming_uniform;
/// let mut x = tensor!([[0.0; 100]; 50]); // 50x100 weight matrix
/// kaiming_uniform(&mut x, 100, 2.0f32.sqrt()).unwrap();
/// ```
pub fn kaiming_uniform<T, D, S>(
    tensor: &mut Tensor<T, D, S>,
    fan_in: usize,
    gain: f32,
) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    let bound = gain * (3.0 / fan_in as f32).sqrt();
    uniform_init(tensor, -bound, bound)
}

/// Initializes a tensor with the Kaiming (He) normal initialization.
///
/// This initialization is designed to keep the scale of the gradients roughly the same
/// in all layers. It's particularly well-suited for layers with ReLU activation functions.
///
/// The weights are sampled from a normal distribution with mean 0 and standard deviation:
/// ```text
/// stddev = gain / sqrt(fan_in)
/// ```
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize.
/// * `fan_in` - The number of input units in the weight tensor.
/// * `gain` - A scaling factor to apply to the standard deviation.
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # References
/// - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::kaiming_normal;
/// let mut x = tensor!([[0.0; 100]; 50]); // 50x100 weight matrix
/// kaiming_normal(&mut x, 100, 2.0f32.sqrt()).unwrap();
/// ```
pub fn kaiming_normal<T, D, S>(
    tensor: &mut Tensor<T, D, S>,
    fan_in: usize,
    gain: f32,
) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    let std = gain / (fan_in as f32).sqrt();
    normal_init(tensor, 0.0, std)
}

/// Initializes a tensor with the Xavier (Glorot) uniform initialization.
///
/// This initialization is designed to keep the scale of the gradients roughly the same
/// in all layers. It's particularly well-suited for layers with tanh or sigmoid activation functions.
///
/// The weights are sampled from a uniform distribution in the range `[-bound, bound]` where:
/// ```text
/// bound = gain * sqrt(6 / (fan_in + fan_out))
/// ```
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize.
/// * `fan_in` - The number of input units in the weight tensor.
/// * `fan_out` - The number of output units in the weight tensor.
/// * `gain` - A scaling factor to apply to the standard deviation.
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # References
/// - [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::xavier_uniform;
/// let mut x = tensor!([[0.0; 100]; 50]); // 50x100 weight matrix
/// xavier_uniform(&mut x, 100, 50, 1.0).unwrap();
/// ```
pub fn xavier_uniform<T, D, S>(
    tensor: &mut Tensor<T, D, S>,
    fan_in: usize,
    fan_out: usize,
    gain: f32,
) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    let bound = gain * (6.0 / (fan_in + fan_out) as f32).sqrt();
    uniform_init(tensor, -bound, bound)
}

/// Initializes a tensor with the Xavier (Glorot) normal initialization.
///
/// This initialization is designed to keep the scale of the gradients roughly the same
/// in all layers. It's particularly well-suited for layers with tanh or sigmoid activation functions.
///
/// The weights are sampled from a normal distribution with mean 0 and standard deviation:
/// ```text
/// stddev = gain * sqrt(2 / (fan_in + fan_out))
/// ```
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize.
/// * `fan_in` - The number of input units in the weight tensor.
/// * `fan_out` - The number of output units in the weight tensor.
/// * `gain` - A scaling factor to apply to the standard deviation.
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # References
/// - [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::xavier_normal;
/// let mut x = tensor!([[0.0; 100]; 50]); // 50x100 weight matrix
/// xavier_normal(&mut x, 100, 50, 1.0).unwrap();
/// ```
pub fn xavier_normal<T, D, S>(
    tensor: &mut Tensor<T, D, S>,
    fan_in: usize,
    fan_out: usize,
    gain: f32,
) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    let std = gain * (2.0 / (fan_in + fan_out) as f32).sqrt();
    normal_init(tensor, 0.0, std)
}

/// Initializes a tensor with orthogonal initialization.
///
/// This initialization generates a random orthogonal matrix, which can help with training
/// deep networks by preventing vanishing/exploding gradients. It's particularly useful for
/// recurrent neural networks (RNNs).
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize. Must have at least 2 dimensions.
/// * `gain` - A scaling factor to apply to the orthogonal matrix.
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # References
/// - [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120)
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::orthogonal_init;
/// let mut x = tensor!([[0.0; 100]; 100]); // 100x100 weight matrix
/// orthogonal_init(&mut x, 1.0).unwrap();
/// ```
pub fn orthogonal_init<T, D, S>(
    tensor: &mut Tensor<T, D, S>,
    gain: f32,
) -> Result<()>
where
    T: Clone + Default + Send + Sync + 'static + std::ops::Neg<Output = T> + From<f32>,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    let shape = tensor.shape().as_slice();
    if shape.len() < 2 {
        return Err(crate::error::TensorustError::invalid_shape(
            "Orthogonal initialization requires at least 2 dimensions",
        ));
    }
    
    let rows = shape[0];
    let cols = shape[1..].iter().product();
    let shape = [rows, cols];
    
    // Create a random matrix with the appropriate shape
    let mut rng = rand::thread_rng();
    let mut matrix = vec![0.0; rows * cols];
    
    // Fill with random normal values
    for i in 0..(rows * cols) {
        matrix[i] = rng.sample(StandardNormal);
    }
    
    // Compute the QR decomposition
    let (q, _) = qr_decomposition(&matrix, rows, cols);
    
    // Flatten the tensor and fill with the orthogonal matrix
    let flat_tensor = tensor.reshape_mut(shape.as_ref())?;
    
    // Scale by the gain
    for (i, val) in q.into_iter().enumerate() {
        flat_tensor.data_mut()[i] = (val * gain).into();
    }
    
    Ok(())
}

/// Performs QR decomposition on a matrix using Householder transformations.
///
/// # Arguments
///
/// * `matrix` - The matrix to decompose, in row-major order.
/// * `rows` - The number of rows in the matrix.
/// * `cols` - The number of columns in the matrix.
///
/// # Returns
///
/// A tuple containing the Q and R matrices in row-major order.
fn qr_decomposition(matrix: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut r = matrix.to_vec();
    let mut q = vec![0.0; rows * rows];
    
    // Initialize Q as identity
    for i in 0..rows {
        q[i * rows + i] = 1.0;
    }
    
    let n = rows.min(cols);
    
    for i in 0..n {
        // Compute the Householder vector
        let mut norm = 0.0;
        for k in i..rows {
            norm += r[k * cols + i] * r[k * cols + i];
        }
        norm = norm.sqrt();
        
        let alpha = -r[i * cols + i].signum() * norm;
        let r_ii = r[i * cols + i];
        let u1 = r_ii - alpha;
        
        // Store the Householder vector in the lower part of R
        for k in (i + 1)..rows {
            r[k * cols + i] = r[k * cols + i] / u1;
        }
        r[i * cols + i] = alpha;
        
        // Apply the Householder transformation to the remaining columns of R
        for j in (i + 1)..cols {
            let mut dot = 0.0;
            for k in i..rows {
                dot += r[k * cols + i] * r[k * cols + j];
            }
            
            for k in i..rows {
                r[k * cols + j] -= dot * r[k * cols + i];
            }
        }
        
        // Apply the Householder transformation to Q
        for j in 0..rows {
            let mut dot = 0.0;
            for k in i..rows {
                dot += q[j * rows + k] * r[k * cols + i];
            }
            
            for k in i..rows {
                q[j * rows + k] -= dot * r[k * cols + i];
            }
        }
    }
    
    (q, r)
}

/// A trait for initializing tensors with various initialization schemes.
///
/// This trait provides a unified interface for different initialization strategies,
/// making it easy to switch between them without changing the rest of the code.
///
/// # Type Parameters
/// - `T`: The element type of the tensor.
/// - `D`: The dimension type of the tensor.
/// - `S`: The storage backend for the tensor.
pub trait Initializer<T, D, S>
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Initializes the tensor with the specified initialization scheme.
    ///
    /// # Arguments
    /// * `tensor` - The tensor to initialize.
    ///
    /// # Returns
    /// `Ok(())` if the initialization was successful, otherwise an error.
    ///
    /// # Example
    /// ```
    /// # use tensorust::tensor;
    /// # use tensorust::nn::init::{Initializer, KaimingNormalInitializer};
    /// let mut x = tensor!([[0.0; 100]; 50]);
    /// let initializer = KaimingNormalInitializer::new("fan_in", "relu");
    /// initializer.initialize(&mut x).unwrap();
    /// ```
    fn initialize(&self, tensor: &mut Tensor<T, D, S>) -> Result<()>;
}

/// A uniform initializer.
///
/// This initializer fills the tensor with values sampled from a uniform distribution
/// in the range `[low, high)`.
pub struct UniformInitializer {
    /// The lower bound of the uniform distribution (inclusive).
    low: f32,
    /// The upper bound of the uniform distribution (exclusive).
    high: f32,
}

impl UniformInitializer {
    /// Creates a new uniform initializer.
    ///
    /// # Arguments
    /// * `low` - The lower bound of the uniform distribution (inclusive).
    /// * `high` - The upper bound of the uniform distribution (exclusive).
    ///
    /// # Panics
    /// Panics if `low` is not less than `high`.
    ///
    /// # Example
    /// ```
    /// # use tensorust::nn::init::UniformInitializer;
    /// let initializer = UniformInitializer::new(-0.1, 0.1);
    /// ```
    pub fn new(low: f32, high: f32) -> Self {
        assert!(low < high, "lower bound must be less than upper bound");
        Self { low, high }
    }
}

impl<T, D, S> Initializer<T, D, S> for UniformInitializer
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    fn initialize(&self, tensor: &mut Tensor<T, D, S>) -> Result<()> {
        uniform_init(tensor, self.low, self.high)
    }
}

/// A normal (Gaussian) initializer.
///
/// This initializer fills the tensor with values sampled from a normal distribution
/// with the specified mean and standard deviation.
pub struct NormalInitializer {
    /// The mean of the normal distribution.
    mean: f32,
    /// The standard deviation of the normal distribution.
    stddev: f32,
}

impl NormalInitializer {
    /// Creates a new normal initializer.
    ///
    /// # Arguments
    /// * `mean` - The mean of the normal distribution.
    /// * `stddev` - The standard deviation of the normal distribution.
    ///
    /// # Panics
    /// Panics if `stddev` is negative.
    ///
    /// # Example
    /// ```
    /// # use tensorust::nn::init::NormalInitializer;
    /// let initializer = NormalInitializer::new(0.0, 0.01);
    /// ```
    pub fn new(mean: f32, stddev: f32) -> Self {
        assert!(stddev >= 0.0, "standard deviation must be non-negative");
        Self { mean, stddev }
    }
}

impl<T, D, S> Initializer<T, D, S> for NormalInitializer
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    fn initialize(&self, tensor: &mut Tensor<T, D, S>) -> Result<()> {
        normal_init(tensor, self.mean, self.stddev)
    }
}

/// A Kaiming (He) uniform initializer.
///
/// This initializer is designed to keep the scale of the gradients roughly the same
/// in all layers. It's particularly well-suited for layers with ReLU activation functions.
///
/// The weights are sampled from a uniform distribution in the range `[-bound, bound]` where:
/// ```text
/// bound = gain * sqrt(3 / fan_mode)
/// ```
/// where `fan_mode` is either `fan_in` or `fan_out` depending on the `mode` parameter.
pub struct KaimingUniformInitializer {
    /// The scaling factor to apply to the standard deviation.
    gain: f32,
    /// The mode to use for fan calculation ("fan_in" or "fan_out").
    mode: &'static str,
    /// The nonlinearity to use for calculating the gain.
    nonlinearity: &'static str,
}

impl KaimingUniformInitializer {
    /// Creates a new Kaiming uniform initializer.
    ///
    /// # Arguments
    /// * `mode` - Either "fan_in" (default) or "fan_out".
    ///   - "fan_in" preserves the magnitude of the variance of the weights in the forward pass.
    ///   - "fan_out" preserves the magnitudes in the backwards pass.
    /// * `nonlinearity` - The name of the nonlinearity function (e.g., "relu", "leaky_relu").
    ///
    /// # Panics
    /// Panics if `mode` is not "fan_in" or "fan_out".
    ///
    /// # Example
    /// ```
    /// # use tensorust::nn::init::KaimingUniformInitializer;
    /// let initializer = KaimingUniformInitializer::new("fan_in", "relu");
    /// ```
    pub fn new(mode: &'static str, nonlinearity: &'static str) -> Self {
        assert!(
            mode == "fan_in" || mode == "fan_out",
            "mode must be 'fan_in' or 'fan_out'"
        );
        Self {
            gain: calculate_gain(nonlinearity, None),
            mode,
            nonlinearity,
        }
    }
}

impl<T, D, S> Initializer<T, D, S> for KaimingUniformInitializer
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    fn initialize(&self, tensor: &mut Tensor<T, D, S>) -> Result<()> {
        let shape = tensor.shape().as_slice();
        if shape.len() < 2 {
            return Err(crate::error::TensorustError::invalid_shape(
                "Kaiming initialization requires at least 2 dimensions",
            ));
        }
        
        let fan_in = shape[1];
        let fan_out = shape[0];
        let fan = match self.mode {
            "fan_in" => fan_in,
            "fan_out" => fan_out,
            _ => fan_in,
        };
        
        kaiming_uniform(tensor, fan, self.gain)
    }
}

/// A Kaiming (He) normal initializer.
///
/// This initializer is designed to keep the scale of the gradients roughly the same
/// in all layers. It's particularly well-suited for layers with ReLU activation functions.
///
/// The weights are sampled from a normal distribution with mean 0 and standard deviation:
/// ```text
/// stddev = gain / sqrt(fan_mode)
/// ```
/// where `fan_mode` is either `fan_in` or `fan_out` depending on the `mode` parameter.
pub struct KaimingNormalInitializer {
    /// The scaling factor to apply to the standard deviation.
    gain: f32,
    /// The mode to use for fan calculation ("fan_in" or "fan_out").
    mode: &'static str,
    /// The nonlinearity to use for calculating the gain.
    nonlinearity: &'static str,
}

impl KaimingNormalInitializer {
    /// Creates a new Kaiming normal initializer.
    ///
    /// # Arguments
    /// * `mode` - Either "fan_in" (default) or "fan_out".
    ///   - "fan_in" preserves the magnitude of the variance of the weights in the forward pass.
    ///   - "fan_out" preserves the magnitudes in the backwards pass.
    /// * `nonlinearity` - The name of the nonlinearity function (e.g., "relu", "leaky_relu").
    ///
    /// # Panics
    /// Panics if `mode` is not "fan_in" or "fan_out".
    ///
    /// # Example
    /// ```
    /// # use tensorust::nn::init::KaimingNormalInitializer;
    /// let initializer = KaimingNormalInitializer::new("fan_in", "relu");
    /// ```
    pub fn new(mode: &'static str, nonlinearity: &'static str) -> Self {
        assert!(
            mode == "fan_in" || mode == "fan_out",
            "mode must be 'fan_in' or 'fan_out'"
        );
        Self {
            gain: calculate_gain(nonlinearity, None),
            mode,
            nonlinearity,
        }
    }
}

impl<T, D, S> Initializer<T, D, S> for KaimingNormalInitializer
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    fn initialize(&self, tensor: &mut Tensor<T, D, S>) -> Result<()> {
        let shape = tensor.shape().as_slice();
        if shape.len() < 2 {
            return Err(crate::error::TensorustError::invalid_shape(
                "Kaiming initialization requires at least 2 dimensions",
            ));
        }
        
        let fan_in = shape[1];
        let fan_out = shape[0];
        let fan = match self.mode {
            "fan_in" => fan_in,
            "fan_out" => fan_out,
            _ => fan_in,
        };
        
        kaiming_normal(tensor, fan, self.gain)
    }
}

/// A Xavier (Glorot) uniform initializer.
///
/// This initializer is designed to keep the scale of the gradients roughly the same
/// in all layers. It's particularly well-suited for layers with tanh or sigmoid activation functions.
///
/// The weights are sampled from a uniform distribution in the range `[-bound, bound]` where:
/// ```text
/// bound = gain * sqrt(6 / (fan_in + fan_out))
/// ```
pub struct XavierUniformInitializer {
    /// The scaling factor to apply to the standard deviation.
    gain: f32,
}

impl XavierUniformInitializer {
    /// Creates a new Xavier uniform initializer.
    ///
    /// # Arguments
    /// * `gain` - A scaling factor to apply to the standard deviation.
    ///
    /// # Example
    /// ```
    /// # use tensorust::nn::init::XavierUniformInitializer;
    /// let initializer = XavierUniformInitializer::new(1.0);
    /// ```
    pub fn new(gain: f32) -> Self {
        Self { gain }
    }
}

impl<T, D, S> Initializer<T, D, S> for XavierUniformInitializer
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    fn initialize(&self, tensor: &mut Tensor<T, D, S>) -> Result<()> {
        let shape = tensor.shape().as_slice();
        if shape.len() < 2 {
            return Err(crate::error::TensorustError::invalid_shape(
                "Xavier initialization requires at least 2 dimensions",
            ));
        }
        
        let fan_in = shape[1];
        let fan_out = shape[0];
        
        xavier_uniform(tensor, fan_in, fan_out, self.gain)
    }
}

/// A Xavier (Glorot) normal initializer.
///
/// This initializer is designed to keep the scale of the gradients roughly the same
/// in all layers. It's particularly well-suited for layers with tanh or sigmoid activation functions.
///
/// The weights are sampled from a normal distribution with mean 0 and standard deviation:
/// ```text
/// stddev = gain * sqrt(2 / (fan_in + fan_out))
/// ```
pub struct XavierNormalInitializer {
    /// The scaling factor to apply to the standard deviation.
    gain: f32,
}

impl XavierNormalInitializer {
    /// Creates a new Xavier normal initializer.
    ///
    /// # Arguments
    /// * `gain` - A scaling factor to apply to the standard deviation.
    ///
    /// # Example
    /// ```
    /// # use tensorust::nn::init::XavierNormalInitializer;
    /// let initializer = XavierNormalInitializer::new(1.0);
    /// ```
    pub fn new(gain: f32) -> Self {
        Self { gain }
    }
}

impl<T, D, S> Initializer<T, D, S> for XavierNormalInitializer
where
    T: Clone + Default + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
{
    fn initialize(&self, tensor: &mut Tensor<T, D, S>) -> Result<()> {
        let shape = tensor.shape().as_slice();
        if shape.len() < 2 {
            return Err(crate::error::TensorustError::invalid_shape(
                "Xavier initialization requires at least 2 dimensions",
            ));
        }
        
        let fan_in = shape[1];
        let fan_out = shape[0];
        
        xavier_normal(tensor, fan_in, fan_out, self.gain)
    }
}

/// An orthogonal initializer.
///
/// This initializer generates a random orthogonal matrix, which can help with training
/// deep networks by preventing vanishing/exploding gradients. It's particularly useful for
/// recurrent neural networks (RNNs).
pub struct OrthogonalInitializer {
    /// The scaling factor to apply to the orthogonal matrix.
    gain: f32,
}

impl OrthogonalInitializer {
    /// Creates a new orthogonal initializer.
    ///
    /// # Arguments
    /// * `gain` - A scaling factor to apply to the orthogonal matrix.
    ///
    /// # Example
    /// ```
    /// # use tensorust::nn::init::OrthogonalInitializer;
    /// let initializer = OrthogonalInitializer::new(1.0);
    /// ```
    pub fn new(gain: f32) -> Self {
        Self { gain }
    }
}

impl<T, D, S> Initializer<T, D, S> for OrthogonalInitializer
where
    T: Clone + Default + Send + Sync + 'static + std::ops::Neg<Output = T> + From<f32>,
    D: Dimension,
    S: Storage<T>,
    f32: Into<T>,
    StandardNormal: Distribution<f32>,
    rand::distributions::Standard: Distribution<f32>,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    fn initialize(&self, tensor: &mut Tensor<T, D, S>) -> Result<()> {
        orthogonal_init(tensor, self.gain)
    }
}

/// Initializes a tensor with LeCun uniform initialization.
///
/// This is a variant of Kaiming initialization that uses `fan_in` scaling.
/// It's particularly well-suited for layers with SELU activation functions.
///
/// The weights are sampled from a uniform distribution in the range `[-bound, bound]` where:
/// ```text
/// bound = gain * sqrt(3 / fan_in)
/// ```
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize.
/// * `fan_in` - The number of input units in the weight tensor.
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # References
/// - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::lecun_uniform;
/// let mut x = tensor!([[0.0; 100]; 50]); // 50x100 weight matrix
/// lecun_uniform(&mut x, 100).unwrap();
/// ```
pub fn lecun_uniform<T, D, S>(tensor: &mut Tensor<T, D, S>, fan_in: usize) -> Result<()>
where
    T: Clone + From<f32> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let bound = (3.0 / fan_in as f32).sqrt();
    uniform_init(tensor, -bound, bound)
}

/// Initializes a tensor with LeCun normal initialization.
///
/// This is a variant of Kaiming initialization that uses `fan_in` scaling.
/// It's particularly well-suited for layers with SELU activation functions.
///
/// The weights are sampled from a normal distribution with mean 0 and standard deviation:
/// ```text
/// stddev = 1 / sqrt(fan_in)
/// ```
///
/// # Arguments
///
/// * `tensor` - The tensor to initialize.
/// * `fan_in` - The number of input units in the weight tensor.
///
/// # Returns
///
/// * `Result<()>` - `Ok(())` if the operation was successful, otherwise an error.
///
/// # References
/// - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
///
/// # Example
/// ```
/// # use tensorust::tensor;
/// # use tensorust::nn::init::lecun_normal;
/// let mut x = tensor!([[0.0; 100]; 50]); // 50x100 weight matrix
/// lecun_normal(&mut x, 100).unwrap();
/// ```
pub fn lecun_normal<T, D, S>(tensor: &mut Tensor<T, D, S>, fan_in: usize) -> Result<()>
where
    T: Clone + From<f32> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    let stddev = (1.0 / fan_in as f32).sqrt();
    normal_init(tensor, 0.0, stddev)
}

/// A LeCun uniform initializer.
pub struct LeCunUniformInitializer {
    _private: (),
}

impl LeCunUniformInitializer {
    /// Creates a new LeCun uniform initializer.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

/// A LeCun normal initializer.
pub struct LeCunNormalInitializer {
    _private: (),
}

impl Default for LeCunNormalInitializer {
    fn default() -> Self {
        Self::new()
    }
}

impl LeCunNormalInitializer {
    /// Creates a new LeCun normal initializer.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

/// A variance scaling initializer.
///
/// This initializer is a generalization of other initialization schemes like
/// `Kaiming` and `Xavier` initialization. It allows for more control over the
/// scaling of the variance of the weights.
///
/// The weights are sampled from a distribution with mean 0 and standard deviation:
/// ```text
/// stddev = sqrt(scale / (fan_in ^ negative_power))
/// ```
/// where `fan_in` is the number of input units in the weight tensor.
pub struct VarianceScalingInitializer {
    /// The scale to apply to the variance.
    scale: f32,
    /// The mode to use for fan calculation ("fan_in", "fan_out", or "fan_avg").
    mode: &'static str,
    /// The power to raise the fan to in the denominator.
    negative_power: f32,
    /// The distribution to sample from ("normal" or "uniform").
    distribution: &'static str,
}

impl VarianceScalingInitializer {
    /// Creates a new variance scaling initializer.
    ///
    /// # Arguments
    /// * `scale` - Scaling factor for the variance (default: 1.0).
    /// * `mode` - One of "fan_in", "fan_out", or "fan_avg".
    /// * `distribution` - Either "normal" or "uniform".
    ///
    /// # Panics
    /// Panics if `mode` is not one of the allowed values or if `distribution` is invalid.
    ///
    /// # Example
    /// ```
    /// # use tensorust::nn::init::VarianceScalingInitializer;
    /// // Equivalent to Kaiming normal with fan_in mode
    /// let initializer = VarianceScalingInitializer::new(2.0, "fan_in", "normal");
    /// ```
    pub fn new(scale: f32, mode: &'static str, distribution: &'static str) -> Self {
        assert!(
            mode == "fan_in" || mode == "fan_out" || mode == "fan_avg",
            "mode must be 'fan_in', 'fan_out', or 'fan_avg'"
        );
        assert!(
            distribution == "normal" || distribution == "uniform",
            "distribution must be 'normal' or 'uniform'"
        );

        // For fan_avg, we use 0.5 as the power
        let negative_power = if mode == "fan_avg" { 0.5 } else { 1.0 };

        Self {
            scale,
            mode,
            negative_power,
            distribution,
        }
    }

    /// Creates a new variance scaling initializer with default parameters.
    ///
    /// This is equivalent to `VarianceScalingInitializer::new(1.0, "fan_in", "normal")`.
    pub fn default() -> Self {
        Self::new(1.0, "fan_in", "normal")
    }
}

impl<T, D, S> Initializer<T, D, S> for VarianceScalingInitializer
where
    T: Clone + From<f32> + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    fn initialize(&self, tensor: &mut Tensor<T, D, S>) -> Result<()> {
        let shape = tensor.shape();
        if shape.ndims() < 2 {
            return Err(crate::error::TensorustError::InvalidInput(
                "VarianceScaling initializer requires tensors with at least 2 dimensions".to_string(),
            ));
        }

        let fan_in = shape.size() / shape.last_dim();
        let fan_out = shape.last_dim();

        let scale = match self.mode {
            "fan_in" => self.scale / (fan_in as f32).powf(self.negative_power),
            "fan_out" => self.scale / (fan_out as f32).powf(self.negative_power),
            "fan_avg" => self.scale / (((fan_in + fan_out) as f32) * 0.5).powf(self.negative_power),
            _ => unreachable!(),
        };

        match self.distribution {
            "normal" => normal_init(tensor, 0.0, scale.sqrt()),
            "uniform" => {
                let bound = (3.0 * scale).sqrt();
                uniform_init(tensor, -bound, bound)
            }
            _ => unreachable!(),
        }
    }
}

/// Calculates the recommended gain value for the given nonlinearity function.
///
/// # Arguments
///
/// * `nonlinearity` - The name of the nonlinearity function.
/// * `param` - An optional parameter for the nonlinearity function.
///
/// # Returns
///
/// The recommended gain value for the specified nonlinearity.
pub fn calculate_gain(nonlinearity: &str, param: Option<f32>) -> f32 {
    match nonlinearity {
        "tanh" => 5.0 / 3.0,
        "relu" => 2.0f32.sqrt(),
        "leaky_relu" => {
            let neg_slope = param.unwrap_or(0.01);
            (2.0 / (1.0 + neg_slope.powi(2))).sqrt()
        },
        "selu" => 3.0 / 4.0,  // Approximate value
        "linear" | "identity" => 1.0,
        "sigmoid" => 1.0,
        "hard_sigmoid" => 1.0,
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;
    use approx::assert_relative_eq;
    use std::f32::consts::SQRT_2;

    // Helper function to calculate mean and variance of a tensor
    fn mean_variance<T: AsRef<[f32]>>(data: T) -> (f32, f32) {
        let data = data.as_ref();
        let n = data.len() as f32;
        let sum: f32 = data.iter().sum();
        let mean = sum / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
        (mean, variance)
    }

    // Helper to check if a matrix is approximately orthogonal
    fn is_approximately_orthogonal<const N: usize, const M: usize>(matrix: &[[f32; M]; N]) -> bool {
        let mut product = [[0.0; M]; M];
        
        // Compute A^T * A
        for i in 0..M {
            for j in 0..M {
                for k in 0..N {
                    product[i][j] += matrix[k][i] * matrix[k][j];
                }
            }
        }
        
        // Check if A^T * A is close to identity
        for i in 0..M {
            for j in 0..M {
                if i == j {
                    if (product[i][j] - 1.0).abs() > 1e-3 {
                        return false;
                    }
                } else if product[i][j].abs() > 1e-3 {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_uniform_init() {
        let mut x = tensor!([0.0; 1000]);
        uniform_init(&mut x, -1.0, 1.0).unwrap();
        
        // Check bounds
        for &val in x.data() {
            assert!(val >= -1.0 && val < 1.0);
        }
        
        // Check mean is close to 0
        let (mean, _) = mean_variance(x.data());
        assert_relative_eq!(mean, 0.0, epsilon = 0.1);
    }
    
    #[test]
    fn test_normal_init() {
        let mut x = tensor!([0.0; 10000]);
        normal_init(&mut x, 1.0, 2.0).unwrap();
        
        let (mean, variance) = mean_variance(x.data());
        
        // Check mean is close to 1.0
        assert_relative_eq!(mean, 1.0, epsilon = 0.1);
        // Check variance is close to 4.0 (stddev^2)
        assert_relative_eq!(variance, 4.0, epsilon = 0.2);
    }
    
    #[test]
    fn test_truncated_normal_init() {
        let mut x = tensor!([0.0; 10000]);
        truncated_normal_init(&mut x, 0.0, 1.0, -2.0, 2.0).unwrap();
        
        // Check bounds
        for &val in x.data() {
            assert!(val >= -2.0 && val <= 2.0);
        }
        
        // Check mean is close to 0
        let (mean, variance) = mean_variance(x.data());
        assert_relative_eq!(mean, 0.0, epsilon = 0.1);
        
        // Variance should be less than normal distribution due to truncation
        assert!(variance < 1.0);
    }
    
    #[test]
    fn test_kaiming_uniform() {
        let mut x = tensor!([[0.0; 100]; 100]);
        kaiming_uniform(&mut x, 100, 2.0f32.sqrt()).unwrap();
        
        // Check that values are within expected bounds
        let bound = (3.0 / 100.0 as f32).sqrt() * 2.0f32.sqrt();
        for &val in x.data() {
            assert!(val >= -bound && val <= bound);
        }
    }
    
    #[test]
    fn test_orthogonal_init() {
        let mut x = tensor!([[0.0; 10]; 10]); // 10x10 weight matrix
        orthogonal_init(&mut x, 1.0).unwrap();
        
        // Convert to 2D array for easier manipulation
        let mut matrix = [[0.0; 10]; 10];
        for i in 0..10 {
            for j in 0..10 {
                matrix[i][j] = x.data()[i * 10 + j];
            }
        }
        
        // Check that the matrix is approximately orthogonal
        assert!(is_approximately_orthogonal(&matrix));
        
        // Test with gain
        orthogonal_init(&mut x, 2.0).unwrap();
        for i in 0..10 {
            for j in 0..10 {
                matrix[i][j] = x.data()[i * 10 + j];
            }
        }
        
        // Check that the matrix scaled by 2 is still orthogonal
        let mut scaled_matrix = [[0.0; 10]; 10];
        for i in 0..10 {
            for j in 0..10 {
                scaled_matrix[i][j] = matrix[i][j] / 2.0;
            }
        }
        assert!(is_approximately_orthogonal(&scaled_matrix));
    } 
    
    #[test]
    fn test_initializer_trait() {
        let mut x = tensor!([0.0; 100]);
        let initializer = UniformInitializer::new(-1.0, 1.0);
        initializer.initialize(&mut x).unwrap();
        
        // Check that all values are in the range [-1, 1]
        for &val in x.data() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }
}
