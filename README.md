# Tensorust

Tensorust is a high-performance tensor computation library for Rust, providing compile-time verification of tensor shapes, compiler-enforced automatic differentiation, and explicit management of tensor operations across CPU and CUDA devices. Designed to support numerical computing and machine learning, Tensorust leverages Rust’s ownership and borrowing system to ensure memory safety and prevent data races.

## 🎯 Project Goals

- **Compile-Time Safety**: All tensor operations are statically checked to prevent shape errors at compile time.
- **Safe Automatic Differentiation**: Automatic differentiation on tensors is implemented with compiler-level safety guarantees.
- **Cross-Device Computation**: Tensors support safe and explicit movement and computation across CPU and CUDA.
- **Memory Safety**: Memory safety of tensors is guaranteed through Rust’s ownership and borrowing system, eliminating data races.

## Features

- **Flexible Tensor Operations**: Support for multi-dimensional arrays with various data types
- **Automatic Differentiation**: Build and train neural networks with automatic differentiation
- **CPU and GPU Support**: Core CPU implementation with CUDA support (experimental)
- **Efficient Views**: Zero-copy tensor views for memory-efficient operations
- **Type Safety**: Leverage Rust's type system for compile-time safety
- **Neural Network Module**: Comprehensive NN building blocks including:
  - Layers: Linear, Conv2d, RNN, LSTM, BatchNorm, Dropout
  - Activations: ReLU, Sigmoid, Tanh, Softmax
  - Loss Functions: MSE, CrossEntropy, BCE
  - Optimizers: SGD, Adam, RMSprop
- **Expression Graph**: Lazy evaluation for optimized computation

## Installation

Add Tensorust to your `Cargo.toml`:

```toml
[dependencies]
tensorust = { git = "https://github.com/tensorust/crate" }
```

### Optional Features

- **CUDA Support** (experimental):
  ```toml
  [dependencies]
  tensorust = { git = "https://github.com/tensorust/crate", features = ["cuda"] }
  ```
  Note: Requires CUDA toolkit to be installed

- **Serialization** (requires `serde`):
  ```toml
  [dependencies]
  tensorust = { git = "https://github.com/tensorust/crate", features = ["serde"] }
  ```

## Quick Start

### Basic Tensor Operations

```rust
use tensorust::prelude::*;

fn main() -> Result<(), TensorustError> {
    // Create tensors
    let a = tensor![[1.0, 2.0], [3.0, 4.0]];
    let b = tensor![[5.0, 6.0], [7.0, 8.0]];
    
    // Element-wise operations
    let c = &a + &b;
    let d = &a * &b;
    
    // Matrix multiplication
    let e = a.matmul(&b)?;
    
    // Print results
    println!("a + b = {:?}", c);
    println!("a * b = {:?}", d);
    println!("a @ b = {:?}", e);
    
    Ok(())
}
```

### Neural Network Example

```rust
use tensorust::prelude::*;
use tensorust::nn::{Sequential, Linear, ReLU, MSELoss};
use tensorust::optim::SGD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple neural network
    let model = Sequential::new()
        .add(Box::new(Linear::new(10, 20)?))
        .add(Box::new(ReLU::new()))
        .add(Box::new(Linear::new(20, 1)?));
    
    // Create sample input and target tensors
    let input = Tensor::randn(&[1, 10])?;  // Batch size of 1, 10 features
    let target = Tensor::randn(&[1, 1])?;   // Single output value
    
    // Create optimizer and loss function
    let mut optimizer = SGD::new(0.01);
    let criterion = MSELoss::new();
    
    // Training loop
    for epoch in 0..10 {
        // Forward pass
        let output = model.forward(&input)?;
        let loss = criterion.forward(&output, &target)?;
        
        // Backward pass
        model.zero_grad();
        loss.backward()?;
        
        // Update weights
        optimizer.step(&mut model.parameters())?;
        
        println!("Epoch {}: Loss = {:?}", epoch, loss.data());
    }
    
    Ok(())
}
```

## Core Components

### Tensors

The core data structure in Tensorust is the `Tensor` type, which represents multi-dimensional arrays:

```rust
// Create a tensor from a vector
let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;

// Create a tensor filled with zeros
let b = Tensor::zeros(&[2, 3])?;

// Create a tensor with random values
let c = Tensor::randn(&[2, 2])?;
```

### Automatic Differentiation

Tensorust supports automatic differentiation through computation graphs:

```rust
// Create tensors with requires_grad=True
let x = tensor!([1.0, 2.0, 3.0]).requires_grad(true);
let w = tensor!([0.5, 0.5, 0.5]).requires_grad(true);

// Forward pass
y = x.dot(&w)?;  // y = x · w
y = y.relu();    // y = ReLU(x · w)

// Backward pass
y.backward()?;

// Access gradients
println!("Gradient of w: {:?}", w.grad());
```

### Neural Network Module

Tensorust provides building blocks for neural networks:

```rust
use tensorust::nn::{Linear, ReLU, Sequential, CrossEntropyLoss, Adam};

// Create a model
let model = Sequential::new()
    .add(Box::new(Linear::new(784, 128)?))
    .add(Box::new(ReLU::new()))
    .add(Box::new(Linear::new(128, 10)?));

// Loss function and optimizer
let criterion = CrossEntropyLoss::new();
let mut optimizer = Adam::new(0.001);
```

## Advanced Usage

### GPU Acceleration

Tensorust supports GPU acceleration through CUDA (requires CUDA toolkit):

```rust
#[cfg(feature = "cuda")]
{
    use tensorust::cuda::CudaTensor;
    
    // Create a tensor on GPU
    let a = CudaTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
    let b = CudaTensor::from_vec(vec![4.0, 5.0, 6.0], vec![3])?;
    
    // Operations are executed on GPU
    let c = &a + &b;
}
```

### Custom Operations

You can define custom operations by implementing the `Function` trait:

```rust
use tensorust::function::Function;

struct MyCustomOp;

impl Function for MyCustomOp {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TensorustError> {
        // Implement forward pass
        let x = inputs[0];
        // Custom operation
        x.map(|v| v * v)
    }
    
    fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>, TensorustError> {
        // Implement backward pass
        let x = inputs[0];
        let grad_x = grad_output * &(x * 2.0);
        Ok(vec![grad_x])
    }
}
```

## API Reference

### Crates and Modules

Tensorust is organized into several crates and modules:

#### Core Crate (`tensorust`)

```toml
[dependencies]
tensorust = { version = "0.1.0", features = ["cuda"] }
```

#### Main Modules

| Module | Description |
|--------|-------------|
| `tensor` | Core tensor types and operations |
| `dimension` | Dimension and shape handling |
| `view` | Tensor views and slicing operations |
| `expression` | Lazy evaluation and computation graphs |
| `nn` | Neural network building blocks |
| `optim` | Optimization algorithms |
| `linalg` | Linear algebra operations |
| `cuda` | CUDA backend (optional) |
| `macros` | Procedural macros for tensor creation |

### Feature Flags

Tensorust supports the following feature flags:

- `cuda`: Enables CUDA GPU acceleration
- `serde`: Enables serialization support
- `ndarray`: Enables interoperability with the `ndarray` crate
- `blas`: Enables BLAS acceleration for linear algebra
- `test`: Includes testing utilities

### Core Types and Traits

#### Tensor Types

| Type | Description |
|------|-------------|
| `Tensor<T, D, S>` | Generic tensor type with element type `T`, dimension `D`, and storage `S` |
| `CpuTensor<T, D>` | Tensor with CPU storage |
| `CudaTensor<T, D>` | Tensor with CUDA storage (requires `cuda` feature) |

#### Key Traits

| Trait | Description |
|-------|-------------|
| `Storage<T>` | Abstract trait for tensor storage backends |
| `Dimension` | Trait for tensor dimensions |
| `View` | Trait for tensor views |
| `Function` | Trait for custom operations |
| `Module` | Trait for neural network modules |
| `Optimizer` | Trait for optimization algorithms |

### Tensor Operations

#### Creation

```rust
// Create from vector
let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;

// Create with specific value
let t2 = Tensor::full(2.0, &[2, 2])?;

// Create identity matrix
let eye = Tensor::eye(3)?;

// Random initialization
let rand_tensor = Tensor::randn(&[3, 3])?;
```

#### Operations

```rust
// Element-wise operations
let sum = &a + &b;
let product = &a * &b;

// Matrix multiplication
let matmul = a.matmul(&b)?;

// Reduction operations
let sum_all = a.sum_all();
let mean = a.mean(0)?;  // Mean along dimension 0

// Shape operations
let reshaped = a.reshape(&[6, 1])?;
let transposed = a.transpose()?;
```

### Neural Network Module

#### Layers

```rust
use tensorust::nn::{Linear, Conv2d, Dropout, BatchNorm1d};

// Linear layer
let linear = Linear::new(784, 256)?;

// Convolutional layer
let conv = Conv2d::new(3, 64, (3, 3))?;

// Batch normalization
let bn = BatchNorm1d::new(100)?;

// Dropout
let dropout = Dropout::new(0.5);
```

#### Loss Functions

```rust
use tensorust::nn::{MSELoss, CrossEntropyLoss, BCELoss};

let mse = MSELoss::new();
let ce = CrossEntropyLoss::new();
let bce = BCELoss::new();
```

#### Optimizers

```rust
use tensorust::optim::{SGD, Adam, RMSprop};

let sgd = SGD::new(0.01);
let adam = Adam::new(0.001);
let rmsprop = RMSprop::new(0.01);
```

### CUDA Support (Experimental)

Tensorust provides experimental CUDA support through the `cuda` feature. This allows you to perform tensor operations on NVIDIA GPUs.

```toml
[dependencies]
tensorust = { git = "https://github.com/tensorust/crate", features = ["cuda"] }
```

Example usage:

```rust
#[cfg(feature = "cuda")]
{
    use tensorust::cuda::CudaTensor;
    
    // Create tensors on GPU
    let a = CudaTensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
    let b = CudaTensor::from_vec(vec![4.0, 5.0, 6.0], vec![3])?;
    
    // Operations execute on GPU
    let c = &a + &b;
    
    // Transfer back to CPU if needed
    let c_cpu = c.to_cpu()?;
}
```

**Note**: CUDA support requires:
- CUDA toolkit installed
- Compatible NVIDIA GPU
- Environment variable `CUDA_HOME` set to your CUDA installation

### Serialization

Enable serialization with the `serde` feature:

```toml
[dependencies]
tensorust = { version = "0.1.0", features = ["serde"] }
```

```rust
use std::fs::File;
use tensorust::serialization::save;

// Save model
save("model.pt", &model)?;

// Load model
let model: MyModel = tensorust::serialization::load("model.pt")?;
```

## Performance

Tensorust is designed for high performance:

- **Efficient Memory Management**: Minimizes allocations and copies
- **Vectorized Operations**: Leverages SIMD instructions on supported hardware
- **Lazy Evaluation**: Optimizes computation graphs for better performance
- **GPU Acceleration**: Offloads computation to NVIDIA GPUs

## Benchmarks

```
# Run benchmarks
cargo bench
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on how to submit pull requests.

### Building from Source

```bash
# Clone the repository
git clone https://github.com/tensorust/crate.git
cd tensorust

# Build in release mode
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Code Style

We use `rustfmt` for consistent code formatting:

```bash
cargo fmt
```

And `clippy` for linting:

```bash
cargo clippy -- -D warnings
```

## License

Tensorust is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by PyTorch, NumPy, and other great numerical computing libraries
- Built with the amazing Rust ecosystem

## Roadmap

See the [open issues](https://github.com/tensorust/crate/issues) for a list of proposed features and known issues.

## Support

For questions and support, please open an issue on GitHub.
