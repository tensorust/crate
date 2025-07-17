//! Storage abstraction for tensor data.
//! This module provides the `Storage` trait and implementations for different backends.

mod cpu;
mod cuda;
mod mmap;

pub use cpu::CpuStorage;
#[cfg(feature = "cuda")]
pub use cuda::CudaStorage;
#[cfg(feature = "mmap")]
pub use mmap::MmapStorage;
pub mod utils;
pub use utils::{StorageBatch, StorageConverter, StorageExt};

use std::sync::Arc;
use thiserror::Error;

/// Error type for storage operations.
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}

/// Trait for tensor storage backends.
pub trait Storage<T>: Send + Sync + std::fmt::Debug {
    /// Create a new storage with the given capacity.
    fn with_capacity(capacity: usize) -> Result<Self, StorageError>
    where
        Self: Sized;

    /// Create a new storage from a vector.
    fn from_vec(data: Vec<T>) -> Result<Self, StorageError>
    where
        Self: Sized;

    /// Get the length of the storage.
    fn len(&self) -> usize;

    /// Check if the storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the value at the given index.
    fn get(&self, index: usize) -> Option<&T>;

    /// Get a mutable reference to the value at the given index.
    fn get_mut(&mut self, index: usize) -> Option<&mut T>;

    /// Set the value at the given index.
    fn set(&mut self, index: usize, value: T) -> Result<(), StorageError>;

    /// Convert the storage to a vector.
    fn to_vec(&self) -> Vec<T>
    where
        T: Clone;

    /// Clone the storage into a new instance.
    fn clone(&self) -> Arc<dyn Storage<T>>
    where
        T: Clone + 'static;
}

/// A reference-counted storage handle.
pub type StorageHandle<T> = Arc<dyn Storage<T>>;
