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

use crate::error::Result;
use std::{fmt::Debug, sync::Arc};


/// A trait for tensor storage backends.
///
/// This trait provides an abstraction over different storage backends, such as
/// CPU memory, CUDA memory, and memory-mapped files.
pub trait Storage<T>: Debug + Send + Sync + 'static {
    /// Returns the number of elements in the storage.
    fn len(&self) -> usize;

    /// Returns `true` if the storage contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a slice containing the entire storage.
    fn as_slice(&self) -> &[T];

    /// Returns a mutable slice containing the entire storage.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Creates a new storage from a vector.
    fn from_vec(data: Vec<T>) -> Self;

    /// Creates a new storage with the given capacity.
    fn with_capacity(capacity: usize) -> Self;

    /// Creates a new storage by cloning the given slice.
    fn from_slice(data: &[T]) -> Self
    where
        T: Clone;

    /// Returns a new storage with the same data.
    fn clone(&self) -> Self
    where
        T: Clone;
}

/// A reference-counted storage handle.
pub type StorageHandle<T> = Arc<dyn Storage<T>>;
