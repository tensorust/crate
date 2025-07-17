//! Utility functions and types for working with tensor storage.

use super::{Storage, StorageError, StorageHandle};
use std::sync::Arc;

/// Converts between different storage backends.
pub trait StorageConverter<T> {
    /// The target storage type.
    type Target: Storage<T>;

    /// Converts the storage to the target type.
    fn convert(&self) -> Result<Self::Target, StorageError>;
}

/// A batch of storage operations that can be executed together.
pub struct StorageBatch<T> {
    operations: Vec<Box<dyn FnOnce() -> Result<(), StorageError> + Send>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for StorageBatch<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StorageBatch<T> {
    /// Creates a new empty batch.
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Adds a copy operation to the batch.
    pub fn add_copy<S>(&mut self, source: &S, target: &mut S) -> &mut Self
    where
        S: Storage<T> + ?Sized,
        T: Clone + 'static,
    {
        let source = source.clone_handle();
        let target_ptr = target as *mut S as *mut ();
        
        self.operations.push(Box::new(move || {
            // SAFETY: The target reference is valid for the duration of the batch
            let target = unsafe { &mut *(target_ptr as *mut S) };
            
            if source.len() != target.len() {
                return Err(StorageError::ShapeMismatch(format!(
                    "Source length {} does not match target length {}",
                    source.len(),
                    target.len()
                )));
            }
            
            // For simplicity, we just copy the data through host memory
            // In a real implementation, you might want direct device-to-device copies
            let data = source.to_vec();
            for (i, value) in data.into_iter().enumerate() {
                target.set(i, value)?;
            }
            
            Ok(())
        }));
        
        self
    }

    /// Executes all operations in the batch.
    pub fn execute(self) -> Result<(), StorageError> {
        for op in self.operations {
            op()?;
        }
        Ok(())
    }
}

/// Extension trait for storage handles.
pub trait StorageExt<T>: Storage<T> {
    /// Creates a new storage with the same type and capacity.
    fn new_like(&self) -> Result<StorageHandle<T>, StorageError> {
        self.with_capacity(self.len())
    }

    /// Creates a new storage with the same type and copies the data.
    fn duplicate(&self) -> Result<StorageHandle<T>, StorageError>
    where
        T: Clone,
    {
        let mut new_storage = self.with_capacity(self.len())?;
        for i in 0..self.len() {
            if let Some(value) = self.get(i) {
                new_storage.set(i, value.clone())?;
            }
        }
        Ok(new_storage)
    }

    /// Creates a new storage with the same type and maps the values.
    fn map<F, U>(&self, f: F) -> Result<StorageHandle<U>, StorageError>
    where
        F: Fn(&T) -> U + Send + Sync + 'static,
        U: Clone + Send + Sync + 'static,
    {
        let mut new_storage = <dyn Storage<U>>::with_capacity(self.len())?;
        for i in 0..self.len() {
            if let Some(value) = self.get(i) {
                new_storage.set(i, f(value))?;
            }
        }
        Ok(new_storage)
    }
}

// Implement StorageExt for all Storage types
impl<T, S> StorageExt<T> for S where S: Storage<T> + ?Sized {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::CpuStorage;

    #[test]
    fn test_storage_ext() {
        let storage = CpuStorage::from_vec(vec![1, 2, 3, 4]).unwrap();
        
        // Test new_like
        let new_storage = storage.new_like().unwrap();
        assert_eq!(new_storage.len(), 4);
        
        // Test duplicate
        let dup = storage.duplicate().unwrap();
        assert_eq!(dup.to_vec(), vec![1, 2, 3, 4]);
        
        // Test map
        let mapped = storage.map(|x| x * 2).unwrap();
        assert_eq!(mapped.to_vec(), vec![2, 4, 6, 8]);
    }

    #[test]
    test_storage_batch() {
        let mut src = CpuStorage::from_vec(vec![1, 2, 3, 4]).unwrap();
        let mut dst1 = CpuStorage::from_vec(vec![0; 4]).unwrap();
        let mut dst2 = CpuStorage::from_vec(vec![0; 4]).unwrap();
        
        let mut batch = StorageBatch::new();
        batch
            .add_copy(&src, &mut dst1)
            .add_copy(&src, &mut dst2);
            
        batch.execute().unwrap();
        
        assert_eq!(dst1.to_vec(), vec![1, 2, 3, 4]);
        assert_eq!(dst2.to_vec(), vec![1, 2, 3, 4]);
    }
}
