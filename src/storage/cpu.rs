//! CPU storage backend implementation.

use super::{Storage, StorageError};
use std::sync::RwLock;

/// CPU storage using a simple `Vec`.
#[derive(Debug)]
pub struct CpuStorage<T> {
    data: RwLock<Vec<T>>,
}

impl<T> CpuStorage<T> {
    /// Create a new CPU storage with the given capacity.
    pub fn with_capacity(capacity: usize) -> Result<Self, StorageError> {
        Ok(Self {
            data: RwLock::new(Vec::with_capacity(capacity)),
        })
    }

    /// Create a new CPU storage from a vector.
    pub fn from_vec(data: Vec<T>) -> Result<Self, StorageError> {
        Ok(Self {
            data: RwLock::new(data),
        })
    }
}

impl<T: Clone + Send + Sync + 'static> Storage<T> for CpuStorage<T> {
    fn with_capacity(capacity: usize) -> Result<Self, StorageError> {
        Self::with_capacity(capacity)
    }

    fn from_vec(data: Vec<T>) -> Result<Self, StorageError> {
        Self::from_vec(data)
    }

    fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }

    fn get(&self, index: usize) -> Option<&T> {
        // This is a bit of a hack to work around RwLock's limitations
        // In practice, you'd want a different approach for production code
        unsafe { std::mem::transmute(self.data.read().unwrap().get(index)) }
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut().unwrap().get_mut(index)
    }

    fn set(&mut self, index: usize, value: T) -> Result<(), StorageError> {
        let mut data = self.data.write().unwrap();
        if index >= data.len() {
            return Err(StorageError::ShapeMismatch(format!(
                "Index {} out of bounds for length {}",
                index,
                data.len()
            )));
        }
        data[index] = value;
        Ok(())
    }

    fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.read().unwrap().clone()
    }

    fn clone(&self) -> std::sync::Arc<dyn Storage<T>> {
        std::sync::Arc::new(Self {
            data: RwLock::new(self.data.read().unwrap().clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_storage() {
        let storage = CpuStorage::from_vec(vec![1, 2, 3, 4]).unwrap();
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.get(0), Some(&1));
        assert_eq!(storage.get(3), Some(&4));
        assert_eq!(storage.get(4), None);

        let mut storage = CpuStorage::from_vec(vec![0; 3]).unwrap();
        storage.set(1, 42).unwrap();
        assert_eq!(storage.get(1), Some(&42));

        assert!(storage.set(3, 0).is_err());
    }
}
