//! CPU storage backend implementation.

use super::Storage;
use std::sync::RwLock;

/// CPU storage using a simple `Vec`.
#[derive(Debug)]
pub struct CpuStorage<T> {
    data: RwLock<Vec<T>>,
}

impl<T: Clone + Send + Sync + 'static> CpuStorage<T> {
    /// Create a new CPU storage with the given capacity.
    pub fn with_capacity(capacity: usize) -> Result<Self, super::StorageError> {
        Ok(Self {
            data: RwLock::new(Vec::with_capacity(capacity)),
        })
    }

    /// Create a new CPU storage from a vector.
    pub fn from_vec(data: Vec<T>) -> Result<Self, super::StorageError> {
        Ok(Self {
            data: RwLock::new(data),
        })
    }
}

impl<T: Clone + Send + Sync + 'static> Storage<T> for CpuStorage<T> {
    fn len(&self) -> usize {
        self.data.read().unwrap().len()
    }

    fn is_empty(&self) -> bool {
        self.data.read().unwrap().is_empty()
    }

    fn as_slice(&self) -> &[T] {
        // This is a bit of a hack to work around RwLock's limitations
        // In practice, you'd want a different approach for production code
        unsafe { std::mem::transmute(&self.data.read().unwrap()[..]) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.get_mut().unwrap().as_mut_slice()
    }

    fn clone(&self) -> Box<dyn Storage<T>> {
        Box::new(Self {
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
