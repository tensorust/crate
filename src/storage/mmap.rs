#![cfg(feature = "mmap")]
//! Memory-mapped storage backend for large tensors.

use super::Storage;
use crate::error::TensorustError as StorageError;
use memmap2::{MmapMut, MmapOptions};
use std::{
    fs::{File, OpenOptions},
    path::PathBuf,
    sync::Arc,
};

/// Memory-mapped storage for large tensors.
///
/// This storage backend is useful for tensors that are too large to fit in memory.
/// It uses memory-mapped files to efficiently access data on disk.
#[derive(Debug)]
pub struct MmapStorage<T> {
    mmap: MmapMut,
    _file: Option<File>,
    _marker: std::marker::PhantomData<T>,
}

impl<T> MmapStorage<T> {
    /// Creates a new memory-mapped storage with the given capacity.
    ///
    /// # Arguments
    /// * `path` - Path to the memory-mapped file (temporary file if None)
    /// * `capacity` - Number of elements to store
    pub fn with_capacity(
        path: Option<PathBuf>,
        capacity: usize,
    ) -> Result<Self, StorageError> {
        let elem_size = std::mem::size_of::<T>();
        let file_size = capacity.checked_mul(elem_size).ok_or_else(|| {
            StorageError::AllocationFailed("Capacity too large".to_string())
        })?;

        let (file, path) = if let Some(path) = path {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&path)
                .map_err(|e| StorageError::IoError(e))?;
            
            // Set the file length
            file.set_len(file_size as u64)
                .map_err(|e| StorageError::IoError(e))?;
                
            (file, Some(path))
        } else {
            // Create a temporary file
            let temp_dir = std::env::temp_dir();
            let path = temp_dir.join(format!("tensorust_{}.mmap", uuid::Uuid::new_v4()));
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&path)
                .map_err(|e| StorageError::IoError(e))?;
            
            file.set_len(file_size as u64)
                .map_err(|e| StorageError::IoError(e))?;
                
            (file, Some(path))
        };

        // Memory map the file
        let mmap = unsafe {
            MmapOptions::new()
                .len(file_size)
                .map_mut(&file)
                .map_err(|e| StorageError::IoError(e.into()))?
        };

        Ok(Self {
            mmap,
            _file: Some(file),
            _marker: std::marker::PhantomData,
        })
    }

    /// Creates a new memory-mapped storage from a vector.
    pub fn from_vec(data: Vec<T>) -> Result<Self, StorageError> {
        let len = data.len();
        let mut storage = Self::with_capacity(None, len)?;
        
        // SAFETY: We just created the storage with sufficient capacity
        unsafe {
            let ptr = storage.mmap.as_mut_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);
        }
        
        Ok(storage)
    }
}

impl<T: Clone + Send + Sync + 'static> Storage<T> for MmapStorage<T> {
    fn len(&self) -> usize {
        self.mmap.len() / std::mem::size_of::<T>()
    }

    fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.mmap.as_ptr() as *const T, self.len())
        }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.mmap.as_mut_ptr() as *mut T, self.len())
        }
    }

    fn from_vec(data: Vec<T>) -> Self {
        Self::from_vec(data).unwrap()
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity(None, capacity).unwrap()
    }

    fn from_slice(data: &[T]) -> Self
    where
        T: Clone,
    {
        Self::from_vec(data.to_vec()).unwrap()
    }

    fn clone(&self) -> Self
    where
        T: Clone,
    {
        let new_storage =
            Self::with_capacity(None, self.len()).expect("Failed to clone mmap storage");
        new_storage.mmap.copy_from_slice(&self.mmap);
        new_storage
    }
}

impl<T> Drop for MmapStorage<T> {
    fn drop(&mut self) {
        // Ensure all changes are flushed to disk
        if let Err(e) = self.mmap.flush() {
            eprintln!("Failed to flush memory map: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_mmap_storage() {
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test.mmap");
        
        // Test with file path
        let mut storage = MmapStorage::<f32>::with_capacity(Some(file_path.clone()), 4).unwrap();
        storage.set(0, 1.0).unwrap();
        storage.set(1, 2.0).unwrap();
        storage.set(2, 3.0).unwrap();
        storage.set(3, 4.0).unwrap();
        
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.get(0), Some(&1.0));
        assert_eq!(storage.get(1), Some(&2.0));
        assert_eq!(storage.get(2), Some(&3.0));
        assert_eq!(storage.get(3), Some(&4.0));
        assert_eq!(storage.get(4), None);
        
        // Test with temporary file
        let mut temp_storage = MmapStorage::<f32>::with_capacity(None, 2).unwrap();
        temp_storage.set(0, 10.0).unwrap();
        temp_storage.set(1, 20.0).unwrap();
        
        assert_eq!(temp_storage.len(), 2);
        assert_eq!(temp_storage.get(0), Some(&10.0));
        assert_eq!(temp_storage.get(1), Some(&20.0));
        
        // Test from_vec
        let vec_storage = MmapStorage::from_vec(vec![100.0, 200.0, 300.0]).unwrap();
        assert_eq!(vec_storage.len(), 3);
        assert_eq!(vec_storage.get(0), Some(&100.0));
        assert_eq!(vec_storage.get(1), Some(&200.0));
        assert_eq!(vec_storage.get(2), Some(&300.0));
        
        // Test clone
        let cloned = vec_storage.clone();
        assert_eq!(cloned.len(), 3);
        assert_eq!(cloned.get(0), Some(&100.0));
    }
}
