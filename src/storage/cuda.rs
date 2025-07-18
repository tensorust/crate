#![cfg(feature = "cuda")]
//! CUDA storage backend for GPU-accelerated tensor operations.

use super::Storage;
use crate::error::TensorustError as StorageError;
use std::sync::Arc;

/// CUDA storage backend for GPU-accelerated operations.
///
/// This backend requires CUDA-compatible hardware and the CUDA toolkit.
#[derive(Debug)]
pub struct CudaStorage<T> {
    device_ptr: *mut std::ffi::c_void,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> CudaStorage<T> {
    /// Creates a new CUDA storage with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                device_ptr: std::ptr::null_mut(),
                len: 0,
                _phantom: std::marker::PhantomData,
            };
        }

        let elem_size = std::mem::size_of::<T>();
        let size = capacity
            .checked_mul(elem_size)
            .ok_or_else(|| StorageError::AllocationFailed("Capacity too large".to_string()))
            .unwrap();

        let mut device_ptr = std::ptr::null_mut();
        let result = unsafe {
            // SAFETY: We've checked that size is non-zero and multiplication didn't overflow
            cuda_driver_sys::cuMemAlloc_v2(&mut device_ptr as *mut _, size)
        };

        if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            panic!(
                "Failed to allocate {} bytes on device: {:?}",
                size, result
            );
        }

        Self {
            device_ptr,
            len: capacity,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new CUDA storage from a vector.
    pub fn from_vec(data: Vec<T>) -> Self {
        let len = data.len();
        let mut storage = Self::with_capacity(len);

        if len > 0 {
            // Copy data to device
            let size = len * std::mem::size_of::<T>();
            let result = unsafe {
                cuda_driver_sys::cuMemcpyHtoD_v2(
                    storage.device_ptr as u64,
                    data.as_ptr() as *const _,
                    size,
                )
            };

            if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
                panic!("Failed to copy data to device: {:?}", result);
            }
        }

        storage
    }

    /// Returns a raw pointer to the device memory.
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.device_ptr
    }

    /// Returns a mutable raw pointer to the device memory.
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.device_ptr
    }
}

impl<T: Clone + Send + Sync + 'static> Storage<T> for CudaStorage<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[T] {
        // This is not safe and is a placeholder.
        // A real implementation would need to copy data from the device.
        unsafe { std::slice::from_raw_parts(self.device_ptr as *const T, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        // This is not safe and is a placeholder.
        // A real implementation would need to copy data from the device.
        unsafe { std::slice::from_raw_parts_mut(self.device_ptr as *mut T, self.len) }
    }

    fn from_vec(data: Vec<T>) -> Self {
        Self::from_vec(data).unwrap()
    }

    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity(capacity).unwrap()
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
        let mut new_storage =
            Self::with_capacity(self.len).expect("Failed to allocate device memory for clone");

        if self.len > 0 {
            let size = self.len * std::mem::size_of::<T>();
            let result = unsafe {
                cuda_driver_sys::cuMemcpyDtoD_v2(
                    new_storage.device_ptr as u64,
                    self.device_ptr as u64,
                    size,
                )
            };

            if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
                panic!("Failed to copy device memory: {:?}", result);
            }
        }

        new_storage
    }
}

impl<T> Drop for CudaStorage<T> {
    fn drop(&mut self) {
        if !self.device_ptr.is_null() {
            let result = unsafe { cuda_driver_sys::cuMemFree_v2(self.device_ptr as u64) };
            if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
                eprintln!("Failed to free CUDA memory: {:?}", result);
            }
            self.device_ptr = std::ptr::null_mut();
        }
    }
}

// SAFETY: The CUDA driver API is thread-safe
unsafe impl<T> Send for CudaStorage<T> {}
unsafe impl<T> Sync for CudaStorage<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_driver_sys as cuda;

    fn init_cuda() -> Result<(), StorageError> {
        static INIT: std::sync::Once = std::sync::Once::new();
        static mut INIT_SUCCESS: bool = false;
        static mut INIT_ERROR: Option<StorageError> = None;

        INIT.call_once(|| {
            unsafe {
                let result = cuda::cuInit(0);
                if result == cuda::CUresult::CUDA_SUCCESS {
                    INIT_SUCCESS = true;
                } else {
                    INIT_ERROR = Some(StorageError::CudaError(format!(
                        "Failed to initialize CUDA: {:?}",
                        result
                    )));
                }
            }
        });

        unsafe {
            if INIT_SUCCESS {
                Ok(())
            } else {
                Err(INIT_ERROR.take().unwrap_or_else(|| {
                    StorageError::CudaError("Unknown CUDA initialization error".to_string())
                }))
            }
        }
    }

    #[test]
    fn test_cuda_storage() -> Result<(), StorageError> {
        // Skip test if CUDA is not available
        if init_cuda().is_err() {
            eprintln!("CUDA not available, skipping test_cuda_storage");
            return Ok(());
        }

        // Test with_capacity
        let storage = CudaStorage::<f32>::with_capacity(4)?;
        assert_eq!(storage.len(), 4);

        // Test from_vec
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut storage = CudaStorage::from_vec(data.clone())?;
        assert_eq!(storage.len(), 4);

        // Test to_vec
        let result = storage.to_vec();
        assert_eq!(result, data);

        // Test set and get (indirectly through to_vec)
        storage.set(1, 42.0)?;
        let result = storage.to_vec();
        assert_eq!(result[1], 42.0);

        // Test clone
        let cloned = storage.clone();
        assert_eq!(cloned.to_vec(), result);

        Ok(())
    }
}
