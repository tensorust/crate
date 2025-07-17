//! CUDA storage backend for GPU-accelerated tensor operations.

use super::{Storage, StorageError};
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
    pub fn with_capacity(capacity: usize) -> Result<Self, StorageError> {
        if capacity == 0 {
            return Ok(Self {
                device_ptr: std::ptr::null_mut(),
                len: 0,
                _phantom: std::marker::PhantomData,
            });
        }

        let elem_size = std::mem::size_of::<T>();
        let size = capacity.checked_mul(elem_size).ok_or_else(|| {
            StorageError::AllocationFailed("Capacity too large".to_string())
        })?;

        let mut device_ptr = std::ptr::null_mut();
        let result = unsafe {
            // SAFETY: We've checked that size is non-zero and multiplication didn't overflow
            cuda_driver_sys::cuMemAlloc_v2(&mut device_ptr as *mut _, size)
        };

        if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            return Err(StorageError::CudaError(format!(
                "Failed to allocate {} bytes on device: {:?}",
                size, result
            )));
        }

        Ok(Self {
            device_ptr,
            len: capacity,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Creates a new CUDA storage from a vector.
    pub fn from_vec(data: Vec<T>) -> Result<Self, StorageError> {
        let len = data.len();
        let mut storage = Self::with_capacity(len)?;
        
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
                return Err(StorageError::CudaError(format!(
                    "Failed to copy data to device: {:?}",
                    result
                )));
            }
        }

        Ok(storage)
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
    fn with_capacity(capacity: usize) -> Result<Self, StorageError> {
        Self::with_capacity(capacity)
    }

    fn from_vec(data: Vec<T>) -> Result<Self, StorageError> {
        Self::from_vec(data)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, _index: usize) -> Option<&T> {
        // Getting a reference to device memory is not directly supported
        // In a real implementation, you would need to copy the data back to host memory
        None
    }

    fn get_mut(&mut self, _index: usize) -> Option<&mut T> {
        // Getting a mutable reference to device memory is not directly supported
        // In a real implementation, you would need to copy the data back to host memory
        None
    }

    fn set(&mut self, index: usize, value: T) -> Result<(), StorageError> {
        if index >= self.len {
            return Err(StorageError::ShapeMismatch(format!(
                "Index {} out of bounds for length {}",
                index, self.len
            )));
        }

        // Copy the single value to device memory
        let ptr = unsafe { self.device_ptr.add(index * std::mem::size_of::<T>()) };
        let result = unsafe {
            cuda_driver_sys::cuMemcpyHtoD_v2(
                ptr as u64,
                &value as *const _ as *const _,
                std::mem::size_of::<T>(),
            )
        };

        if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
            return Err(StorageError::CudaError(format!(
                "Failed to set value at index {}: {:?}",
                index, result
            )));
        }

        Ok(())
    }

    fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        if self.len == 0 {
            return Vec::new();
        }

        let mut vec = Vec::with_capacity(self.len);
        let size = self.len * std::mem::size_of::<T>();
        
        // SAFETY: We've allocated the vector with the correct capacity
        unsafe {
            vec.set_len(self.len);
            let result = cuda_driver_sys::cuMemcpyDtoH_v2(
                vec.as_mut_ptr() as *mut _,
                self.device_ptr as u64,
                size,
            );

            if result != cuda_driver_sys::CUresult::CUDA_SUCCESS {
                panic!("Failed to copy data from device: {:?}", result);
            }
        }

        vec
    }

    fn clone(&self) -> std::sync::Arc<dyn Storage<T>> {
        let mut new_storage = Self::with_capacity(self.len)
            .expect("Failed to allocate device memory for clone");
        
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

        std::sync::Arc::new(new_storage)
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
