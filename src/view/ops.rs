//! Operations for tensor views.
//!
//! This module provides various operations that can be performed on tensor views,
//! including element-wise operations, reductions, and transformations.

use std::ops::{Add, Div, Mul, Sub};
use crate::error::{Result, TensorustError};
use crate::dimension::{dynamic::DynamicDim, Dimension};
use crate::view::{Slice, SliceRange, TensorView, View};

/// A trait for operations that can be performed on tensor views.
pub trait ViewOps: View {
    /// Returns a transposed view of the tensor.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of axes. If `None`, the axes are reversed.
    ///
    /// # Returns
    ///
    /// A transposed view of the tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of axes doesn't match the tensor's
    /// dimensionality or if any axis is out of bounds.
    fn transpose(&self, axes: Option<&[usize]>) -> Result<TensorView> {
        let ndim = self.ndim();
        let axes = match axes {
            Some(axes) => {
                if axes.len() != ndim {
                    return Err(TensorustError::invalid_axes(
                        axes.to_vec(),
                        ndim,
                    ));
                }
                axes.to_vec()
            }
            None => (0..ndim).rev().collect(),
        };

        // Validate axes
        let mut seen = vec![false; ndim];
        for &axis in &axes {
            if axis >= ndim {
                return Err(TensorustError::invalid_axis(axis, ndim));
            }
            if seen[axis] {
                return Err(TensorustError::duplicate_axis(axis));
            }
            seen[axis] = true;
        }

        // Compute new shape and strides
        let mut new_shape = vec![0; ndim];
        let mut new_strides = vec![0; ndim];
        
        for (new_axis, &old_axis) in axes.iter().enumerate() {
            new_shape[new_axis] = self.shape()[old_axis];
            new_strides[new_axis] = self.strides()[old_axis];
        }

        Ok(TensorView {
            data: self.as_slice().as_ptr(),
            shape: new_shape,
            strides: new_strides,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns a squeezed view of the tensor, removing dimensions of size 1.
    ///
    /// # Returns
    ///
    /// A new view with singleton dimensions removed.
    fn squeeze(&self) -> TensorView {
        let mut new_shape = Vec::with_capacity(self.ndim());
        let mut new_strides = Vec::with_capacity(self.ndim());
        
        for (&dim, &stride) in self.shape().iter().zip(self.strides().iter()) {
            if dim != 1 {
                new_shape.push(dim);
                new_strides.push(stride);
            }
        }
        
        if new_shape.is_empty() {
            new_shape.push(1);
            new_strides.push(1);
        }
        
        TensorView {
            data: self.as_slice().as_ptr(),
            shape: new_shape,
            strides: new_strides,
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns an unsqueezed view of the tensor, adding a new dimension of size 1.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis at which to insert the new dimension.
    ///
    /// # Returns
    ///
    /// A new view with an additional dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if the axis is out of bounds.
    fn unsqueeze(&self, axis: usize) -> Result<TensorView> {
        let ndim = self.ndim();
        if axis > ndim {
            return Err(TensorustError::invalid_axis(axis, ndim));
        }
        
        let mut new_shape = self.shape().to_vec();
        let mut new_strides = self.strides().to_vec();
        
        new_shape.insert(axis, 1);
        
        // Insert stride 0 for the new dimension to allow broadcasting
        if axis < ndim {
            new_strides.insert(axis, 0);
        } else {
            // If appending at the end, the new dimension has stride 1
            new_strides.push(1);
        }
        
        Ok(TensorView {
            data: self.as_slice().as_ptr(),
            shape: new_shape,
            strides: new_strides,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns a view with the specified shape if compatible.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape.
    ///
    /// # Returns
    ///
    /// A new view with the specified shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the new shape is incompatible with the current size.
    fn reshape(&self, shape: &[usize]) -> Result<TensorView> {
        let size: usize = shape.iter().product();
        if size != self.len() {
            return Err(TensorustError::reshape_error(
                self.shape().to_vec(),
                shape.to_vec(),
            ));
        }
        
        // For contiguous tensors, we can just change the shape
        if self.is_contiguous() {
            return Ok(TensorView::new(self.as_slice(), shape.to_vec())?);
        }
        
        // For non-contiguous tensors, we need to create a new view
        // with the same data but new strides
        let strides = DynamicDim::row_major(shape);
        self.view_with_strides(shape.to_vec(), strides.into())
    }

    /// Returns a view with the specified slice applied.
    ///
    /// # Arguments
    ///
    /// * `slice` - The slice to apply.
    ///
    /// # Returns
    ///
    /// A new view with the slice applied.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice is invalid for this tensor.
    fn slice(&self, slice: &Slice) -> Result<TensorView> {
        slice.validate(self.shape())?;
        
        let mut new_shape = Vec::with_capacity(self.ndim());
        let mut new_strides = self.strides().to_vec();
        let mut offset = 0;
        
        // Apply each range in the slice
        for (i, (range, &stride)) in slice.ranges().iter().zip(self.strides().iter()).enumerate() {
            let dim_size = self.shape()[i];
            let start = range.start().unwrap_or(0);
            let step = range.step().unwrap_or(1);
            
            offset += start * stride;
            new_shape.push(range.len(dim_size)?);
            new_strides[i] = stride * step;
        }
        
        // Add remaining dimensions
        if slice.ranges().len() < self.ndim() {
            new_shape.extend_from_slice(&self.shape()[slice.ranges().len()..]);
        }
        
        Ok(TensorView {
            data: unsafe { self.as_slice().as_ptr().add(offset) },
            shape: new_shape,
            strides: new_strides,
            _marker: std::marker::PhantomData,
        })
    }

    /// Returns `true` if the tensor is stored contiguously in memory.
    fn is_contiguous(&self) -> bool {
        if self.shape().is_empty() {
            return true;
        }
        
        let mut expected_stride = 1;
        
        for (&dim, &stride) in self.shape().iter().zip(self.strides()).rev() {
            if stride != expected_stride && dim > 1 {
                return false;
            }
            expected_stride = dim * stride;
        }
        
        true
    }

    /// Returns a contiguous copy of the tensor if it's not already contiguous.
    fn contiguous(&self) -> TensorView {
        if self.is_contiguous() {
            return TensorView {
                data: self.as_slice().as_ptr(),
                shape: self.shape().to_vec(),
                strides: self.strides().to_vec(),
                _marker: std::marker::PhantomData,
            };
        }
        
        // For non-contiguous tensors, create a new contiguous view
        let data: Vec<f32> = self.as_slice().to_vec();
        let strides = DynamicDim::row_major(self.shape());
        
        TensorView {
            data: data.as_ptr(),
            shape: self.shape().to_vec(),
            strides: strides.into(),
            _marker: std::marker::PhantomData,
        }
    }
}

// Implement ViewOps for all types that implement View
impl<T: View> ViewOps for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::TensorView;
    
    #[test]
    fn test_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();
        
        // Transpose without specifying axes (reverses axes)
        let transposed = view.transpose(None).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
        
        // Transpose with specific axes
        let transposed = view.transpose(Some(&[1, 0])).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.strides(), &[1, 3]);
    }
    
    #[test]
    fn test_squeeze() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let view = TensorView::new(&data, vec![1, 2, 1, 2]).unwrap();
        
        let squeezed = view.squeeze();
        assert_eq!(squeezed.shape(), &[2, 2]);
        assert_eq!(squeezed.strides(), &[2, 1]);
    }
    
    #[test]
    fn test_unsqueeze() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let view = TensorView::new(&data, vec![2, 2]).unwrap();
        
        // Add dimension at the beginning
        let unsqueezed = view.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 2, 2]);
        assert_eq!(unsqueezed.strides(), &[0, 2, 1]);
        
        // Add dimension in the middle
        let unsqueezed = view.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 1, 2]);
        assert_eq!(unsqueezed.strides(), &[2, 0, 1]);
        
        // Add dimension at the end
        let unsqueezed = view.unsqueeze(2).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 2, 1]);
        assert_eq!(unsqueezed.strides(), &[2, 1, 1]);
    }
    
    #[test]
    fn test_reshape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();
        
        // Reshape to 3x2
        let reshaped = view.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        
        // Invalid reshape
        assert!(view.reshape(&[4, 2]).is_err());
    }
    
    #[test]
    fn test_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();
        
        // Slice first row
        let slice = Slice::new(vec![SliceRange::single(0)]);
        let sliced = view.slice(&slice).unwrap();
        assert_eq!(sliced.shape(), &[1, 3]);
        assert_eq!(sliced.strides(), &[3, 1]);
        
        // Slice last column
        let slice = Slice::new(vec![SliceRange::all(), SliceRange::single(2)]);
        let sliced = view.slice(&slice).unwrap();
        assert_eq!(sliced.shape(), &[2, 1]);
        assert_eq!(sliced.strides(), &[3, 1]);
    }
    
    #[test]
    fn test_contiguous() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::with_strides(&data, vec![2, 3], vec![3, 1]).unwrap();
        
        // Original view is contiguous
        assert!(view.is_contiguous());
        
        // Transposed view is not contiguous
        let transposed = view.transpose(None).unwrap();
        assert!(!transposed.is_contiguous());
        
        // Contiguous copy of transposed view
        let contiguous = transposed.contiguous();
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.shape(), &[3, 2]);
    }
}
