//! Tensor views for zero-copy operations.
//!
//! This module provides types and traits for creating and manipulating tensor views.
//! Views allow for efficient operations on tensors without copying the underlying data.

use std::fmt;
use std::ops::{Deref, Index, IndexMut};
use crate::error::{Result, TensorustError};
use crate::dimension::{Dimension, DynamicDim, Stride};

mod slice;
mod ops;

pub use slice::{Slice, SliceRange};
pub use ops::ViewOps;

/// A trait for types that can be viewed as a tensor.
pub trait View: fmt::Debug + Send + Sync + 'static {
    /// Returns the underlying data as a slice.
    fn as_slice(&self) -> &[f32];
    
    /// Returns a mutable reference to the underlying data as a slice.
    fn as_mut_slice(&mut self) -> &mut [f32];
    
    /// Returns the shape of the view.
    fn shape(&self) -> &[usize];
    
    /// Returns the strides of the view.
    fn strides(&self) -> &[usize];
    
    /// Returns the number of dimensions.
    fn ndim(&self) -> usize {
        self.shape().len()
    }
    
    /// Returns the total number of elements in the view.
    fn len(&self) -> usize {
        self.shape().iter().product()
    }
    
    /// Returns `true` if the view contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Returns the offset of the given index in the underlying storage.
    fn offset(&self, index: &[usize]) -> Result<usize> {
        if index.len() != self.ndim() {
            return Err(TensorustError::invalid_index(
                index.to_vec(),
                self.shape().to_vec(),
            ));
        }
        
        let mut offset = 0;
        for (i, (&idx, &stride)) in index.iter().zip(self.strides().iter()).enumerate() {
            if idx >= self.shape()[i] {
                return Err(TensorustError::index_out_of_bounds(
                    idx,
                    self.shape()[i],
                    i,
                ));
            }
            offset += idx * stride;
        }
        
        Ok(offset)
    }
    
    /// Returns a reference to the element at the given index.
    fn get(&self, index: &[usize]) -> Result<&f32> {
        let offset = self.offset(index)?;
        self.as_slice()
            .get(offset)
            .ok_or_else(|| TensorustError::invalid_index(index.to_vec(), self.shape().to_vec()))
    }
    
    /// Returns a mutable reference to the element at the given index.
    fn get_mut(&mut self, index: &[usize]) -> Result<&mut f32> {
        let offset = self.offset(index)?;
        self.as_mut_slice()
            .get_mut(offset)
            .ok_or_else(|| TensorustError::invalid_index(index.to_vec(), self.shape().to_vec()))
    }
    
    /// Returns a view of the tensor with the given shape and strides.
    fn view_with_strides(&self, shape: Vec<usize>, strides: Vec<usize>) -> Result<TensorView> {
        if shape.iter().product::<usize>() != self.len() {
            return Err(TensorustError::reshape_error(
                self.shape().to_vec(),
                shape,
            ));
        }
        
        if strides.len() != shape.len() {
            return Err(TensorustError::invalid_input(
                "Shape and strides must have the same length",
            ));
        }
        
        Ok(TensorView {
            data: self.as_slice().as_ptr(),
            shape,
            strides,
            _marker: std::marker::PhantomData,
        })
    }
    
    /// Returns a view of the tensor with the given shape.
    fn view(&self, shape: Vec<usize>) -> Result<TensorView> {
        if shape.iter().product::<usize>() != self.len() {
            return Err(TensorustError::reshape_error(
                self.shape().to_vec(),
                shape.clone(),
            ));
        }
        
        let strides = Stride::row_major(&shape);
        self.view_with_strides(shape, strides.into())
    }
    
    /// Returns a slice of the tensor along the given axis.
    fn slice_axis(&self, axis: usize, range: SliceRange) -> Result<TensorView>;
}

/// A view into a tensor's data.
#[derive(Debug)]
pub struct TensorView<'a> {
    data: *const f32,
    shape: Vec<usize>,
    strides: Vec<usize>,
    _marker: std::marker::PhantomData<&'a f32>,
}

impl<'a> TensorView<'a> {
    /// Creates a new tensor view.
    pub fn new(data: &'a [f32], shape: Vec<usize>) -> Result<Self> {
        let strides = Stride::row_major(&shape);
        Self::with_strides(data, shape, strides.into())
    }
    
    /// Creates a new tensor view with custom strides.
    pub fn with_strides(data: &'a [f32], shape: Vec<usize>, strides: Vec<usize>) -> Result<Self> {
        if shape.iter().product::<usize>() != data.len() {
            return Err(TensorustError::invalid_shape(
                "Shape does not match data length",
            ));
        }
        
        if strides.len() != shape.len() {
            return Err(TensorustError::invalid_input(
                "Shape and strides must have the same length",
            ));
        }
        
        Ok(Self {
            data: data.as_ptr(),
            shape,
            strides,
            _marker: std::marker::PhantomData,
        })
    }
    
    /// Returns the shape of the view.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Returns the strides of the view.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    
    /// Returns the total number of elements in the view.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Returns `true` if the view contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Returns a reference to the element at the given index.
    pub fn get(&self, index: &[usize]) -> Result<&'a f32> {
        if index.len() != self.ndim() {
            return Err(TensorustError::invalid_index(
                index.to_vec(),
                self.shape.to_vec(),
            ));
        }
        
        let mut offset = 0;
        for (i, (&idx, &stride)) in index.iter().zip(self.strides.iter()).enumerate() {
            if idx >= self.shape[i] {
                return Err(TensorustError::index_out_of_bounds(
                    idx,
                    self.shape[i],
                    i,
                ));
            }
            offset += idx * stride;
        }
        
        unsafe { self.data.as_ref().ok_or(TensorustError::invalid_pointer()) }
            .map(|data| &data[offset])
    }
}

impl<'a> View for TensorView<'a> {
    fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.data, self.len()) }
    }
    
    fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.data as *mut f32, self.len()) }
    }
    
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    fn slice_axis(&self, axis: usize, range: SliceRange) -> Result<TensorView> {
        if axis >= self.ndim() {
            return Err(TensorustError::invalid_axis(axis, self.ndim()));
        }
        
        let start = range.start().unwrap_or(0);
        let end = range.end().unwrap_or(self.shape[axis]);
        let step = range.step().unwrap_or(1);
        
        if start >= self.shape[axis] || end > self.shape[axis] || start >= end || step == 0 {
            return Err(TensorustError::invalid_slice(range, self.shape[axis]));
        }
        
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        
        new_shape[axis] = (end - start + step - 1) / step; // Ceiling division
        new_strides[axis] *= step;
        
        let offset = start * self.strides[axis];
        let data = unsafe { self.data.add(offset) };
        
        Ok(TensorView {
            data,
            shape: new_shape,
            strides: new_strides,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<'a> Index<&[usize]> for TensorView<'a> {
    type Output = f32;
    
    fn index(&self, index: &[usize]) -> &Self::Output {
        self.get(index).expect("Index out of bounds")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    
    #[test]
    fn test_tensor_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();
        
        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        assert_eq!(view.get(&[0, 0]).unwrap(), &1.0);
        assert_eq!(view.get(&[0, 1]).unwrap(), &2.0);
        assert_eq!(view.get(&[1, 0]).unwrap(), &4.0);
        assert_eq!(view.get(&[1, 2]).unwrap(), &6.0);
    }
    
    #[test]
    fn test_slice_axis() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();
        
        // Slice first row
        let row = view.slice_axis(0, SliceRange::single(0)).unwrap();
        assert_eq!(row.shape(), &[1, 3]);
        assert_eq!(row.get(&[0, 0]).unwrap(), &1.0);
        assert_eq!(row.get(&[0, 1]).unwrap(), &2.0);
        
        // Slice second column
        let col = view.slice_axis(1, SliceRange::single(1)).unwrap();
        assert_eq!(col.shape(), &[2, 1]);
        assert_eq!(col.get(&[0, 0]).unwrap(), &2.0);
        assert_eq!(col.get(&[1, 0]).unwrap(), &5.0);
    }
    
    #[test]
    fn test_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();
        
        // Reshape to 3x2
        let reshaped = view.view(vec![3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.strides(), &[2, 1]);
        assert_eq!(reshaped.get(&[0, 0]).unwrap(), &1.0);
        assert_eq!(reshaped.get(&[0, 1]).unwrap(), &2.0);
        assert_eq!(reshaped.get(&[1, 0]).unwrap(), &3.0);
    }
}
