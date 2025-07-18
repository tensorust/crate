//! Slice operations for tensor views.
//!
//! This module provides types and utilities for working with tensor slices,
//! including range-based indexing and strided access.

use std::fmt;
use std::ops::{Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeTo};
use crate::error::{Result, TensorustError};

/// A range that can be used for slicing tensors.
///
/// This is similar to `std::ops::Range` but supports more flexible bounds
/// and step sizes for tensor operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SliceRange {
    start: Option<usize>,
    end: Option<usize>,
    step: usize,
}

impl SliceRange {
    /// Creates a new slice range with the given bounds and step size.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index (inclusive). If `None`, defaults to 0.
    /// * `end` - The ending index (exclusive). If `None`, defaults to the length of the dimension.
    /// * `step` - The step size. Must be greater than 0.
    ///
    /// # Returns
    ///
    /// A new `SliceRange` with the specified bounds and step size.
    ///
    /// # Panics
    ///
    /// Panics if `step` is 0.
    pub fn new(start: Option<usize>, end: Option<usize>, step: usize) -> Self {
        assert_ne!(step, 0, "Step cannot be zero");
        Self { start, end, step }
    }
    
    /// Creates a new slice range that selects a single index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to select.
    ///
    /// # Returns
    ///
    /// A new `SliceRange` that selects only the specified index.
    pub fn single(index: usize) -> Self {
        Self {
            start: Some(index),
            end: Some(index + 1),
            step: 1,
        }
    }
    
    /// Creates a new slice range that selects all elements.
    ///
    /// # Returns
    ///
    /// A new `SliceRange` that selects all elements with a step size of 1.
    pub fn all() -> Self {
        Self {
            start: None,
            end: None,
            step: 1,
        }
    }
    
    /// Returns the start index, defaulting to 0 if not specified.
    pub fn start(&self) -> Option<usize> {
        self.start
    }
    
    /// Returns the end index, defaulting to the length of the dimension if not specified.
    pub fn end(&self) -> Option<usize> {
        self.end
    }
    
    /// Returns the step size.
    pub fn step(&self) -> Option<usize> {
        if self.step == 1 {
            None
        } else {
            Some(self.step)
        }
    }
    
    /// Converts this slice range to a `std::ops::Range` for a given dimension size.
    ///
    /// # Arguments
    ///
    /// * `dim_size` - The size of the dimension being sliced.
    ///
    /// # Returns
    ///
    /// A `Range<usize>` representing the slice range for the given dimension size.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds for the given dimension size.
    pub fn to_range(&self, dim_size: usize) -> Result<Range<usize>> {
        let start = self.start.unwrap_or(0);
        let end = self.end.unwrap_or(dim_size);
        
        if start > dim_size || end > dim_size || start > end {
            return Err(TensorustError::invalid_slice(self.clone(), dim_size));
        }
        
        Ok(start..end)
    }
    
    /// Returns the number of elements in this slice range for a given dimension size.
    ///
    /// # Arguments
    ///
    /// * `dim_size` - The size of the dimension being sliced.
    ///
    /// # Returns
    ///
    /// The number of elements in the slice range.
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds for the given dimension size.
    pub fn len(&self, dim_size: usize) -> Result<usize> {
        let range = self.to_range(dim_size)?;
        if range.is_empty() {
            return Ok(0);
        }
        
        let len = (range.end - range.start + self.step - 1) / self.step;
        Ok(len)
    }
}

impl Default for SliceRange {
    fn default() -> Self {
        Self::all()
    }
}

impl From<Range<usize>> for SliceRange {
    fn from(range: Range<usize>) -> Self {
        Self {
            start: Some(range.start),
            end: Some(range.end),
            step: 1,
        }
    }
}

impl From<RangeTo<usize>> for SliceRange {
    fn from(range: RangeTo<usize>) -> Self {
        Self {
            start: None,
            end: Some(range.end),
            step: 1,
        }
    }
}

impl From<RangeFrom<usize>> for SliceRange {
    fn from(range: RangeFrom<usize>) -> Self {
        Self {
            start: Some(range.start),
            end: None,
            step: 1,
        }
    }
}

impl From<RangeFull> for SliceRange {
    fn from(_: RangeFull) -> Self {
        Self::all()
    }
}


/// A slice of a tensor along one or more dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Slice {
    ranges: Vec<SliceRange>,
}

impl Slice {
    /// Creates a new slice with the given ranges.
    pub fn new(ranges: Vec<SliceRange>) -> Self {
        Self { ranges }
    }
    
    /// Creates a new slice that selects a single index along the first dimension.
    pub fn single(index: usize) -> Self {
        Self {
            ranges: vec![SliceRange::single(index)],
        }
    }
    
    /// Returns the number of dimensions in this slice.
    pub fn ndim(&self) -> usize {
        self.ranges.len()
    }
    
    /// Returns the ranges in this slice.
    pub fn ranges(&self) -> &[SliceRange] {
        &self.ranges
    }
    
    /// Validates this slice against a given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor being sliced.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the slice is valid for the given shape, or an error if not.
    pub fn validate(&self, shape: &[usize]) -> Result<()> {
        if self.ranges.len() > shape.len() {
            return Err(TensorustError::invalid_index(
                vec![0; self.ranges.len()],
                shape.to_vec(),
            ));
        }
        
        for (i, (range, &dim_size)) in self.ranges.iter().zip(shape.iter()).enumerate() {
            if let Some(start) = range.start() {
                if start >= dim_size {
                    return Err(TensorustError::index_out_of_bounds(start, dim_size, i));
                }
            }
            
            if let Some(end) = range.end() {
                if end > dim_size {
                    return Err(TensorustError::index_out_of_bounds(
                        end.saturating_sub(1),
                        dim_size,
                        i,
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Computes the shape of the result of applying this slice.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor being sliced.
    ///
    /// # Returns
    ///
    /// The shape of the resulting tensor after applying this slice.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice is invalid for the given shape.
    pub fn output_shape(&self, shape: &[usize]) -> Result<Vec<usize>> {
        self.validate(shape)?;
        
        let mut output_shape = Vec::with_capacity(shape.len());
        
        // Handle sliced dimensions
        for (range, &dim_size) in self.ranges.iter().zip(shape.iter()) {
            let len = range.len(dim_size)?;
            if len > 0 || dim_size == 0 {
                output_shape.push(len);
            }
        }
        
        // Add remaining dimensions
        if self.ranges.len() < shape.len() {
            output_shape.extend_from_slice(&shape[self.ranges.len()..]);
        }
        
        Ok(output_shape)
    }
}

impl From<SliceRange> for Slice {
    fn from(range: SliceRange) -> Self {
        Self { ranges: vec![range] }
    }
}

impl<const N: usize> From<[SliceRange; N]> for Slice {
    fn from(ranges: [SliceRange; N]) -> Self {
        Self { ranges: ranges.to_vec() }
    }
}

impl From<Vec<SliceRange>> for Slice {
    fn from(ranges: Vec<SliceRange>) -> Self {
        Self { ranges }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_slice_range_new() {
        let range = SliceRange::new(Some(1), Some(4), 1);
        assert_eq!(range.start(), Some(1));
        assert_eq!(range.end(), Some(4));
        assert_eq!(range.step(), None);
    }
    
    #[test]
    fn test_slice_range_single() {
        let range = SliceRange::single(2);
        assert_eq!(range.start(), Some(2));
        assert_eq!(range.end(), Some(3));
        assert_eq!(range.step(), None);
    }
    
    #[test]
    fn test_slice_range_all() {
        let range = SliceRange::all();
        assert_eq!(range.start(), None);
        assert_eq!(range.end(), None);
        assert_eq!(range.step(), None);
    }
    
    #[test]
    fn test_slice_range_to_range() {
        let range = SliceRange::new(Some(1), Some(4), 1);
        assert_eq!(range.to_range(5).unwrap(), 1..4);
        
        let range = SliceRange::new(None, Some(3), 1);
        assert_eq!(range.to_range(5).unwrap(), 0..3);
        
        let range = SliceRange::new(Some(2), None, 1);
        assert_eq!(range.to_range(5).unwrap(), 2..5);
        
        let range = SliceRange::new(None, None, 1);
        assert_eq!(range.to_range(5).unwrap(), 0..5);
    }
    
    #[test]
    fn test_slice_range_len() {
        let range = SliceRange::new(Some(1), Some(4), 1);
        assert_eq!(range.len(5).unwrap(), 3);
        
        let range = SliceRange::new(Some(1), Some(4), 2);
        assert_eq!(range.len(5).unwrap(), 2);
        
        let range = SliceRange::new(Some(1), Some(5), 2);
        assert_eq!(range.len(5).unwrap(), 2);
    }
    
    #[test]
    fn test_slice_validate() {
        let slice = Slice::new(vec![SliceRange::new(Some(1), Some(3), 1)]);
        assert!(slice.validate(&[5]).is_ok());
        assert!(slice.validate(&[2]).is_err());
        
        let slice = Slice::new(vec![SliceRange::all(), SliceRange::single(1)]);
        assert!(slice.validate(&[5, 3]).is_ok());
        assert!(slice.validate(&[5]).is_err());
    }
    
    #[test]
    fn test_slice_output_shape() {
        let slice = Slice::new(vec![SliceRange::new(Some(1), Some(3), 1)]);
        assert_eq!(slice.output_shape(&[5]).unwrap(), vec![2]);
        
        let slice = Slice::new(vec![SliceRange::all(), SliceRange::single(1)]);
        assert_eq!(slice.output_shape(&[5, 3]).unwrap(), vec![5]);
        
        let slice = Slice::new(vec![SliceRange::all(), SliceRange::all()]);
        assert_eq!(slice.output_shape(&[5, 3]).unwrap(), vec![5, 3]);
    }
}
