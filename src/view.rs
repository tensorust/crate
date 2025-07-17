use crate::{
    dimension::{Dimension, DynamicDim, StaticDim},
    error::{Result, TensorustError},
    storage::Storage,
    tensor::Tensor,
};
use std::marker::PhantomData;
use std::ops::{Bound, Range, RangeBounds};
use std::sync::Arc;

/// A view into a tensor that shares the underlying storage
#[derive(Debug, Clone)]
pub struct TensorView<'a, T, D, S = crate::storage::CpuStorage<T>>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Reference to the original tensor
    tensor: &'a Tensor<T, D, S>,
    /// The shape of this view
    shape: D,
    /// The offset in the original storage
    offset: usize,
    /// The strides for this view
    strides: Vec<isize>,
    _marker: PhantomData<T>,
}

impl<'a, T, D, S> TensorView<'a, T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Create a new view into a tensor
    pub fn new(tensor: &'a Tensor<T, D, S>, shape: D, offset: usize, strides: Vec<isize>) -> Self {
        Self {
            tensor,
            shape,
            offset,
            strides,
            _marker: PhantomData,
        }
    }

    /// Get the shape of this view
    pub fn shape(&self) -> &D::Shape {
        self.shape.shape()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements in this view
    pub fn len(&self) -> usize {
        self.shape.size()
    }

    /// Check if the view is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the underlying tensor
    pub fn tensor(&self) -> &Tensor<T, D, S> {
        self.tensor
    }

    /// Get the offset in the original storage
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the strides of this view
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Convert this view into a new tensor with its own storage
    pub fn to_tensor(&self) -> Tensor<T, D, S>
    where
        T: Default,
    {
        let mut new_storage = S::from_vec(vec![T::default(); self.len()]);
        // Copy data from view to new storage
        // This is a simplified version - in practice, you'd want to handle different storage types
        // and potentially use parallel iterators for large tensors
        for i in 0..self.len() {
            // Calculate the index in the original tensor's storage
            let mut index = self.offset as isize;
            let mut remaining = i;
            
            for (dim_size, stride) in self.shape.shape().as_ref().iter().zip(&self.strides) {
                let pos = remaining % dim_size;
                index += (pos as isize) * stride;
                remaining /= dim_size;
            }
            
            // Get the value from the original tensor and copy it to the new storage
            let value = self.tensor.storage.get(index as usize).unwrap().clone();
            new_storage.get_mut(i).map(|v| *v = value);
        }
        
        Tensor::new(new_storage, self.shape.clone()).unwrap()
    }
}

/// Helper trait for indexing operations
pub trait IndexingExt<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    /// Get a view of a single element
    fn at(&self, indices: &[usize]) -> Result<T>;
    
    /// Get a subview using the given range for each dimension
    fn slice<I, R>(&self, ranges: I) -> Result<TensorView<'_, T, DynamicDim, S>>
    where
        I: IntoIterator<Item = R>,
        R: RangeBounds<usize>;
}

impl<T, D, S> IndexingExt<T, D, S> for Tensor<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    fn at(&self, indices: &[usize]) -> Result<T> {
        if indices.len() != self.dim.ndim() {
            return Err(TensorustError::invalid_dimension(indices.len(), self.dim.ndim()));
        }
        
        let index = self.dim.compute_offset(indices)?;
        self.storage.get(index).cloned()
    }
    
    fn slice<I, R>(&self, ranges: I) -> Result<TensorView<'_, T, DynamicDim, S>>
    where
        I: IntoIterator<Item = R>,
        R: RangeBounds<usize>,
    {
        let ranges: Vec<Range<usize>> = ranges
            .into_iter()
            .map(|r| {
                let start = match r.start_bound() {
                    Bound::Included(&x) => x,
                    Bound::Excluded(&x) => x + 1,
                    Bound::Unbounded => 0,
                };
                let end = match r.end_bound() {
                    Bound::Included(&x) => x + 1,
                    Bound::Excluded(&x) => x,
                    Bound::Unbounded => self.dim.shape().as_ref()[0], // Simplified for 1D
                };
                start..end
            })
            .collect();
            
        // For simplicity, this is a basic 1D implementation
        // A full implementation would handle multi-dimensional slicing
        if ranges.len() != 1 {
            return Err(TensorustError::unsupported_operation(
                "Multi-dimensional slicing not yet implemented",
            ));
        }
        
        let range = &ranges[0];
        let new_shape = DynamicDim::new(vec![range.end - range.start]);
        let offset = self.offset + range.start;
        let strides = self.dim.strides();
        
        Ok(TensorView::new(self, new_shape, offset, strides))
    }
}

// Implement indexing syntax support
impl<T, D, S> std::ops::Index<usize> for Tensor<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    type Output = T;
    
    fn index(&self, index: usize) -> &Self::Output {
        self.storage.get(self.offset + index).unwrap()
    }
}

// Implement slicing syntax support
impl<T, D, S> std::ops::Index<std::ops::Range<usize>> for Tensor<T, D, S>
where
    T: Clone + Send + Sync + 'static,
    D: Dimension,
    S: Storage<T>,
{
    type Output = TensorView<'static, T, DynamicDim, S>;
    
    fn index(&self, range: std::ops::Range<usize>) -> &Self::Output {
        // This is a simplified version - in practice, you'd want to return an owned TensorView
        // and handle the lifetime properly
        todo!("Implement proper indexing with lifetimes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::CpuStorage;

    #[test]
    fn test_view_creation() {
        let tensor = Tensor::<f32, DynamicDim, _>::from(vec![1.0, 2.0, 3.0, 4.0]);
        let view = tensor.slice(1..3).unwrap();
        assert_eq!(view.shape(), &vec![2]);
        assert_eq!(view.to_tensor().to_vec(), vec![2.0, 3.0]);
    }

    #[test]
    fn test_indexing() {
        let tensor = Tensor::<f32, DynamicDim, _>::from(vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor.at(&[1]).unwrap(), 2.0);
    }

    #[test]
    fn test_view_to_tensor() {
        let tensor = Tensor::<f32, DynamicDim, _>::from(vec![1.0, 2.0, 3.0, 4.0]);
        let view = tensor.slice(1..3).unwrap();
        let new_tensor = view.to_tensor();
        assert_eq!(new_tensor.shape(), &vec![2]);
        assert_eq!(new_tensor.to_vec(), vec![2.0, 3.0]);
    }
}
