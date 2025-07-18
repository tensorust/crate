use thiserror::Error;

#[derive(Error, Debug)]
pub enum TensorustError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Incompatible shapes for operation: {0:?} and {1:?}")]
    IncompatibleShapes(Vec<usize>, Vec<usize>),
    #[error("Invalid shape: {0}")]
    InvalidShape(String),
    #[error("Invalid index: {0:?} for shape {1:?}")]
    InvalidIndex(Vec<usize>, Vec<usize>),
    #[error("Index out of bounds: {0} for dimension of size {1} at axis {2}")]
    IndexOutOfBounds(usize, usize, usize),
    #[error("Invalid axis: {0} for tensor of dimension {1}")]
    InvalidAxis(usize, usize),
    #[error("Duplicate axis: {0}")]
    DuplicateAxis(usize),
    #[error("Invalid axes: {0:?} for tensor of dimension {1}")]
    InvalidAxes(Vec<usize>, usize),
    #[error("Invalid slice: {0:?} for dimension of size {1}")]
    InvalidSlice(crate::view::SliceRange, usize),
    #[error("Reshape error: cannot reshape tensor of size {0:?} to {1:?}")]
    ReshapeError(Vec<usize>, Vec<usize>),
    #[error("Broadcast error: cannot broadcast tensor of shape {0:?} to {1:?}")]
    BroadcastError(Vec<usize>, Vec<usize>),
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    #[error("Invalid pointer")]
    InvalidPointer,
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, TensorustError>;
