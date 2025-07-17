//! Macros for creating and manipulating tensors with a convenient syntax.
//! This module provides several macros to make tensor operations more ergonomic.

/// Creates a new tensor from a list of values with inferred dimensions.
///
/// # Examples
/// ```
/// use tensorust::*;
///
/// // Create a 1D tensor
/// let t = tensor!([1.0, 2.0, 3.0]);
/// assert_eq!(t.shape(), &[3]);
///
/// // Create a 2D tensor
/// let t = tensor!([[1.0, 2.0], [3.0, 4.0]]);
/// assert_eq!(t.shape(), &[2, 2]);
/// ```
#[macro_export]
macro_rules! tensor {
    // Handle 1D case
    ([$($x:expr),+ $(,)?]) => {
        $crate::Tensor::from(vec![$($x as f32),+])
    };
    
    // Handle 2D case
    ([$([$($x:expr),+ $(,)?]),+ $(,)?]) => {
        {
            let data = vec![$(
                vec![$($x as f32),+]
            ),+];
            let rows = data.len();
            let cols = if rows > 0 { data[0].len() } else { 0 };
            let flat_data: Vec<f32> = data.into_iter().flatten().collect();
            $crate::Tensor::from(flat_data).reshape([rows, cols].as_ref()).unwrap()
        }
    };
    
    // Handle 3D case
    ([$([$([$($x:expr),+ $(,)?]),+ $(,)?]),+ $(,)?]) => {
        {
            let mut data = Vec::new();
            let mut dims = vec![];
            
            // First dimension
            dims.push(0);
            
            for dim1 in [$(
                vec![$(
                    vec![$($x as f32),+]
                ),+]
            ),+] {
                dims[0] += 1;
                
                // Second dimension
                if dims.len() == 1 {
                    dims.push(dim1.len());
                } else if dim1.len() != dims[1] {
                    panic!("Inconsistent dimensions in tensor macro");
                }
                
                // Third dimension
                if dims.len() == 2 {
                    dims.push(dim1[0].len());
                }
                
                for row in dim1 {
                    if row.len() != dims[2] {
                        panic!("Inconsistent dimensions in tensor macro");
                    }
                    data.extend(row);
                }
            }
            
            $crate::Tensor::from(data).reshape(dims.as_slice()).unwrap()
        }
    };
}

/// Creates a tensor filled with zeros.
///
/// # Examples
/// ```
/// use tensorust::*;
///
/// // Create a 2x3 tensor of zeros
/// let t = zeros!([2, 3]);
/// assert_eq!(t.shape(), &[2, 3]);
/// ```
#[macro_export]
macro_rules! zeros {
    ([$($dim:expr),+ $(,)?]) => {
        $crate::Tensor::zeros([$($dim),+].as_ref()).unwrap()
    };
}

/// Creates a tensor filled with ones.
///
/// # Examples
/// ```
/// use tensorust::*;
///
/// // Create a 3x3 tensor of ones
/// let t = ones!([3, 3]);
/// assert_eq!(t.shape(), &[3, 3]);
/// ```
#[macro_export]
macro_rules! ones {
    ([$($dim:expr),+ $(,)?]) => {
        $crate::Tensor::ones([$($dim),+].as_ref()).unwrap()
    };
}

/// Creates an identity matrix (2D tensor).
///
/// # Examples
/// ```
/// use tensorust::*;
///
/// // Create a 3x3 identity matrix
/// let t = eye!(3);
/// assert_eq!(t.shape(), &[3, 3]);
/// ```
#[macro_export]
macro_rules! eye {
    ($n:expr) => {
        {
            let n = $n;
            let mut data = vec![0.0; n * n];
            for i in 0..n {
                data[i * n + i] = 1.0;
            }
            $crate::Tensor::from(data).reshape([n, n].as_ref()).unwrap()
        }
    };
}

/// Creates a tensor with values in a range.
///
/// # Examples
/// ```
/// use tensorust::*;
///
/// // Create a tensor with values from 0 to 4
/// let t = range!(5);
/// assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
///
/// // Create a tensor with values from 1 to 5
/// let t = range!(1, 6);
/// assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
///
/// // Create a tensor with values from 0 to 10 with step 2
/// let t = range!(0, 10, 2);
/// assert_eq!(t.to_vec(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
/// ```
#[macro_export]
macro_rules! range {
    ($end:expr) => {
        $crate::range!(0, $end, 1)
    };
    ($start:expr, $end:expr) => {
        $crate::range!($start, $end, 1)
    };
    ($start:expr, $end:expr, $step:expr) => {
        {
            let start = $start as f32;
            let end = $end as f32;
            let step = $step as f32;
            let len = ((end - start) / step).ceil() as usize;
            let data: Vec<f32> = (0..len)
                .map(|i| start + (i as f32) * step)
                .collect();
            $crate::Tensor::from(data)
        }
    };
}

/// Creates a tensor with random values between 0 and 1.
///
/// # Examples
/// ```
/// use tensorust::*;
///
/// // Create a 2x3 tensor with random values
/// let t = rand!([2, 3]);
/// assert_eq!(t.shape(), &[2, 3]);
/// ```
#[macro_export]
macro_rules! rand {
    ([$($dim:expr),+ $(,)?]) => {
        {
            use rand::Rng;
            let size = [$($dim),+].iter().product();
            let mut rng = rand::thread_rng();
            let data: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
            $crate::Tensor::from(data).reshape([$($dim),+].as_ref()).unwrap()
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_macro_1d() {
        let t = tensor!([1.0, 2.0, 3.0]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tensor_macro_2d() {
        let t = tensor!([
            [1.0, 2.0],
            [3.0, 4.0]
        ]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_zeros_macro() {
        let t = zeros!([2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec(), vec![0.0; 6]);
    }

    #[test]
    fn test_ones_macro() {
        let t = ones!([2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.to_vec(), vec![1.0; 4]);
    }

    #[test]
    fn test_eye_macro() {
        let t = eye!(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(
            t.to_vec(),
            vec![
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0
            ]
        );
    }
}
