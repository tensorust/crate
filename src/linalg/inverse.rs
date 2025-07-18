//! Matrix inverse and determinant operations.

use crate::{
    dimension::{static_dim::StaticDim, Dimension},
    error::{Result, TensorustError},
    tensor::Tensor,
    storage::Storage,
};
use num_traits::{One, Zero};

/// Compute the determinant of a square matrix.
pub fn determinant<T, S>(
    tensor: &Tensor<T, StaticDim<2>, S>,
) -> Result<T>
where
    T: Clone + Default + One + Zero + PartialEq + std::fmt::Debug + Send + Sync + 'static,
    for<'a> &'a T: std::ops::Mul<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T>,
    S: Storage<T>,
{
    let shape = tensor.shape();
    let n = shape[0];
    
    // Check if the matrix is square
    if n != shape[1] {
        return Err(TensorustError::invalid_shape(
            "Determinant is only defined for square matrices",
        ));
    }
    
    // For small matrices, use direct formulas
    match n {
        0 => Ok(T::zero()),
        1 => {
            let data = tensor.storage().to_vec();
            Ok(data[0].clone())
        },
        2 => {
            let data = tensor.storage();
            // |a b|
            // |c d| = ad - bc
            let a = &data[0];
            let b = &data[1];
            let c = &data[2];
            let d = &data[3];
            
            Ok(a * d - b * c)
        },
        3 => {
            let data = tensor.storage();
            // |a b c|
            // |d e f| = a(ei - fh) - b(di - fg) + c(dh - eg)
            // |g h i|
            let a = &data[0];
            let b = &data[1];
            let c = &data[2];
            let d = &data[3];
            let e = &data[4];
            let f = &data[5];
            let g = &data[6];
            let h = &data[7];
            let i = &data[8];
            
            let term1 = a * &(e * i - f * h);
            let term2 = b * &(d * i - f * g);
            let term3 = c * &(d * h - e * g);
            
            Ok(term1 - term2 + term3)
        },
        _ => {
            // For larger matrices, use LU decomposition
            // This is a simplified version - in production, you'd want to use a more
            // efficient method like LU decomposition with partial pivoting
            
            // Create a copy of the matrix data
            let mut lu = tensor.storage().to_vec();
            let n = n as isize;
            let mut det = T::one();
            
            for i in 0..n {
                // Partial pivoting
                let mut max = i;
                for j in (i + 1)..n {
                    if lu[(j * n + i) as usize].abs() > lu[(max * n + i) as usize].abs() {
                        max = j;
                    }
                }
                
                // Swap rows if needed
                if max != i {
                    for k in 0..n {
                        lu.swap((i * n + k) as usize, (max * n + k) as usize);
                    }
                    // Multiply determinant by -1 for row swap
                    det = T::zero() - det;
                }
                
                let pivot = lu[(i * n + i) as usize].clone();
                
                // If the pivot is zero, the matrix is singular
                if pivot == T::zero() {
                    return Ok(T::zero());
                }
                
                // Update determinant
                det = det * pivot.clone();
                
                // Perform elimination
                for j in (i + 1)..n {
                    let factor = lu[(j * n + i) as usize].clone() / pivot.clone();
                    lu[(j * n + i) as usize] = factor.clone();
                    
                    for k in (i + 1)..n {
                        let idx1 = (j * n + k) as usize;
                        let idx2 = (i * n + k) as usize;
                        lu[idx1] = lu[idx1].clone() - factor.clone() * lu[idx2].clone();
                    }
                }
            }
            
            Ok(det)
        }
    }
}

/// Compute the inverse of a square matrix.
pub fn inverse<T, S>(
    tensor: &Tensor<T, StaticDim<2>, S>,
) -> Result<Tensor<T, StaticDim<2>, S>>
where
    T: Clone + Default + One + Zero + PartialEq + std::fmt::Debug + Send + Sync + 'static,
    for<'a> &'a T: std::ops::Mul<Output = T> 
                  + std::ops::Sub<Output = T> 
                  + std::ops::Div<Output = T>
                  + std::ops::Add<Output = T>,
    S: Storage<T>,
{
    let shape = tensor.shape();
    let n = shape[0];
    
    // Check if the matrix is square
    if n != shape[1] {
        return Err(TensorustError::invalid_shape(
            "Inverse is only defined for square matrices",
        ));
    }
    
    // For small matrices, use direct formulas
    match n {
        0 => Ok(tensor.clone()),
        1 => {
            let data = tensor.storage().to_vec();
            let inv_val = T::one() / data[0].clone();
            let result_storage = S::from_vec(vec![inv_val]);
            Tensor::new(result_storage, [1, 1].into())
        },
        2 => {
            let data = tensor.storage();
            // [a b]^-1   =  1/(ad - bc) [d -b]
            // [c d]                    [-c a]
            let a = &data[0];
            let b = &data[1];
            let c = &data[2];
            let d = &data[3];
            
            let det = a * d - b * c;
            
            if det == T::zero() {
                return Err(TensorustError::invalid_argument(
                    "Matrix is not invertible (determinant is zero)",
                ));
            }
            
            let inv_det = T::one() / det;
            
            let result_data = vec![
                d.clone() * inv_det.clone(),
                (T::zero() - b.clone()) * inv_det.clone(),
                (T::zero() - c.clone()) * inv_det.clone(),
                a.clone() * inv_det.clone(),
            ];
            
            let result_storage = S::from_vec(result_data);
            Tensor::new(result_storage, [2, 2].into())
        },
        _ => {
            // For larger matrices, use LU decomposition with partial pivoting
            // This is a simplified version - in production, you'd want to use a more
            // efficient method like LAPACK or a similar library
            
            // Create augmented matrix [A | I]
            let mut aug = vec![T::default(); n * n * 2];
            let data = tensor.storage().to_vec();
            
            // Copy A into the left half
            for i in 0..n {
                for j in 0..n {
                    aug[i * n * 2 + j] = data[i * n + j].clone();
                }
                // Set up identity matrix in the right half
                aug[i * n * 2 + n + i] = T::one();
            }
            
            // Perform Gaussian elimination
            for i in 0..n {
                // Partial pivoting
                let mut max = i;
                for j in (i + 1)..n {
                    if aug[j * n * 2 + i].abs() > aug[max * n * 2 + i].abs() {
                        max = j;
                    }
                }
                
                // Swap rows if needed
                if max != i {
                    for k in 0..n * 2 {
                        aug.swap(i * n * 2 + k, max * n * 2 + k);
                    }
                }
                
                let pivot = aug[i * n * 2 + i].clone();
                
                // If the pivot is zero, the matrix is not invertible
                if pivot == T::zero() {
                    return Err(TensorustError::invalid_argument(
                        "Matrix is not invertible (singular matrix)",
                    ));
                }
                
                // Normalize the pivot row
                for k in 0..n * 2 {
                    aug[i * n * 2 + k] = aug[i * n * 2 + k].clone() / pivot.clone();
                }
                
                // Eliminate other rows
                for j in 0..n {
                    if j != i && aug[j * n * 2 + i] != T::zero() {
                        let factor = aug[j * n * 2 + i].clone();
                        for k in 0..n * 2 {
                            aug[j * n * 2 + k] = aug[j * n * 2 + k].clone() - aug[i * n * 2 + k].clone() * factor.clone();
                        }
                    }
                }
            }
            
            // Extract the inverse from the right half of the augmented matrix
            let mut inv_data = vec![T::default(); n * n];
            for i in 0..n {
                for j in 0..n {
                    inv_data[i * n + j] = aug[i * n * 2 + n + j].clone();
                }
            }
            
            let result_storage = S::from_vec(inv_data);
            Tensor::new(result_storage, [n, n].into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dimension::StaticDim,
        storage::CpuStorage,
        tensor,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_determinant_2x2() {
        let a = tensor!([
            [1.0, 2.0],
            [3.0, 4.0]
        ]);
        let det = determinant(&a).unwrap();
        assert_relative_eq!(det, -2.0);
    }

    #[test]
    fn test_determinant_3x3() {
        let a = tensor!([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]);
        let det = determinant(&a).unwrap();
        assert_relative_eq!(det, 0.0); // This matrix is singular
    }

    #[test]
    fn test_inverse_2x2() {
        let a = tensor!([
            [4.0, 7.0],
            [2.0, 6.0]
        ]);
        let a_inv = inverse(&a).unwrap();
        
        // Expected inverse:
        // [ 0.6 -0.7]
        // [-0.2  0.4]
        assert_relative_eq!(a_inv[[0, 0]], 0.6, epsilon = 1e-6);
        assert_relative_eq!(a_inv[[0, 1]], -0.7, epsilon = 1e-6);
        assert_relative_eq!(a_inv[[1, 0]], -0.2, epsilon = 1e-6);
        assert_relative_eq!(a_inv[[1, 1]], 0.4, epsilon = 1e-6);
        
        // Verify A * A^-1 = I
        let identity = a.matmul(&a_inv).unwrap();
        assert_relative_eq!(identity[[0, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(identity[[0, 1]], 0.0, epsilon = 1e-6);
        assert_relative_eq!(identity[[1, 0]], 0.0, epsilon = 1e-6);
        assert_relative_eq!(identity[[1, 1]], 1.0, epsilon = 1e-6);
    }
}
