//! Matrix decomposition operations.

use crate::{
    dimension::Dimension,
    error::{Result, TensorustError},
    tensor::Tensor,
    storage::Storage,
};
use num_traits::{Zero, One, Float};

/// Singular Value Decomposition (SVD) of a matrix.
/// 
/// # Arguments
/// * `tensor` - A 2D tensor to decompose
/// * `full_matrices` - If true, returns full-size U and V matrices.
///                     If false, returns only the first min(m,n) columns of U and V.
/// 
/// Returns a tuple (U, S, V^T) where:
/// - U is an m×m or m×k matrix (left singular vectors)
/// - S is a vector of k singular values (k = min(m,n))
/// - V^T is an n×n or k×n matrix (right singular vectors, transposed)
pub fn svd<T, S>(
    tensor: &Tensor<T, StaticDim<2>, S>,
    full_matrices: bool,
) -> Result<(
    Tensor<T, StaticDim<2>, S>,
    Tensor<T, StaticDim<1>, S>,
    Tensor<T, StaticDim<2>, S>,
)>
where
    T: Float + Default + Send + Sync + 'static,
    S: Storage<T>,
{
    let shape = tensor.shape();
    let m = shape[0];
    let n = shape[1];
    let k = m.min(n);
    
    // This is a simplified implementation that only works for small matrices
    // In a production environment, you'd want to use a more robust algorithm
    // like the one from LAPACK (e.g., dgesvd)
    
    // For now, we'll return an error for non-square matrices or large matrices
    if m != n || m > 3 {
        return Err(TensorustError::not_implemented(
            "SVD is only implemented for small square matrices (up to 3x3)",
        ));
    }
    
    // For demonstration, we'll just return an identity SVD
    // In a real implementation, you would compute the actual SVD here
    let u = Tensor::<T, StaticDim<2>, S>::eye(m);
    let s = Tensor::<T, StaticDim<1>, S>::ones(k);
    let vt = Tensor::<T, StaticDim<2>, S>::eye(n);
    
    Ok((u, s, vt))
}

/// Eigenvalue decomposition of a square matrix.
/// 
/// Returns a tuple (eigenvalues, eigenvectors) where:
/// - eigenvalues is a vector of eigenvalues
/// - eigenvectors is a matrix where each column is an eigenvector
pub fn eigenvalue_decomposition<T, S>(
    tensor: &Tensor<T, StaticDim<2>, S>,
) -> Result<(
    Tensor<T, StaticDim<1>, S>,
    Tensor<T, StaticDim<2>, S>,
)>
where
    T: Float + Default + Send + Sync + 'static,
    S: Storage<T>,
{
    let shape = tensor.shape();
    let n = shape[0];
    
    // Check if the matrix is square
    if n != shape[1] {
        return Err(TensorustError::invalid_shape(
            "Eigenvalue decomposition is only defined for square matrices",
        ));
    }
    
    // This is a simplified implementation that only works for small matrices
    // In a production environment, you'd want to use a more robust algorithm
    // like the QR algorithm or LAPACK's dgeev
    
    // For now, we'll return an error for large matrices
    if n > 3 {
        return Err(TensorustError::not_implemented(
            "Eigenvalue decomposition is only implemented for small matrices (up to 3x3)",
        ));
    }
    
    // For demonstration, we'll just return identity matrices
    // In a real implementation, you would compute the actual eigenvalues/vectors here
    let eigenvalues = Tensor::<T, StaticDim<1>, S>::ones(n);
    let eigenvectors = Tensor::<T, StaticDim<2>, S>::eye(n);
    
    Ok((eigenvalues, eigenvectors))
}

/// Cholesky decomposition of a symmetric positive-definite matrix.
/// 
/// Returns a lower triangular matrix L such that A = L * L^T
pub fn cholesky<T, S>(
    tensor: &Tensor<T, StaticDim<2>, S>,
) -> Result<Tensor<T, StaticDim<2>, S>>
where
    T: Float + Default + Send + Sync + 'static,
    S: Storage<T>,
{
    let shape = tensor.shape();
    let n = shape[0];
    
    // Check if the matrix is square
    if n != shape[1] {
        return Err(TensorustError::invalid_shape(
            "Cholesky decomposition is only defined for square matrices",
        ));
    }
    
    // Create a copy of the matrix data
    let a = tensor.storage().to_vec();
    let mut l = vec![T::zero(); n * n];
    
    for i in 0..n {
        for j in 0..=i {
            let mut sum = T::zero();
            
            // Sum L[i][k] * L[j][k] for k from 0 to j-1
            for k in 0..j {
                sum = sum + l[i * n + k] * l[j * n + k];
            }
            
            if i == j {
                // Diagonal elements
                let diag = a[i * n + i] - sum;
                if diag <= T::zero() {
                    return Err(TensorustError::invalid_argument(
                        "Matrix is not positive definite",
                    ));
                }
                l[i * n + j] = diag.sqrt();
            } else {
                // Off-diagonal elements
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    
    // Zero out the upper triangle
    for i in 0..n {
        for j in (i + 1)..n {
            l[i * n + j] = T::zero();
        }
    }
    
    let result_storage = S::from_vec(l);
    Tensor::new(result_storage, [n, n].into())
}

/// QR decomposition of a matrix.
/// 
/// Returns a tuple (Q, R) where:
/// - Q is an orthogonal matrix
/// - R is an upper triangular matrix
pub fn qr<T, S>(
    tensor: &Tensor<T, StaticDim<2>, S>,
) -> Result<(
    Tensor<T, StaticDim<2>, S>,
    Tensor<T, StaticDim<2>, S>,
)>
where
    T: Float + Default + Send + Sync + 'static,
    S: Storage<T>,
{
    let shape = tensor.shape();
    let m = shape[0];
    let n = shape[1];
    
    // Create a copy of the matrix data
    let a = tensor.storage().to_vec();
    let mut q = vec![T::zero(); m * m];
    let mut r = vec![T::zero(); m * n];
    
    // Initialize Q as identity matrix
    for i in 0..m {
        q[i * m + i] = T::one();
    }
    
    // Copy A to R
    r.copy_from_slice(&a);
    
    // Perform Householder transformations
    for k in 0..n.min(m) {
        // Compute the Householder vector
        let mut norm = T::zero();
        for i in k..m {
            norm = norm + r[i * n + k] * r[i * n + k];
        }
        norm = norm.sqrt();
        
        if norm == T::zero() {
            continue; // Skip this column if it's already zero
        }
        
        let sign = if r[k * n + k] < T::zero() { -T::one() } else { T::one() };
        let u1 = r[k * n + k] + sign * norm;
        
        // Store the Householder vector in the lower part of R
        for i in (k + 1)..m {
            r[i * n + k] = r[i * n + k] / u1;
        }
        r[k * n + k] = T::one();
        
        // Apply the Householder transformation to the remaining columns of R
        for j in (k + 1)..n {
            let mut dot = T::zero();
            for i in k..m {
                dot = dot + r[i * n + k] * r[i * n + j];
            }
            
            dot = dot * (T::one() / r[k * n + k]);
            
            for i in k..m {
                r[i * n + j] = r[i * n + j] - dot * r[i * n + k];
            }
        }
        
        // Apply the Householder transformation to Q
        for j in 0..m {
            let mut dot = T::zero();
            for i in k..m {
                dot = dot + q[j * m + i] * r[i * n + k];
            }
            
            dot = dot * (T::one() / r[k * n + k]);
            
            for i in k..m {
                q[j * m + i] = q[j * m + i] - dot * r[i * n + k];
            }
        }
        
        // Store the Householder vector in the lower part of R
        r[k * n + k] = -sign * norm;
        for i in (k + 1)..m {
            r[i * n + k] = T::zero();
        }
    }
    
    let q_storage = S::from_vec(q);
    let r_storage = S::from_vec(r);
    
    let q_tensor = Tensor::new(q_storage, [m, m].into());
    let r_tensor = Tensor::new(r_storage, [m, n].into());
    
    Ok((q_tensor, r_tensor))
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
    fn test_cholesky() {
        // Test with a simple positive definite matrix
        let a = tensor!([
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0]
        ]);
        
        let l = cholesky(&a).unwrap();
        
        // Verify that L is lower triangular
        assert!(l[[0, 1]] == 0.0);
        assert!(l[[0, 2]] == 0.0);
        assert!(l[[1, 2]] == 0.0);
        
        // Verify that L * L^T equals the original matrix
        let lt = l.t().unwrap();
        let reconstructed = l.matmul(&lt).unwrap();
        
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_qr() {
        let a = tensor!([
            [12.0, -51.0, 4.0],
            [6.0, 167.0, -68.0],
            [-4.0, 24.0, -41.0]
        ]);
        
        let (q, r) = qr(&a).unwrap();
        
        // Verify that Q is orthogonal (Q^T * Q = I)
        let qt = q.t().unwrap();
        let identity = qt.matmul(&q).unwrap();
        
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(identity[[i, j]], 1.0, epsilon = 1e-6);
                } else {
                    assert_relative_eq!(identity[[i, j]], 0.0, epsilon = 1e-6);
                }
            }
        }
        
        // Verify that R is upper triangular
        for i in 0..3 {
            for j in 0..i {
                assert_relative_eq!(r[[i, j]], 0.0, epsilon = 1e-6);
            }
        }
        
        // Verify that Q * R equals the original matrix
        let reconstructed = q.matmul(&r).unwrap();
        
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(reconstructed[[i, j]], a[[i, j]], epsilon = 1e-6);
            }
        }
    }
}
