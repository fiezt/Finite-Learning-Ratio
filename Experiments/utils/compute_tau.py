import numpy as np
import numpy.linalg as la
from scipy.linalg import block_diag as blkdiag


def duplication_matrix_op(n):
    """generate duplication matrix: see https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices"""
    
    dim_1 = int(n*(n+1)/2)
    dim_2 = n**2
    
    duplication_matrix = np.zeros((dim_1, dim_2))
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i < j: continue
            
            # generate unit vector of dimension n*(n+1)/2 with value 1 in position (j-1)*n + i - j*(j-1)/2
            unit_ij = np.zeros((dim_1, 1))
            ij_position = (j-1)*n + i - int(j*(j-1)/2)-1
            unit_ij[ij_position] = 1
            
            # n x n matrix with 1 in position (i, j) and (j, i) and zero elsewhere
            T = np.zeros((n, n))
            T[i-1, j-1] = 1 
            T[j-1, i-1] = 1
            T = T.reshape(1, -1)
            
            duplication_matrix += unit_ij@T
    
    duplication_matrix = duplication_matrix.T
    
    return duplication_matrix

def pinv(D):
    """left pseudo inverse"""
    
    return la.inv(D.T@D)@(D.T)

def oplus_op(A, B):
    """oplus operator"""
    
    m = len(A)
    n = len(B)
    
    return np.kron(A, np.eye(n)) + np.kron(np.eye(m),B)

def square_plus_op(A, duplication_matrix):
    """square plus operator"""
    
    duplication_matrix_pinv = pinv(duplication_matrix)
    
    return duplication_matrix_pinv@oplus_op(A, A)@duplication_matrix


def compute_tau_star(J, m, n):
    """compute tau star for a critical point.
    
    :param J: Jacobian matrix of the vector field evaluated at a critical point.
    :param m: Dimension of strategy for player 1.
    :param n: Dimension of strategy for player 2.
    
    return: tau_star
    """
    
    A11 = J[:n, :n] 
    A12 = J[:n, n:]
    A21 = J[n:, :n] 
    A22 = J[n:, n:]
    
    if np.min(np.linalg.eigvals(A22)) <= 0 or np.min(np.linalg.eigvals(A11 - A12@la.inv(A22)@A21)) <= 0:
        return 'Given critical point is not a DSE so it fails to satisfy the theorem'
    
    Dn = duplication_matrix_op(n)
    Dm = duplication_matrix_op(m)
    
    Q = -np.kron(A11, la.inv(A22))
  
    S = A11 -A12@la.inv(A22)@A21
    
    G1 = np.hstack(((np.kron(A12, la.inv(A22))@Dm), np.kron(np.eye(n), la.inv(A22)@A21)@Dn))
    
    G2 = blkdiag(la.inv(square_plus_op(A22, Dm)), -la.inv(square_plus_op(S, Dn)))
    
    G3 = np.vstack((pinv(Dm)@np.kron(A21, np.eye(m)), pinv(Dn)@np.kron(S, A12@la.inv(A22))))
    
    Q = Q + 2*G1@G2@G3

    tau_star = max(np.max(np.real(la.eigvals(Q))), 0)
    
    return tau_star