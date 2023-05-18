import numpy as np

def Gauss_elimination(A, b=None, return_b=False, pivot=False, back_substitution=False):
    n, m = A.shape
    if not return_b:
        b = np.array([0]*m)
    
    assert n == m, "Matrix is not square."
    assert m == len(b), "Dimension of b vector is {}, but should be {}".format(len(b), m)
    
    def swap_row(i,j):
        temp = np.copy(A[i])
        A[i] = A[j]
        A[j] = temp
        temp = np.copy(b[i])
        b[i] = b[j]
        b[j] = temp
    
    i = 0
    j = 0
    while i < n and j < m:
        # Find the ith pivot
        i_max = np.argmax(abs(A[i:, j])) + i
        if A[i_max, j] == 0 :
            j += 1
        else:
            if pivot:
                swap_row(i, i_max)
            
            # Gauss elimination
            for k in range(i+1, n):
                a = A[k,j]/A[i,j]
                A[k,j] = 0
                for l in range(j+1, m):
                    A[k,l] = A[k,l] - A[i,l]*a
                b[k] = b[k] - b[i]*a
            i += 1
            j += 1

    A_org = np.copy(A)
    if back_substitution:
        for j in reversed(range(1, m)):
            for i in range(j-1):
                if A[j,j] == 0: 
                    print("Matrix is not diagonalizable.")
                    return A_org
                a = A[i,j]/A[j,j]
                A[i,j] = 0
                b[i] = b[i] - b[j]*a

    if return_b:
        return A, b
    return A


def GF2_Gauss_elimination(A):
    n, m = A.shape
    
    def swap_row(i,j):
        temp = np.copy(A[i])
        A[i] = A[j]
        A[j] = temp

    i = 0
    j = 0
    while i < n and j < m:
        # Find the ith pivot
        i_max = np.argmax(A[i:, j]%2) + i
        if A[i_max, j]%2 == 0 :
            j += 1
        else:
            swap_row(i, i_max)
            # Gauss elimination
            for k in range(n):
                if A[k,j]%2 == 0 or k == i:
                        pass
                else:
                    for l in range(j, m):
                        A[k,l] = (A[k,l] + A[i,l])%2
            i += 1
            j += 1    

    return A
