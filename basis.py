import numpy as np

def Gram_Schmidt(X):
    n, dim = X.shape

    for i in range(n):
        for j in range(i):
            X[i] = X[i] - np.dot(X[i],X[j])/np.dot(X[j],X[j]) * X[j]
        X[i] = X[i]/np.sqrt(np.dot(X[i],X[i]))

    return X


