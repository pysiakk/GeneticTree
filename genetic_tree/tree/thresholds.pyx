import numpy as np
cimport numpy as np
from scipy.sparse import issparse

from numpy import float32 as DTYPE
ctypedef np.npy_float32 DTYPE_t

cpdef DTYPE_t[:, :] prepare_thresholds_array(int n_thresholds, object X):
    if issparse(X):
        return prepare_thresholds_array_sparse(n_thresholds, X)
    else:
        return prepare_thresholds_array_dense(n_thresholds, X)


cpdef DTYPE_t[:, :] prepare_thresholds_array_dense(int n_thresholds, object X):
    cdef int n_features = X.shape[1]

    cdef DTYPE_t[:, :] thresholds = np.zeros([n_thresholds, n_features], dtype=DTYPE)
    cdef DTYPE_t[:, :] X_ndarray = X
    cdef DTYPE_t[:] X_column

    cdef int i
    cdef int j
    cdef int index

    for i in range(n_features):
        X_column = X_ndarray[:, i]
        X_column = np.sort(X_column)
        for j in range(n_thresholds):
            index = int(X_column.shape[0] / (n_thresholds+1) * (j+1))
            thresholds[j, i] = X_column[index]
    return thresholds


cpdef DTYPE_t[:, :] prepare_thresholds_array_sparse(int n_thresholds, object X):
    cdef int n_features = X.shape[1]

    cdef DTYPE_t[:, :] thresholds = np.zeros([n_thresholds, n_features], dtype=DTYPE)
    cdef DTYPE_t[:] X_column

    cdef int i
    cdef int j
    cdef int index

    for i in range(n_features):
        X_column = X[:, i].toarray()[:, 0]
        X_column = np.sort(X_column)
        for j in range(n_thresholds):
            index = int(X_column.shape[0] / (n_thresholds+1) * (j+1))
            thresholds[j, i] = X_column[index]
    return thresholds
