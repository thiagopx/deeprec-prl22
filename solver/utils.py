import numpy as np


def replace_inf(matrix):
    # https://rdrr.io/cran/TSP/man/TSPLIB.html
    # inf = val +/- 2*range

    # exclude diagonal/infs
    vals = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    vals = vals[(vals != np.inf) & (vals != -np.inf)]

    # replace
    min_val, max_val = vals.min(), vals.max()
    pinf = max_val + 2 * (max_val - min_val)
    ninf = min_val - 2 * (max_val - min_val)
    matrix[matrix == np.inf] = pinf
    matrix[matrix == -np.inf] = ninf
