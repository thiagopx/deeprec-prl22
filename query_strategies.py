import random
import numpy as np
from scipy.stats import entropy


def perm_to_pairs(perm):
    return [(perm[i], perm[i + 1]) for i in range(len(perm) - 1)]


def uncertainty(matrix, norm_method="softmax", axis=0, soft_range=1000):
    """uncertainty."""
    assert norm_method in ["standard", "softmax"]

    matrix = np.array(matrix)
    if norm_method == "softmax":
        mask = ~np.isin(matrix, [np.inf, -np.inf])
        max_val = matrix[mask].max()
        matrix = soft_range * (matrix / max_val)
        num = np.exp(-matrix)
        den = np.sum(num, axis=axis, keepdims=True)
        probs = np.divide(num, den, out=np.zeros_like(num), where=num != np.inf)
        probs[num == np.inf] = 1.0
    else:
        np.fill_diagonal(matrix, 0)
        probs = matrix / matrix.sum(axis=axis, keepdims=True)
    return entropy(probs, axis=axis)


# strategies
def left_to_right_qs(pairs, n=None):
    """left-to-right."""
    if n is None:
        n = len(pairs)
    return pairs[:n]


# random
def random_qs(pairs, n=None):
    """random (arbitrary choice)."""

    random.shuffle(pairs)
    if n is None:
        n = len(pairs)
    return pairs[:n]


def opt_l_qs(pairs, matrix, n=None):
    """best in the row based on the minimum/maximum value in the row."""
    pairs_opt = []
    for pair in pairs:
        if matrix[pair[0], pair[1]] != matrix[pair[0]].min():
            pairs_opt.insert(0, pair)
        else:
            pairs_opt.append(pair)
    if n is None:
        n = len(pairs)
    return pairs_opt[:n]


def opt_rl_qs(pairs, matrix, n=None):
    """best in the row/col based on the minimum/maximum value in the row."""
    pairs_opt = []
    for pair in pairs:
        if (matrix[pair[0], pair[1]] != matrix[pair[0]].min()) or (
            matrix[pair[0], pair[1]] != matrix[:, pair[1]].min()
        ):
            pairs_opt.insert(0, pair)
        else:
            pairs_opt.append(pair)
    if n is None:
        n = len(pairs)
    return pairs_opt[:n]


def unc_l_qs(pairs, matrix, norm_method="standard", n=None, soft_range=1000):
    """entropy-based uncertainty calculated on the rows."""
    # sort by uncertainty
    unc = uncertainty(matrix, norm_method=norm_method, axis=1, soft_range=soft_range)
    pairs_unc = [(pair, unc[pair[0]]) for pair in pairs]
    pairs_unc = sorted(pairs_unc, key=lambda x: x[1], reverse=True)
    pairs_unc = [pair for pair, _ in pairs_unc]
    if n is None:
        n = len(pairs)
    return pairs_unc[:n]


def unc_rl_qs(pairs, matrix, norm_method="standard", n=None, soft_range=1000):
    """entropy-based uncertainty calculated on the rows and cols."""
    # sort by uncertainty
    unc_row = uncertainty(
        matrix, norm_method=norm_method, axis=1, soft_range=soft_range
    )
    unc_col = uncertainty(
        matrix, norm_method=norm_method, axis=0, soft_range=soft_range
    )
    pairs_unc = [(pair, unc_row[pair[0]] + unc_col[pair[1]]) for pair in pairs]
    pairs_unc = sorted(pairs_unc, key=lambda x: x[1], reverse=True)
    pairs_unc = [pair for pair, _ in pairs_unc]
    if n is None:
        n = len(pairs)
    return pairs_unc[:n]


# joined opt- entropy-based uncertainty calculated on the rows
def opt_unc_l_qs(pairs, matrix, norm_method="standard", n=None, soft_range=1000):
    """joined opt- entropy-based uncertainty calculated on the rows."""

    # sort by uncertainty (all pairs)
    pairs_unc = unc_l_qs(pairs, matrix, norm_method, None, soft_range)
    # select by the optimality criteria
    pairs_opt = opt_l_qs(pairs_unc, matrix, n)
    return pairs_opt


def opt_unc_rl_qs(pairs, matrix, norm_method="standard", n=None, soft_range=1000):
    """joined opt- entropy-based uncertainty calculated on the rows."""

    # sort by uncertainty (all pairs)
    pairs_unc = unc_rl_qs(pairs, matrix, norm_method, None, soft_range)
    # select by the optimality criteria
    pairs_opt = opt_rl_qs(pairs_unc, matrix, n)
    return pairs_opt
