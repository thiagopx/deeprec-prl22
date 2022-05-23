import numpy as np
from numba import jit


@jit(nopython=True)
def first_zero(arr):
    for i in range(arr.size):
        if arr[i] == 0:
            return i
    return -1


@jit(nopython=True)
def last_zero(arr):
    for i in range(arr.size - 1, -1, -1):
        if arr[i] == 0:
            return i
    return -1


@jit(nopython=True)
def first_nonzero(arr):
    for i in range(arr.size):
        if arr[i] != 0:
            return i
    return -1


@jit(nopython=True)
def last_nonzero(arr):
    for i in range(arr.size - 1, -1, -1):
        if arr[i] != 0:
            return i
    return -1


@jit(nopython=True)
def remove_gaps(arr):
    last = arr[0]
    arr[0] = 0
    for i in range(1, arr.size):
        if arr[i] == last:
            arr[i] = arr[i - 1]
        elif arr[i] > last:
            last = arr[i]
            arr[i] = arr[i - 1] + 1


def transitions(arr, edges_val=0):
    arr_ext = np.pad(arr.astype(np.int32), (1, 1), mode="constant", constant_values=edges_val)
    trans = np.diff(arr_ext)
    starts = np.where(trans == 1)[0]
    ends = np.where(trans == -1)[0] - 1
    return starts, ends


def shuffle(*arrays):
    """Shuffle one or more arrays in unison."""

    rng_state = np.random.get_state()
    for array in arrays[:-1]:
        np.random.shuffle(array)
        np.random.set_state(rng_state)
    np.random.shuffle(arrays[-1])


def choice(array, num_samples, replace=False):
    idxs = np.arange(array.shape[0])
    idxs_sel = np.random.choice(idxs, num_samples, replace)
    return array[idxs_sel]
