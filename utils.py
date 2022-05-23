import os
import random
import tempfile
import cv2
import numpy as np
import argparse
import shutil

# from skimage.util.shape import view_as_blocks
# from skimage.util import montage
from collections import defaultdict
from skimage.filters import threshold_sauvola


class defaultdict_factory(defaultdict):
    def __init__(self, factory_func):
        super().__init__(None)
        self.factory_func = factory_func

    def __missing__(self, key):
        ret = self.factory_func(key)
        self[key] = ret

        return self[key]


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def clear_dir(path):
    for root, _, fnames in os.walk(path):
        for fname in fnames:
            os.remove("{}/{}".format(root, fname))


def remove_dir(path):
    shutil.rmtree(path, ignore_errors=True)


def move_dir(src, dst=None):
    if dst is None:
        dst = tempfile.NamedTemporaryFile().name
    try:
        _ = shutil.move(src, dst)
    except FileNotFoundError as exc:
        pass


def load_rgb(fname):

    rgb = cv2.imread(fname)[..., ::-1]
    return rgb


def save_rgb(image, fname):
    assert image.ndim == 3

    bgr = image[..., ::-1]
    cv2.imwrite(fname, bgr)  # it saves rgb, so we have to invert before save


def load_grayscale(fname, num_channels=1):
    assert num_channels in [1, 3]

    gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if num_channels == 3:
        gray = np.stack(3 * [gray]).transpose(1, 2, 0)
    return gray


def load_binary(fname, thresh_func=threshold_sauvola, num_channels=1):
    assert num_channels in [1, 3]

    gray = load_grayscale(fname)
    binary = grayscale_to_binary(gray, thresh_func)
    thresh = thresh_func(gray)
    binary = (255 * (binary > thresh)).astype(np.uint8)
    if num_channels == 3:
        binary = np.stack(3 * [binary], axis=0).transpose(1, 2, 0)
    return binary


# def get_label(fname):
#     label = 1 if "positives" in fname else 0
#     return label


def crop_central_area(image, w_crop):
    _, w = image.shape[:2]
    x_start = (w - w_crop) // 2
    return image[:, x_start : x_start + w_crop].copy()


def rgb_to_grayscale(image):
    assert image.ndim == 3

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray


def grayscale_to_rgb(image):
    assert image.ndim == 2

    rgb = np.stack([image, image, image]).transpose((1, 2, 0))
    return rgb


def grayscale_to_binary(image, thresh_func=threshold_sauvola, num_channels=1):
    assert num_channels in [1, 3]
    assert image.ndim == 2

    thresh = thresh_func(image)
    binary = (255 * (image > thresh)).astype(np.uint8)
    if num_channels == 3:
        binary = np.stack(3 * [binary], axis=0).transpose(1, 2, 0)
    return binary


def rgb_to_binary(image, thresh_func=threshold_sauvola, num_channels=1):
    assert num_channels in [1, 3]
    assert image.ndim == 3

    gray = rgb_to_grayscale(image)
    binary = grayscale_to_binary(gray, thresh_func, num_channels)
    return binary


def compute_threshold(image, thresh_func=threshold_sauvola):
    assert image.ndim in [2, 3]

    if image.ndim == 3:
        image = rgb_to_grayscale(image)
    return thresh_func(image)


def apply_threshold(image, thresh, num_channels=1):
    assert image.ndim in [2, 3]
    if image.ndim == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image
    binary = (255 * (gray > thresh)).astype(np.uint8)
    if num_channels == 3:
        binary = np.stack(3 * [binary], axis=0).transpose(1, 2, 0)
    return binary


def is_integer(value):
    return value == int(value)


def doc_id_from_path(fname):
    doc_without_ext = os.path.splitext(fname)[0]
    doc_basename = os.path.basename(doc_without_ext)
    return doc_basename


def save_list_as_txt(list_, fname, shuffle=False):
    list_cpy = list_.copy()
    if shuffle:
        random.shuffle(list_cpy)
    txt = "\n".join(list_cpy)
    open(fname, "w").write(txt)


def decode_txt(fname, num_cols=2):
    assert num_cols > 1
    txt = []
    lines = open(fname, "r").readlines()
    for line in lines:
        line_split = line.strip().split()[:num_cols]
        txt.append(line_split)
    cols = [list(col) for col in zip(*txt)]
    return cols


def sample_from_lists(*lists, k=10):
    lists_merged = list(zip(*lists))
    lists_sampled = random.sample(lists_merged, k=k)
    return list(zip(*lists_sampled))


def merge_by_keys(dict_, f_merge):
    """Merge entries with similar keys, where similarity is defined by f_merge function."""
    dict_merged = defaultdict(list)
    for k, v in dict_.items():
        k_ = f_merge(k)
        dict_merged[k_] += v
    return dict_merged


def function_on_lists(lists, func):
    list_all = []
    for list_ in lists:
        list_all += list_
    return func(list_all)


def flatten(lists_or_tuples):
    list_all = []
    for x in lists_or_tuples:
        if not (isinstance(x, list) or isinstance(x, tuple)):
            list_all.append(x)
        # elif isflat(x):
        #     list_all.extend(x)
        else:
            list_all.extend(flatten(x))
    return list_all


# def shuffle(*params):
#     # ensure at least one list/array
#     assert len(params) > 0

#     # ensure the type is a list or array
#     for list_or_arr in params:
#         assert type(list_or_arr) in [list, np.ndarray]

#     # ensure the list/arrays has the same sime
#     sizes = []
#     for list_or_arr in params:
#         if type(list_or_arr) is list:
#             sizes.append(len(list_or_arr))
#         else:
#             sizes.append(list_or_arr.shape[0])
#     assert len(set(sizes)) == 1

#     # shuffle
#     idx = [x for x in range(sizes[0])]
#     random.shuffle(idx)
#     lists_shuffled = []
#     for list_or_arr in params:
#         if type(list_or_arr) is np.ndarray:
#             lists_shuffled.append(list_or_arr[idx])
#         else:
#             lists_shuffled.append([list_or_arr[i] for i in idx])
#     if len(lists_shuffled) == 1:
#         return lists_shuffled[0]
#     return lists_shuffled


# text
def stopwords():
    path_abs = os.path.dirname(os.path.realpath(__file__))
    words = open("{}/stopwords.txt".format(path_abs)).readlines()
    words = [word.strip() for word in words]
    return words


# import string

# NGRAMS = []

# # bigrams
# for c1 in string.ascii_uppercase + string.ascii_lowercase:
#     for c2 in string.ascii_lowercase:
#         NGRAMS.append(c1 + c2)

# # trigrams
# for c1 in string.ascii_uppercase + string.ascii_lowercase:
#     for c2 in string.ascii_lowercase:
#         for c3 in string.ascii_lowercase:
#             NGR
# AMS.append(c1 + c2 + c3)


def blocks_from_documents_collection(sizes):
    """Returns blocks of strips where each block belongs to the same document."""

    assert len(sizes) > 0

    blocks = dict()
    next_id = 0
    cum = 0
    for size in sizes:
        block = []
        for idx in range(size):
            block.append(cum + idx)
        blocks[next_id] = block
        cum += size
        next_id += 1

    return blocks


def pairs_from_solution(solution):
    pairs = []
    for i in range(len(solution) - 1):
        pair = (solution[i], solution[i + 1])
        pairs.append(pair)
    return pairs


def neighbors_from_blocks(blocks, top_level_neighbors=None):
    """Returns a mapping from blocks of strips and their neighbors. The neighbor definition are two-fold:
    1) Adjacent strips in the same block/document.
    2) The last strip of a black adjacent to the first of any other block/document.
    """

    neighbors = defaultdict(list)
    if not top_level_neighbors:
        for i in range(len(blocks)):
            # 1) same block
            for j in range(len(blocks[i]) - 1):
                neighbors[blocks[i][j]].append(blocks[i][j + 1])
            # 2) between blocks
            for j in range(len(blocks)):
                if i != j:
                    neighbors[blocks[i][-1]].append(blocks[j][0])
    else:
        for i in range(len(blocks)):
            for j in range(len(blocks)):
                if i == j:
                    continue
                # i as the left side of j
                # if blocks[j][0] in top_level_neighbors[blocks[i][-1]]:
                if blocks[j][0] in top_level_neighbors[blocks[i][-1]]:
                    neighbors[i].append(j)

    return neighbors


def blocks_from_pairs(pairs, neighbors):
    """Merge pairs into blocks (mapping id->block).
    Example:
    pairs = [(7, 0), (5, 6), (4, 5), (1, 2), (2, 4)] # 3 is missing
    neighbors = {0: [1], 1: [2], 2: [3, 6], 3: [4], 4: [5], 5: [0, 6], 6: [7], 7: [0, 3]}
    blocks = {0: [4, 5, 6], 1: [1, 2], 2: [3], 3: [7, 0]}
    """

    blocks = dict()
    adjacent_pairs = []
    for pair in pairs:
        if pair[1] in neighbors[pair[0]]:
            adjacent_pairs.append(pair)
    next_id = 0

    # group the adjacent pairs
    for pair in adjacent_pairs:
        found = False
        # iterate over the blocks
        for block in blocks.values():
            # fit in the end
            if block[-1] == pair[0]:
                block.append(pair[1])
                found = True
                break
            # fits in the beginning
            elif pair[1] == block[0]:
                block.insert(0, pair[0])
                found = True
                break
        if not found:
            blocks[next_id] = list(pair)
            next_id += 1

    # fill with the individual blocks
    blocks_values = flatten(blocks.values())
    next_id = len(blocks)
    blocks = blocks.copy()
    for i in range(len(neighbors)):
        if i not in blocks_values:
            blocks[next_id] = [i]
            next_id += 1

    # reindex blocks by sorting according the first element
    sorted_blocks_as_list = sorted(blocks.items(), key=lambda item: item[1][0])
    blocks = {i: item[1] for i, item in enumerate(sorted_blocks_as_list)}

    return blocks


def break_solution_into_blocks(solution, forbidden_pairs):
    all_pairs = [(solution[i], solution[i + 1]) for i in range(len(solution) - 1)]
    blocks = dict()
    next_id = 0
    blocks[next_id] = [all_pairs[0][0]]
    for pair in all_pairs:
        if pair not in forbidden_pairs:
            blocks[next_id].append(pair[1])
        else:
            next_id += 1
            blocks[next_id] = [pair[1]]
    return blocks


def merge_compatibilities_rows_cols(compatibilities, blocks):
    """Merge blocks of entries of a compatibility matrix"""

    num_blocks = len(blocks)
    # max_label = max(blocks)
    # print()
    # # print(max_label, ": ", blocks[max_label])
    # for label, block in blocks.items():
    #     print(label, ": ", block)

    # print("----------------")
    # sys.exit()
    compatibilities_ = np.zeros((num_blocks, num_blocks), dtype=compatibilities.dtype)
    for i_ in range(num_blocks):
        for j_ in range(num_blocks):
            if i_ != j_:
                i = blocks[i_][-1]
                j = blocks[j_][0]
                # print(i_, i, "---", j_, j, num_blocks)
                compatibilities_[i_, j_] = compatibilities[i, j]

    return compatibilities_


def transform_current_to_original_solution(solution, blocks_sequence):
    """Transform the current solution to the original (matrix space) given a blocks_sequence."""

    curr_solution = solution
    for block in blocks_sequence[::-1]:
        curr_solution = flatten([block[x] for x in curr_solution])

    return curr_solution


def unshuffle_solution(solution, init_permutation):
    """Write the solution in the ground-truth enumeration."""

    return [init_permutation[s] for s in solution]


def unshuffle_compatibilities(compatibilities, init_permutation):
    """Write the matrix in the ground-truth enumeration."""

    inv_permutation = np.argsort(init_permutation)
    return compatibilities[inv_permutation][:, inv_permutation]
