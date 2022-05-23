import re
import glob
import random
import numpy as np

# from collections import defaultdict
# from utils import flatten


def transform_compatibilities_cost(compatibilities):
    mask = ~np.isin(compatibilities, [np.inf, -np.inf])
    max_val = compatibilities[mask].max()
    return max_val - compatibilities


def get_instances(dataset, mode="test", split_train_val=False, val_perc=0.1, seed=0):
    assert mode in ["train", "test"]

    if mode == "train":
        assert dataset in ["cdip", "isri-ocr"]
        instances = glob.glob("datasets/train/{}_bin/*".format(dataset))
    else:
        assert dataset in ["S-MARQUES", "S-ISRI-OCR", "S-CDIP"]

        # reconstruction instances
        # pattern: <path>/D<digits>
        regex = re.compile("(.|/)*D(\d)+$")
        dirs = sorted(glob.glob("datasets/test/{}/mechanical/*".format(dataset)))
        instances = [dir_ for dir_ in dirs if regex.match(dir_)]

    if split_train_val:
        random.shuffle(instances)
        k = int(val_perc * len(instances))
        return {"train": instances[k:], "val": instances[:k]}

    return instances


# def get_results_id(dataset_test, vshift, input_size_h, model_id):
#     results_id = "{}_{}_{}_{}".format(dataset_test, vshift, input_size_h, model_id)
#     return results_id


# def blocks_from_pairs(pairs):
#     """Merge pairs into blocks (mapping id->block).
#     Example:
#     pairs = [(1, 4), (4, 5), (5, 6), (6, 7), (7, 2), (2, 3)]
#     blocks = {0: [1], 1: []}
#     """

#     blocks = dict()
#     next_id = 0
#     for pair in pairs:
#         found = False
#         # iterate over the blocks
#         for block in blocks.values():
#             # fit in the end
#             if block[-1] == pair[0]:
#                 block.append(pair[1])
#                 found = True
#                 break
#             # fits in the beginning
#             elif pair[1] == block[0]:
#                 block.insert(0, pair[0])
#                 found = True
#                 break
#         if not found:
#             blocks[next_id] = list(pair)
#             next_id += 1

#     return blocks


# def fill_blocks(blocks, n):
#     entries_to_merge = sorted(flatten(blocks.values()))
#     next_id = len(blocks)
#     blocks = blocks.copy()
#     for i in range(n):
#         if i not in entries_to_merge:
#             blocks[next_id] = [i]
#             next_id += 1

#     return blocks


# def split_solution_into_blocks(solution):

#     blocks = defaultdict(list)
#     next_id = 0
#     # for i in range(len(solution) - 1):
#     for s in solution:
#         if len(blocks[next_id]) == 0:
#             blocks[next_id].append(s)
#         elif s == blocks[next_id][-1] + 1:
#             blocks[next_id].append(s)
#         else:
#             next_id += 1
#             blocks[next_id].append(s)
#     return blocks


# def reindex_blocks(blocks):

#     sorted_blocks_as_list = sorted(blocks.items(), key=lambda item: item[1][0])
#     reindexed_blocks = {i: item[1] for i, item in enumerate(sorted_blocks_as_list)}

#     return reindexed_blocks
