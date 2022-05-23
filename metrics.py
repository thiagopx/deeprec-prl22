import numpy as np
from utils import blocks_from_documents_collection, neighbors_from_blocks


def Qc(compatibilities, normalized=True, infty=1e7):
    """Equation 1 of the SIBGRAPI 2018 paper."""

    compatibilities = np.array(compatibilities)
    assert compatibilities.ndim in (2, 3)

    num_strips = compatibilities.shape[0]

    max_val = compatibilities[compatibilities != np.inf].max()
    compatibilities[compatibilities == np.inf] = infty
    compatibilities[compatibilities == -np.inf] = -infty

    # calculated using the min function
    np.fill_diagonal(compatibilities, 1e7)
    F_ij = 0
    for i in range(num_strips - 1):
        # row-wise verification
        row_min = compatibilities[i].min()
        Rv = (compatibilities[i, i + 1] == row_min) and (np.sum(compatibilities[i] == row_min) == 1)
        # column-wise verification
        col_min = compatibilities[:, i + 1].min()
        Cv = (compatibilities[i, i + 1] == col_min) and (np.sum(compatibilities[:, i + 1] == col_min) == 1)

        # Equation 2
        F_ij += int(Rv and Cv)

    if normalized:
        return F_ij / (num_strips - 1)
    return F_ij


def accuracy(solution, sizes=None):
    """Accuracy by neighbor comparison."""

    assert len(solution) > 0

    # assume an unique instance with N = len(solution) strips
    num_strips = len(solution)
    if sizes is None:
        sizes = [num_strips]

    blocks = blocks_from_documents_collection(sizes)
    neighbors = neighbors_from_blocks(blocks)
    num_hits = 0
    for i in range(num_strips - 1):
        if solution[i + 1] in neighbors[solution[i]]:
            num_hits += 1

    first_in_each_block = [block[0] for block in blocks.values()]
    if (num_hits == (num_strips - 1)) and (solution[0] not in first_in_each_block):
        num_hits -= 1

    return num_hits / (num_strips - 1)
