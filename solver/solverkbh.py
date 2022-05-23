import numpy as np

from .solver import Solver
from .utils import replace_inf


class SolverKBH(Solver):
    def __init__(self):
        """
        R. Ranca and I. Murray, "A Composable Strategy for Shredded Document  Reconstruction,"
        in International  Conference  on  Computer  Anal-ysis of Images and Patterns, 2013, pp. 324–331

        Implementation is based on the Python implementations of the book "Algorithms - Dasgupta, Papadimitriou and Vazurani"
        https://github.com/israelst/Algorithms-Book--Python
        """

        self.solution = None
        self.overall_compatibility = -1.0
        self.parent = dict()
        # self.rank = dict()

    def solve(self, instance):

        instance = np.array(instance)
        np.fill_diagonal(instance, np.inf)
        replace_inf(instance)
        instance -= instance.min()

        # defining graph
        # V: vertices
        # A: arcs
        n = instance.shape[1]
        V = list(range(n))
        A = [(u, v, instance[u, v]) for u in range(n) for v in range(n) if u != v]

        # step 1
        for v in V:
            self._make_set(v)

        # step 2
        minimum_spanning_tree = set()
        forbidden_src = set()
        forbidden_dst = set()
        A.sort(key=lambda tup: tup[2])  # sort in place by weight
        for u, v, _ in A:
            if self._find(u) != self._find(v):  # ensure not to form a cycle
                if (u not in forbidden_src) and (v not in forbidden_dst):  # path restriction
                    self._union(u, v, V)
                    minimum_spanning_tree.add((u, v))
                    forbidden_src.add(u)
                    forbidden_dst.add(v)

        # step 3
        solution = [self.parent[V[0]]]
        for i in range(n - 1):
            curr = solution[i]
            # find the neighbor of the current vertex in solution
            for u, v in minimum_spanning_tree:
                if u == curr:
                    solution.append(v)
                    break

        self.solution = solution
        self.overall_compatibility = float(instance[solution[:-1], solution[1:]].sum())
        return self

    def _make_set(self, v):

        self.parent[v] = v
        # self.rank[v] = 0

    def _find(self, v):
        """Find the root (first node) of the path which includes v."""

        if self.parent[v] != v:
            self.parent[v] = self._find(self.parent[v])
        return self.parent[v]

    def _union(self, u, v, V):

        root_u = self._find(u)
        self.parent[v] = root_u
        for v in V:
            self._find(v)
