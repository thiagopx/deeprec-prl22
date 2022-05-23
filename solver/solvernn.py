import numpy as np

from .solver import Solver
from .utils import replace_inf


class SolverNN(Solver):
    def __init__(self):
        self.solution = None
        self.overall_compatibility = -1

    def solve(self, instance):

        instance = np.array(instance)
        np.fill_diagonal(instance, np.inf)
        replace_inf(instance)
        instance -= instance.min()

        n = instance.shape[0]
        # backup_instance = instance.copy()
        A = [(u, v, instance[u, v]) for u in range(n) for v in range(n) if u != v]

        # start with the vertices of the minimum-weight arc
        a_start = A[0]
        for a in A[1:]:
            if a[2] < a_start[2]:
                a_start = a
        u, v, cost = a_start
        path = [u, v]

        # forbid u and v to be visited
        instance[:, u] = instance.max()
        instance[:, v] = instance.max()

        # complete solution
        for _ in range(n - 2):
            # current node
            u = path[-1]
            # best neighbor
            v = int(instance[u, :].argmin())
            try:
                cost += instance[u, v]
            except Warning:
                print(instance[u, v], type(instance[u, v]))
                import sys

                sys.exit()
            path.append(v)
            instance[:, v] = instance.max()

        self.overall_compatibility = cost
        self.solution = path
        return self
