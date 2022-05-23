import os
import numpy as np
import re
import subprocess
import uuid

from math import log10
from .solver import Solver
from .utils import replace_inf

# ATSP: Find a HAMILTONIAN CIRCUIT (Tour)  whose global cost is minimum (Asymmetric Travelling Salesman Problem: ATSP)
# https://github.com/coin-or/metslib-examples/tree/master/atsp
# http://www.localsolver.com/documentation/exampletour/tsp.html
# http://or.dei.unibo.it/research_pages/tspsoft.html

MAX_INT32 = 2**31 - 1


class SolverConcorde(Solver):
    def __init__(
        self, seed=0, verbose=False, basedir="/tmp", basename=None, timeout=300
    ):

        self.solution = None
        self.overall_compatibility = -1
        self.seed = seed
        self.basedir = basedir
        self.basename = basename
        self.verbose = verbose
        self.timeout = timeout

    def solve(self, instance, seed=None):

        # overwrite object params
        seed = seed if seed is not None else self.seed

        assert seed is not None

        instance = np.array(instance)
        instance_atsp = instance.copy()

        # dummy node inserted in the end
        instance_atsp = np.pad(
            instance_atsp, ((0, 1), (0, 1)), mode="constant", constant_values=0
        )
        num_cities = instance_atsp.shape[0]

        solverATSP = ConcordeATSPSolver(
            seed, self.verbose, self.basedir, self.basename, self.timeout
        )
        try:
            solution = solverATSP.solve(instance_atsp)
        except IndexError:
            self.solution = None
            # return None
            return self

        if solution is None:
            self.solution = None
            return self
            # return None

        # remove repeated element and dummy node
        solution = solution[:-1]
        dummy_idx = solution.index(num_cities - 1)
        solution = solution[dummy_idx + 1 :] + solution[:dummy_idx]
        self.solution = solution
        self.overall_compatibility = float(instance[solution[:-1], solution[1:]].sum())

        return self

    def id(self):
        return "Concorde"


class ConcordeATSPSolver:
    """Solver for ATSP using Concorde."""

    @staticmethod
    def load_tsplib(filename):
        """Load a tsplib instance for testing."""

        lines = open(filename).readlines()
        regex_non_numeric = re.compile(r"[^\d]+")
        n = int(
            next(
                regex_non_numeric.sub("", line)
                for line in lines
                if line.startswith("DIMENSION")
            )
        )
        start = next(
            i for i, line in enumerate(lines) if line.startswith("EDGE_WEIGHT_SECTION")
        )
        end = next(i for i, line in enumerate(lines) if line.startswith("EOF"))
        matrix = np.array(
            [int(v) for v in " ".join(lines[start + 1 : end]).split()], dtype=np.int32
        ).reshape((n, -1))

        return matrix

    @staticmethod
    def dump_tsplib(matrix, filename):
        """Dump a tsplib instance.

        For detais on tsplib format, check: http://ftp.uni-bayreuth.de/math/statlib/R/CRAN/doc/packages/TSP.pdf
        """

        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]

        template = """NAME: {name}
TYPE: TSP
COMMENT: {name}
DIMENSION: {n}
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
{matrix_str}EOF"""

        name = os.path.splitext(os.path.basename(filename))[0]
        n = matrix.shape[0]

        # space delimited string
        matrix_str = " "
        for row in matrix:
            matrix_str += " ".join([str(val) for val in row])
            matrix_str += "\n"
        open(filename, "w").write(
            template.format(**{"name": name, "n": n, "matrix_str": matrix_str})
        )

    def __init__(self, seed, verbose=False, basedir="/tmp", basename=None, timeout=300):
        """Class constructor."""

        self.seed = seed
        self.solution = None
        self.cost = 0.0
        self.verbose = verbose
        self.basedir = basedir
        self.basename = basename
        self.timeout = timeout

    def _run_concorde(self, matrix):
        """Run Concorde solver for instance named with filename.

        Check https://github.com/mhahsler/TSP/blob/master/R/tsp_concorde.R for some tricks.
        """

        # dump matrix in int32 format
        basename = self.basename
        if basename is None:
            basename = "{}.tsp".format(str(uuid.uuid4()))

        tsp_filename = "{}/{}".format(self.basedir, basename)
        ConcordeATSPSolver.dump_tsplib(matrix.astype(np.int32), tsp_filename)

        # call Concorde solver
        curr_dir = os.path.abspath(".")
        # dir_ = os.path.dirname(tsp_filename)
        # os.chdir(dir_)
        os.chdir(self.basedir)
        sol_filename = "{}/{}.sol".format(self.basedir, os.path.splitext(basename)[0])
        # sol_filename = "{}/{}.sol".format(dir_, os.path.splitext(os.path.basename(tsp_filename))[0])

        # run Concode
        cmd = ["concorde", "-s", str(self.seed), "-o", sol_filename, tsp_filename]
        try:
            with open(os.devnull, "w") as devnull:
                try:
                    # print('---------------')
                    # print(output.decode('UTF-8'))
                    # output = subprocess.check_output(cmd, timeout=self.timeout)
                    output = subprocess.check_output(
                        cmd, stderr=devnull, timeout=self.timeout
                    )
                except subprocess.CalledProcessError:
                    if self.verbose:
                        print("ERROR: The process was not properly run.")
                    os.chdir(curr_dir)
                except subprocess.TimeoutExpired:
                    if self.verbose:
                        print("ERROR: Timeout.")
                    os.chdir(curr_dir)

                    return None
        except OSError as exc:
            if "No such file or directory" in str(exc):
                raise Exception("ERROR: Concorde solver not found.")
            raise exc
        os.chdir(curr_dir)
        tour = [int(v) for v in open(sol_filename).read().split()[1:]]
        cost = matrix.astype(np.int32)[tour[:-1], tour[1:]].sum()

        return tour

    def _atsp_to_tsp(self, C):
        """
        Reformulate an asymmetric TSP as a symmetric TSP:
        "Jonker and Volgenant 1983"
        This is possible by doubling the number of nodes. For each city a dummy
        node is added: (a, b, c) => (a, a', b, b', c, c')

        distance = "value"
        distance (for each pair of dummy nodes and pair of nodes is INF)
        distance (for each pair node and its dummy node is -INF)
        ------------------------------------------------------------------------
          |a    |b    |c    |a'   |b'   |c'   |
        a |-    |INF  |INF  |-INF |dBA  |dCA  |
        b |INF  |-    |INF  |dAB  |-INF |dCB  |
        c |INF  |INF  |-    |dAC  |dBC  |-INF |
        a'|-INF |dAB  |dAC  |-    |INF  |INF  |
        b'|dBA  |-INF |dBC  |INF  |-    |INF  |
        c'|dCA  |dCB  |-INF |INF  |INF  |-    |

        @return: new symmetric matrix

        [INF][C.T]
        [C  ][INF]
        """

        n = C.shape[0]
        n_tilde = 2 * n
        C_tilde = np.empty((n_tilde, n_tilde), dtype=np.float32)
        C_tilde[:, :] = np.inf
        C_tilde[n:, :n] = C
        C_tilde[:n, n:] = C.T
        np.fill_diagonal(C_tilde[n:, :n], -np.inf)
        np.fill_diagonal(C_tilde[:n, n:], -np.inf)
        np.fill_diagonal(C_tilde, np.inf)

        return C_tilde

    def solve(self, instance):
        """Solve ATSP instance."""

        num_cities_atsp = instance.shape[0]
        num_cities_tsp = 2 * num_cities_atsp
        instance_tsp = self._atsp_to_tsp(instance)
        replace_inf(instance_tsp)
        instance_tsp -= instance_tsp.min()

        # max. possible tsp tour
        max_tour = np.sort(instance_tsp.flatten())[-num_cities_tsp:].sum()
        p = int(log10(MAX_INT32 / max_tour))
        instance_tsp *= 10**p
        # print("p={}".format(p))

        # TSP solution
        solution_tsp = self._run_concorde(instance_tsp)
        if solution_tsp is None:
            self.solution = None
            self.cost = -1.0
        else:
            # convert to ATSP solution
            solution = solution_tsp[::2] + [solution_tsp[0]]

            # TSP - Infrastructure for the Traveling Salesperson Problem (Hahsler and Hornik)
            # "Note that the tour needs to be reversed if the dummy cities appear before and
            # not after the original cities in the solution of the TSP."
            if solution_tsp[1] != num_cities_atsp:
                solution = solution[::-1]
            self.cost = float(instance[solution[:-1], solution[1:]].sum())
            self.solution = solution

        return self.solution


# Testing
if __name__ == "__main__":
    """
    Best known solutions for asymmetric TSPs
    br17: 39
    ft53: 6905
    ft70: 38673
    ftv33: 1286
    ftv35: 1473
    ftv38: 1530
    ftv44: 1613
    ftv47: 1776
    ftv55: 1608
    ftv64: 1839
    ftv70: 1950
    ftv90: 1579
    ftv100: 1788
    ftv110: 1958
    ftv120: 2166
    ftv130: 2307
    ftv140: 2420
    ftv150: 2611
    ftv160: 2683
    ftv170: 2755
    kro124: 36230
    p43: 5620
    rbg323: 1326
    rbg358: 1163
    rbg403: 2465
    rbg443: 2720
    ry48p: 14422
    """

    import sys
    import time
    import matplotlib.pyplot as plt

    path = "/home/tpaixao/software/tsplib"

    print("TSPLIB instances")

    optimal_costs = {
        "br17": 39,
        "ft53": 6905,
        "ft70": 38673,
        "ftv33": 1286,
        "ftv35": 1473,
        "ftv38": 1530,
        "ftv44": 1613,
        "ftv47": 1776,
        "ftv55": 1608,
        "ftv64": 1839,
        "ftv70": 1950,
        "ftv90": 1579,
        "ftv100": 1788,
        "ftv110": 1958,
        "ftv120": 2166,
        "ftv130": 2307,
        "ftv140": 2420,
        "ftv150": 2611,
        "ftv160": 2683,
        "ftv170": 2755,
        "kro124p": 36230,
        "p43": 5620,
        "rbg323": 1326,
        "rbg358": 1163,
        "rbg403": 2465,
        "rbg443": 2720,
        "ry48p": 14422,
    }
    filenames = [
        filename for filename in os.listdir(path) if filename.endswith(".atsp")
    ]
    T = []
    N = []
    solver = ConcordeATSPSolver(precision=0)
    for filename in filenames[:3]:
        t0 = time.time()
        matrix = ConcordeATSPSolver.load_tsplib("{}/{}".format(path, filename))
        # print(matrix.dtype)
        solution = solver.solve(matrix)
        cost = solver.cost
        t = time.time() - t0
        N.append(matrix.shape[0])
        T.append(t)
        case = filename.split(".")[0]
        print(
            "{} - {}/{} [{:.10f}s]".format(
                filename, cost, optimal_costs[filename[:-5]], t
            )
        )
    # idx = np.argsort(N)
    # N = [N[i] for i in idx]
    # T = [T[i] for i in idx]
    # plt.plot(N, T)
    # plt.savefig('preliminar/tsplib-time.pdf', bbox_inches='tight')
