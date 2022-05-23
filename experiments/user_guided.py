import os
import argparse
from multiprocessing import Pool
import time
import json
import random
import numpy as np
import metrics
from .utils import transform_compatibilities_cost
from solver import SolverConcorde, SolverNN, SolverKBH
from config import WORKLOAD_PERC
from utils import (
    blocks_from_documents_collection,
    neighbors_from_blocks,
)
from query_strategies import (
    opt_l_qs,
    opt_rl_qs,
    unc_l_qs,
    unc_rl_qs,
    perm_to_pairs,
    opt_unc_l_qs,
    opt_unc_rl_qs,
)


def print_report(cnt, total, record):
    print(
        "[{}/{}] {} {} wload={:.2f} iter={}/{} num_pred_mistakes={}/{} acc_ori={:.2f} acc_new={:.2f}".format(
            cnt,
            total,
            record["dataset_test"],
            record["query_st"],
            record["workload"],
            record["curr_iter"],
            record["num_iter"],
            record["num_pred_mistakes"],
            record["num_total_mistakes"],
            100 * record["accuracy_original"],
            100 * record["accuracy"],
        )
    )


def process_instance(instance):

    args = instance["args"]
    assert args.approach in ["cl", "dml"]
    assert args.solver in ["concorde", "kbh", "nn"]

    # seed experiment
    seed = instance["seed"]
    random.seed(seed)

    # configure solver
    solver_nn = SolverNN()
    solver_kbh = SolverKBH()
    solver = SolverConcorde(
        seed=seed,
        verbose=True,
        basedir=args.solver_basedir,
        # basename="instance.tsp",
        timeout=300,
    )
    if args.solver == "nn":
        solver = solver_nn
    elif args.solver == "kbh":
        solver = solver_kbh

    solution = instance["solution"]
    num_strips = len(solution)

    # compatibility/cost matrix
    compatibilities = np.load(instance["compatibilities_path"])
    # transform into costs
    if args.approach == "cl":
        compatibilities = transform_compatibilities_cost(compatibilities)

    # adjust diagonal
    forbid_val = np.inf
    lock_val = -np.inf
    np.fill_diagonal(compatibilities, forbid_val)

    sizes = instance["sizes"]
    neighbors = neighbors_from_blocks(blocks_from_documents_collection(sizes))
    dataset_test = instance["dataset_test"]

    map_fquery = {
        "opt-l": lambda pairs, comp, n: opt_l_qs(pairs, comp, n=n),
        "opt-rl": lambda pairs, comp, n: opt_rl_qs(pairs, comp, n=n),
        "unc-l": lambda pairs, comp, n: unc_l_qs(
            pairs, comp, "softmax", n, args.soft_range
        ),
        "unc-rl": lambda pairs, comp, n: unc_rl_qs(
            pairs, comp, "softmax", n, args.soft_range
        ),
        "opt-unc-l": lambda pairs, comp, n: opt_unc_l_qs(
            pairs, comp, "softmax", n, args.soft_range
        ),
        "opt-unc-rl": lambda pairs, comp, n: opt_unc_rl_qs(
            pairs, comp, "softmax", n, args.soft_range
        ),
    }

    # for each query strategy, workload, num_iter, ...
    records = []
    for query_st in args.query_st:
        fquery = map_fquery[query_st]
        for workload in WORKLOAD_PERC:
            for num_iter in range(1, args.max_num_iter + 1):
                num_pairs_to_be_analyzed = int(workload * (num_strips - 1) / num_iter)
                curr_solution = solution
                curr_compatibilities = compatibilities.copy()
                backup_compatibilities = np.zeros_like(compatibilities)
                for curr_iter in range(1, num_iter + 1):
                    t0 = time.time()

                    # pairs to be analyzed
                    pairs_solution = perm_to_pairs(curr_solution)
                    pairs_to_be_analyzed = fquery(
                        pairs_solution, curr_compatibilities, num_pairs_to_be_analyzed
                    )

                    # real number of mistakes
                    num_total_mistakes = 0
                    for i in range(len(curr_solution) - 1):
                        pair = (curr_solution[i], curr_solution[i + 1])
                        if not (pair[1] in neighbors[pair[0]]):
                            num_total_mistakes += 1

                    # forbid/lock op.
                    num_pred_mistakes = 0
                    backup_compatibilities[:] = curr_compatibilities
                    for pair in pairs_to_be_analyzed:
                        if not (pair[1] in neighbors[pair[0]]):
                            curr_compatibilities[pair[0], pair[1]] = forbid_val
                            num_pred_mistakes += 1
                        else:
                            curr_compatibilities[pair[0]] = forbid_val
                            curr_compatibilities[:, pair[1]] = forbid_val
                            curr_compatibilities[pair[0], pair[1]] = lock_val

                    # solve the problem problem
                    backup_solution = curr_solution
                    curr_solution = solver(instance=curr_compatibilities).solution

                    # update info if the solver was able to ouput a valid solution
                    # if not valid, the other iterations will keep the same result
                    if not curr_solution:
                        # restore to the previous state
                        curr_compatibilities[:] = backup_compatibilities
                        curr_solution = backup_solution

                    accuracy = metrics.accuracy(curr_solution, sizes=sizes)
                    record = {
                        "dataset_test": dataset_test,
                        "num_pred_mistakes": num_pred_mistakes,
                        "num_total_mistakes": num_total_mistakes,
                        "num_pairs_to_be_analyzed": num_pairs_to_be_analyzed,
                        "workload": workload,
                        "num_iter": num_iter,
                        "curr_iter": curr_iter,
                        "accuracy_original": instance["accuracy"],
                        "accuracy": accuracy,
                        "solution": curr_solution,
                        "query_st": query_st,
                        "solver": args.solver,
                        "approach": args.approach,
                        "soft_range": args.soft_range,
                        "seed": seed,
                        "elapsed_time": time.time() - t0,
                    }
                    records.append(record)

    return records


if __name__ == "__main__":

    # start cron
    start = time.time()

    parser = argparse.ArgumentParser(description="User-guided experiment.")
    # parser.add_argument(
    #     "-r",
    #     "--recompute",
    #     action="store",
    #     dest="recompute",
    #     required=False,
    #     type=str,
    #     default="False",
    #     help="Recompute all results.",
    # )
    parser.add_argument(
        "-ap",
        "--approach",
        action="store",
        dest="approach",
        required=False,
        type=str,
        default="dml",
        help="Reconstruction approach [cl/dml].",
    )
    parser.add_argument(
        "-so",
        "--solver",
        action="store",
        dest="solver",
        required=False,
        type=str,
        default="concorde",
        help="Solver.",
    )
    parser.add_argument(
        "-qt",
        "--query-st",
        action="store",
        dest="query_st",
        required=False,
        nargs="+",
        default=["opt-l", "opt-rl", "unc-l", "unc-rl", "opt-unc-l", "opt-unc-rl"],
        help="Query strategy.",
    )
    parser.add_argument(
        "-mi",
        "--max-num-iter",
        action="store",
        dest="max_num_iter",
        required=False,
        type=int,
        default=1,
        help="Max. number of iterations.",
    )
    parser.add_argument(
        "-np",
        "--num-proc",
        action="store",
        dest="num_proc",
        required=False,
        type=int,
        default=1,
        help="Number of processors for parallel computing.",
    )

    parser.add_argument(
        "-sb",
        "--solver-basedir",
        action="store",
        dest="solver_basedir",
        required=False,
        type=str,
        default="/tmp",
        help="Base directory for input/output files of the solver.",
    )
    parser.add_argument(
        "-sr",
        "--soft-range",
        action="store",
        dest="soft_range",
        required=False,
        type=float,
        default=100,
        help="Range for the softmax func.",
    )

    args = parser.parse_args()
    # recompute = str_to_bool(args.recompute)

    for query_st in args.query_st:
        assert query_st in [
            "opt-l",
            "opt-rl",
            "unc-l",
            "unc-rl",
            "opt-unc-l",
            "opt-unc-rl",
        ]

    fname = ".results/user_guided.json"
    records = []
    if os.path.exists(fname):
        for record in json.load(open(fname))["records"]:
            if (
                record["solver"] != args.solver
                or record["approach"] != args.approach
                or record["query_st"] not in args.query_st
            ):
                records.append(record)

    instances = []
    for instance in json.load(open(".results/multi.json"))["records"]:
        if instance["solver"] == args.solver and instance["approach"] == args.approach:
            instances.append(instance)
            instance["args"] = args

    # computed and add new records
    with Pool(processes=args.num_proc) as pool:
        for cnt, records_ in enumerate(
            pool.imap_unordered(process_instance, instances), 1
        ):
            records += records_
            for record in records_:
                print_report(cnt, len(instances), record)

    info_test = {"records": records}
    json.dump(info_test, open(fname, "w"))
    elapsed_time = time.time() - start
    print(
        "Elapsed time={:.2f} minutes ({} seconds)".format(
            elapsed_time / 60.0, elapsed_time
        )
    )
