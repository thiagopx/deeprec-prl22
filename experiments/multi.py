import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import json
import glob
import argparse
import tensorflow.compat.v1 as tf
import numpy as np
import random
import metrics
from strips import Strips
from compatibility import DeeprecDML, DeeprecCL
from solver import SolverConcorde, SolverNN, SolverKBH
from models import AffiNet, SqueezeNet
from experiments.utils import get_instances, transform_compatibilities_cost
from config import (
    DATASETS_TEST,
    MAP_DATASET_TRAIN_TEST,
    SAMPLES_SIZE_CL,
    SAMPLES_SIZE_DML,
    FEAT_BLOCK,
    FEAT_DIM,
    POOL_SIZE,
    VSHIFT,
    INPUT_SIZE_H,
    SEEDS,
)
from utils import str_to_bool


def get_datasets_test_sizes():
    map_dataset_test_sizes = dict()
    for dataset_test in DATASETS_TEST:
        instances = get_instances(dataset_test)
        strips_mixed = Strips(path=instances[0], filter_blanks=True)
        sizes = [strips_mixed.size()]
        for instance in instances[1:]:
            strips = Strips(path=instance, filter_blanks=True)
            sizes.append(strips.size())
        map_dataset_test_sizes[dataset_test] = sizes
    return map_dataset_test_sizes


if __name__ == "__main__":

    tf.disable_eager_execution()

    parser = argparse.ArgumentParser(description="Multi-page reconstruction.")
    parser.add_argument(
        "-r",
        "--recompute",
        action="store",
        dest="recompute",
        required=False,
        type=str,
        default="False",
        help="Recompute all results.",
    )
    parser.add_argument(
        "-a",
        "--approach",
        action="store",
        dest="approach",
        required=False,
        type=str,
        default="dml",
        help="Reconstruction approach [cl/dml].",
    )
    parser.add_argument(
        "-s",
        "--solver",
        action="store",
        dest="solver",
        required=False,
        type=str,
        default="concorde",
        help="Solver.",
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
        "-se",
        "--seeds",
        action="store",
        dest="seeds",
        required=False,
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds for the experiment.",
    )
    args = parser.parse_args()
    recompute = str_to_bool(args.recompute)

    assert args.approach in ["cl", "dml"]
    assert args.solver in ["concorde", "kbh", "nn"]
    solver_nn = SolverNN()
    solver_kbh = SolverKBH()

    # start cron
    start = time.time()

    # records of the results
    fname = ".results/multi.json"
    records = []
    if os.path.exists(fname):
        records = json.load(open(fname))["records"]
    if recompute:
        # delete the records for the specific solver and approach
        records = [
            record
            for record in records
            if record["seed"] != args.seed or record["solver"] != args.solver or record["approach"] != args.approach
        ]
    computed = []
    for record in records:
        computed.append((record["dataset_test"], record["seed"]))

    # size of each document was computed for the processed datasets
    map_datasets_test_sizes = get_datasets_test_sizes()
    os.makedirs(".results", exist_ok=True)
    os.makedirs(".results/multi_compatibilities_{}".format(args.approach), exist_ok=True)

    samples_size = SAMPLES_SIZE_CL if args.approach == "cl" else SAMPLES_SIZE_DML

    # graph definition
    tf.reset_default_graph()
    if args.approach == "dml":
        # placeholders
        image_ph1 = tf.placeholder(tf.float32, name="image_ph1", shape=(None, INPUT_SIZE_H, samples_size[1] // 2, 3))
        image_ph2 = tf.placeholder(tf.float32, name="image_ph2", shape=(None, INPUT_SIZE_H, samples_size[1] // 2, 3))

        # model
        model = AffiNet(
            image_ph1, image_ph2, samples_size[0], "test", "AffiNet", "sigmoid", FEAT_BLOCK, FEAT_DIM, POOL_SIZE
        )
        # pipelines (algorithm + solver)
        algorithm = DeeprecDML(
            model,
            VSHIFT,
            (INPUT_SIZE_H, samples_size[1]),
            verbose=False,
        )
    else:
        image_ph = tf.placeholder(tf.float32, name="image_ph", shape=(None, INPUT_SIZE_H, samples_size[1], 3))
        model = SqueezeNet(image_ph, include_top=True, num_classes=2, mode="test")
        algorithm = DeeprecCL(model, VSHIFT, (INPUT_SIZE_H, samples_size[1]), verbose=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.attach_session(sess)
        models_dir = glob.glob(".traindata/{}/*".format(args.approach))
        cnt = 1
        total = len(models_dir)
        for model_dir in models_dir:
            info_train = json.load(open("{}/info_train.json".format(model_dir)))
            # if info_train["seed"] not in SEEDS:
            #     continue
            seed = info_train["seed"]
            if seed not in args.seeds:
                continue

            model_id = info_train["model_id"]
            dataset_train = info_train["dataset_train"]

            random.seed(seed)
            np.random.seed(seed)
            info_val = json.load(open("{}/info_val.json".format(model_dir)))
            best_epoch = info_val["best_epoch"]

            # configure solver
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

            best_model_path = "{}/model/{}.npy".format(model_dir, best_epoch)
            model.load_vars(best_model_path)
            for dataset_test in DATASETS_TEST:
                # skip if the model was not trained with the designated dataset
                if dataset_test not in MAP_DATASET_TRAIN_TEST[dataset_train]:
                    continue

                # skip if the dataset was already processed
                if (dataset_test, seed, args.solver, args.approach) in computed:
                    continue

                # compatibilities
                compatibilities_path = ".results/multi_compatibilities_{}/{}_{}.npy".format(
                    args.approach, dataset_test, model_id
                )
                load_time = -1.0
                algorithm_time = -1.0
                # load compatibility matrix
                if os.path.exists(compatibilities_path):
                    compatibilities = np.load(compatibilities_path)
                    sizes = map_datasets_test_sizes[dataset_test]
                else:
                    # load strips
                    t0 = time.time()
                    instances = get_instances(dataset_test)
                    strips_mixed = Strips(path=instances[0], filter_blanks=True)
                    for instance in instances[1:]:
                        strips = Strips(path=instance, filter_blanks=True)
                        strips_mixed += strips
                    sizes = strips_mixed.sizes_per_doc()
                    load_time = time.time() - t0
                    t0 = time.time()
                    compatibilities = algorithm(strips=strips_mixed).compatibilities
                    algorithm_time = time.time() - t0
                    # save matrix
                    np.save(compatibilities_path, compatibilities)

                t0 = time.time()
                if args.approach == "cl":
                    compatibilities = transform_compatibilities_cost(compatibilities)
                solution = solver(instance=compatibilities).solution
                solver_time = time.time() - t0
                assert solution is not None

                sizes = map_datasets_test_sizes[dataset_test]
                accuracy = metrics.accuracy(solution, sizes)
                qc = metrics.Qc(compatibilities)

                print("[{}/{}] {} acc={:.2f}% qc={:.3f}".format(cnt, total, dataset_test, 100 * accuracy, qc))
                record = {
                    "dataset_test": dataset_test,
                    "compatibilities_path": compatibilities_path,
                    "sizes": sizes,
                    "accuracy": accuracy,
                    "qc": qc,
                    "solution": solution,
                    "load_time": load_time,
                    "algorithm_time": algorithm_time,
                    "solver_time": solver_time,
                    "approach": args.approach,
                    "solver": args.solver,
                    "seed": seed,
                }
                records.append(record)

                # save records
                info_test = {"records": records}
                json.dump(info_test, open(fname, "w"))

            cnt += 1

    elapsed_time = time.time() - start
    print("Elapsed time={:.2f} minutes ({} seconds)".format(elapsed_time / 60.0, elapsed_time))
