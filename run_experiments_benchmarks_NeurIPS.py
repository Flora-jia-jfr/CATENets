"""
File to run the catenets experiments for
"Really Doing Great at Estimating CATE? A Critical Look at ML Benchmarking Practices in
Treatment Effect Estimation" (Curth & vdS, NeurIPS21)
from shell
"""
# Author: Alicia Curth
import argparse
import sys
from typing import Any

import catenets.logger as log
from experiments.experiments_benchmarks_NeurIPS21.acic_experiments_catenets import (
    do_acic_experiments,
)
from experiments.experiments_benchmarks_NeurIPS21.ihdp_experiments_catenets import (
    do_ihdp_experiments,
)
from experiments.experiments_benchmarks_NeurIPS21.twins_experiments_catenets import (
    do_twins_experiment_loop,
)
from experiments.experiments_benchmarks_NeurIPS21.speedDating_experiments_catenets import (
    do_speedDating_experiments,
)

log.add(sink=sys.stderr, level="DEBUG")


def init_arg() -> Any:
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="D", type=str)
    parser.add_argument("--experiment", default="ihdp", type=str)
    parser.add_argument("--file_name", default="results", type=str)
    parser.add_argument("--n_exp", default=100, type=int)
    parser.add_argument("--n_reps", default=5, type=int)
    # parser.add_argument("--n_exp", default=2, type=int)
    # parser.add_argument("--n_reps", default=2, type=int)
    parser.add_argument("--pre_trans", type=bool, default=False)
    parser.add_argument("--simu_num", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg()
    if args.experiment.lower() == "ihdp":
        do_ihdp_experiments(
            file_name=args.file_name,
            n_exp=args.n_exp,
            setting=args.setting,
            n_reps=args.n_reps,
        )
    elif args.experiment.lower() == "acic":
        do_acic_experiments(
            file_name=args.file_name,
            n_reps=args.n_reps,
            simu_num=args.simu_num,
            n_exp=args.n_exp,
            pre_trans=args.pre_trans,
        )
    elif args.experiment.lower() == "twins":
        do_twins_experiment_loop(file_name=args.file_name, n_exp=args.n_reps)
    # Implement SpeedDating
    elif args.experiment.lower() == "speeddating":
        do_speedDating_experiments()

    else:
        raise ValueError(
            f"Experiment should be one of ihdp/IHDP, acic/ACIC and twins/Twins. You "
            f"passed {args.experiment}"
        )
