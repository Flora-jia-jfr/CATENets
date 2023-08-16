"""
Utils to run SpeedDating experiments with catenets
Modified by Flora Jia
"""
import csv
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn import clone

# TODO: change imports
from catenets.datasets.dataset_speedDating import load_data, split_data
from catenets.experiment_utils.base import eval_root_mse
from catenets.models.jax import RNET_NAME, T_NAME, TARNET_NAME, TARNET_SINGLE_NAME, RNet, TARNet, TNet, TARNet_single
from catenets.models.jax import TARNet_SINGLE_2_NAME, TARNet_single_2

DATA_DIR = Path("catenets/datasets/data/SpeedDatingDat")
RESULT_DIR = Path("results/experiments_benchmarking/speedDating/comparison/")
SEP = "_"

PARAMS_DEPTH = {"n_layers_r": 3, "n_layers_out": 2}
PARAMS_DEPTH_2 = {
    "n_layers_r": 3,
    "n_layers_out": 2,
    "n_layers_r_t": 3,
    "n_layers_out_t": 2,
}

ALL_MODELS = {
    # T_NAME: TNet(**PARAMS_DEPTH),
    # RNET_NAME: RNet(**PARAMS_DEPTH_2),
    TARNET_SINGLE_NAME: TARNet_single(**PARAMS_DEPTH),
    TARNET_NAME: TARNet(**PARAMS_DEPTH),
    TARNet_SINGLE_2_NAME: TARNet_single_2(**PARAMS_DEPTH)
}


def do_speedDating_experiments(
    file_name: str = "speedDating",
    model_params: Optional[dict] = None,
    models: Optional[dict] = None,
) -> None:
    if models is None:
        models = ALL_MODELS

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    out_file = open(RESULT_DIR / (file_name + ".csv"), "w", buffering=1)
    writer = csv.writer(out_file)
    # according to SpeedDating: mod, dim, dat
    header = (
        ["mod", "dim", "run", "cate_var_in", "cate_var_out", "y_var_in"]
        + [name + "_in" for name in models.keys()]
        + [name + "_out" for name in models.keys()]
        + [name + "_ate_in" for name in models.keys()]
        + [name + "_ate_out" for name in models.keys()]
    )
    writer.writerow(header)

    mods = range(1, 5)
    dims = ['high', 'med', 'low']
    dats = range(1,11)

    for mod_num in mods:
        for dim in dims:
            for dat in dats:
                # TODO: get data within loop
                X, Y, W, ITE_oracle = load_data(DATA_DIR, mod_num, dim, dat)
                X, y, w, cate_true_in, X_t, y_t, w_t, cate_true_out = split_data(X, Y, W, ITE_oracle)
                
                # compute some stats
                cate_var_in = np.var(cate_true_in)
                cate_var_out = np.var(cate_true_out)
                y_var_in = np.var(y)

                pehe_in = []
                pehe_out = []
                ate_in = []
                ate_out = []

                for model_name, estimator in models.items():
                    print(f"Mod {mod_num}, Dim {dim}, dat {dat}, with {model_name}")
                    estimator_temp = clone(estimator)
                    estimator_temp.set_params(seed=dat)
                    if model_params is not None:
                        estimator_temp.set_params(**model_params)

                    # fit estimator
                    estimator_temp.fit(X=X, y=y, w=w)

                    cate_pred_in = estimator_temp.predict(X, return_po=False)
                    cate_pred_out = estimator_temp.predict(X_t, return_po=False)

                    pehe_in.append(eval_root_mse(cate_pred_in, cate_true_in))
                    pehe_out.append(eval_root_mse(cate_pred_out, cate_true_out))

                    ate_train = np.mean(cate_true_in)
                    ate_test = np.mean(cate_true_out)
                    ate_in.append(abs(np.mean(cate_pred_in) - ate_train))
                    ate_out.append(abs(np.mean(cate_pred_out) - ate_test)) 

                writer.writerow(
                    [mod_num, dim, dat, cate_var_in, cate_var_out, y_var_in] + pehe_in + pehe_out + ate_in + ate_out
                )

    out_file.close()
    