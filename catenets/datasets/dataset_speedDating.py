"""
SpeedDating dataset
"""
# stdlib
import os
import random
from pathlib import Path
from typing import Any, Tuple

# third party
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import catenets.logger as log

from .network import download_if_needed

torch.manual_seed(1)


# Helper functions to load data

def load_data(file_path, mod_num, dim, dat):
    df_path = os.path.join(file_path,
                       'Mod{}/speedDateMod{}{}{}.csv'.format(mod_num, mod_num, dim, str(dat)))
    df = pd.read_csv(df_path)
    data = df.values
    oracle = pd.read_csv(os.path.join(
        file_path, 'Mod{}/speedDateMod{}{}Oracle{}.csv'.format(mod_num, mod_num, dim, str(dat))))
    ITE_oracle = oracle['ITE'].values.reshape(-1, 1)

    # print("data", data.shape) #(6000, 187)
    # rows: different data, columns: Y, A(our T), 185 covariates

    Y = data[:, 0].reshape(-1, 1) # (6000, 1)
    W = data[:, 1].reshape(-1, 1) # (6000, 1)
    X = data[:, 2:] # (6000, 185)

    return X, Y, W, ITE_oracle

def prepare_data(X, Y, T, ITE_oracle):
    return {
        'covariates': torch.from_numpy(X),
        'outcome': torch.from_numpy(Y),
        'treatment': torch.from_numpy(T),
        'ITE_oracle': torch.from_numpy(ITE_oracle)
    }

def split_data(X, Y, W, ITE_oracle, train_split=0.75):
    data_num = X.shape[0]
    train_num = int(data_num*train_split)
    test_num = data_num - train_num
    return  X[:train_num], Y[:train_num], W[:train_num], ITE_oracle[:train_num], \
            X[train_num:], Y[train_num:], W[train_num:], ITE_oracle[train_num:]