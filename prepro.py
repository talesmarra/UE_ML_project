# Copyright (c) 2019, IMT Atlantique
# All rights reserved.

# ==============================================================================
"""Contains differents functions for the preprocessing of the data."""
import re
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing


def load_preprocessing_data(path,header = 'infer', index_col = None):

    df = pd.read_csv(path,header = header, index_col=index_col)
    lines = df.shape[0]
    columns = df.shape[1]
    x = df.drop(df.columns[columns - 1], axis=1)
    y = df.take([-1], axis=1)

    varx, xn = stringDetection(x)
    print("String detection process done")
    vary, yn = stringDetection(y)
    print("String detection process done")

    return xn, yn




def stringDetection(x):
    """
    x as a pandas array
    """
    xn = pd.DataFrame(columns=x.columns, index=x.index)
    var = []

    for j in range(x.shape[1]):
        j = []
        var.append(j)
    for i, row in enumerate(x.values):
        for j in range(x.shape[1]):
            try:
                row[j] = float(row[j])
            except ValueError:
                row[j] = row[j]
            if isinstance(row[j], str):
                row[j] = re.sub(r"[^a-zA-Z0-9]+", '', row[j])
                if ((row[j] in var[j]) == False) and (row[j] != ''):
                    var[j].append(row[j])
                if (row[j] == ''):
                    row[j] = float('nan')
        xn.loc[i] = row

    return var, xn


def replacement(xn, mod):
    xf = pd.DataFrame(np.zeros(shape=(xn.shape[0], 1)), columns=[xn.name], index=np.arange(0, xn.shape[0]))
    if mod == 2:
        p = xn.mean()
        m = xn.isnull().sum()
        v = np.random.binomial(1, round(p, 2), m)
        j = 0
        for k in range(xn.shape[0]):
            if np.isnan(xn[k]):
                xf.loc[k] = v[j]
                j += 1
            else:
                xf.loc[k] = xn[k]
    return xf

