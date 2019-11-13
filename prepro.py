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
    binar = False
    # cleaning

    vary, yn, mody = cleaning(vary, yn, binar)
    varx, xn, modx = cleaning(varx, xn, binar)

    # validation
    if yn.isnull().any().all():
        print("Error in the cleaning process")
    else:
        print("Cleaning process done")
    if yn.isnull().any().all():
        print("Error in the cleaning process")
    else:
        print("Cleaning process done")

    #scale and normalize
    xn = scale_norm(xn, varx, modx)
    print("Scaling and normalize process done")

    return xn, yn


def scale_norm(xn, var, mod):
    for i in range(len(var)):
        if mod[i] == 'numeric':
            xn.loc[:, xn.columns[i]] = pd.DataFrame(preprocessing.scale(xn[xn.columns[i]]), columns=[xn.columns[i]])
    return xn

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


def cleaning(var, xn, binar):
    mod = [[] for i in range(len(var))]
    for i in range(len(var)):

        if not var[i]:
            mod[i] = 'numeric'
            xn[xn.columns[i]].fillna(float(xn[xn.columns[i]].mean()), inplace=True)
        else:
            mod[i] = 'modal'
            if binar:
                mod = len(var[i])
                v = [i for i in range(len(var[i]))]
                print("Binarization...", xn.columns[i])
                xn[xn.columns[i]] = xn[xn.columns[i]].replace(var[i], v)
                var[i] = []

                #print(xn[xn.columns[i]].mean())

                xn.loc[:, xn.columns[i]] = replacement(xn[xn.columns[i]], mod)
                #print(xn[xn.columns[i]].mean())
            else:
                xn[xn.columns[i]].fillna(method ='bfill', inplace = True)


    return var, xn, mod
