#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__version__ = 1.0
__author__ = "Tozammel Hossain, Shuyang Gao"
__email__ = "tozammel@isi.edu, sgao@isi.edu"

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error


def get_rate(true_ts, pred_ts):
    rate = pred_ts.sum() / float(len(pred_ts))
    return rate


def get_precision_recall(true_ts, pred_ts):
    ratio = float(true_ts.sum()) / float(pred_ts.sum())
    return min(ratio, 1. / ratio)


def get_nash_sutcliffe_score(true_ts, pred_ts):
    nash_sutcliffe = 1 - ((true_ts - pred_ts) ** 2).sum() / \
                         ((true_ts - true_ts.mean()) ** 2).sum()

    return nash_sutcliffe


def get_mae(true_ts, pred_ts):
    mae = mean_absolute_error(true_ts, pred_ts)
    return mae


def get_rmse(true_ts, pred_ts):
    mse = mean_squared_error(true_ts, pred_ts, multioutput='raw_values')[0]
    rmse = np.sqrt(mse)
    return rmse


def get_mase(true_ts, pred_ts):
    # mean absolute error
    mae = mean_absolute_error(true_ts, pred_ts)
    # mean absolute scaled error (see wiki)
    mase = mae / mean_absolute_error(true_ts[1:], true_ts[:-1])
    return mase


def get_scores(true_ts, pred_ts):
    # mean squared error
    # norm mean squared error
    mse = mean_squared_error(true_ts, pred_ts, multioutput='raw_values')[0]
    nmse = mse / (true_ts.mean() * pred_ts.mean())
    rmse = np.sqrt(mse)
    # norm rmse
    # ref 1:
    # https://docs.google.com/viewer?url=http%3A%2F%2Fwww.ctec.ufal.br%2Fprofessor%2Fcrfj%2FGraduacao%2FMSH%2FModel%2520evaluation%2520methods.doc
    nrmse = rmse / (max(pred_ts) - min(pred_ts))
    # ref 2:
    nrmse2 = rmse / (max(true_ts) - min(true_ts))
    # ref 3: https://www.mathworks.com/help/ident/ref/goodnessoffit.html
    # this is similar to Nash-Sutcliffe coefficient as in Ref 1
    # https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient
    nrmse3 = 1 - np.linalg.norm((true_ts - pred_ts), 2) / \
                 np.linalg.norm((true_ts - true_ts.mean()), 2)

    # Nash-Sutcliffe coefficient or R^2 (coefficient of determination)
    nash_sutcliffe = 1 - ((true_ts - pred_ts) ** 2).sum() / \
                         ((true_ts - true_ts.mean()) ** 2).sum()

    scatter_index = rmse / np.mean(pred_ts)
    scatter_index2 = rmse / np.mean(true_ts)

    # mean absolute error
    mae = mean_absolute_error(true_ts, pred_ts)

    # mean absolute scaled error (see wiki)
    mase = mae / mean_absolute_error(true_ts[1:], true_ts[:-1])

    # median absolute error
    medae = median_absolute_error(true_ts, pred_ts)
    # return mse, nmse, rmse, nrmse, nrmse2, nrmse3, scatter_index, \
    #     scatter_index2, mae, mase, medae, nash_sutcliffe

    # nrmse, nrmse3, scatter_index are not good
    return mse, nmse, rmse, nrmse2, scatter_index2, mae, mase, \
           medae, nash_sutcliffe
