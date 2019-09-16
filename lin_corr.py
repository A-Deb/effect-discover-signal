#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import ccf
import utils.dataprocessing as dp

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def pairwise_ccf(ts_ref, ts_exogs, slice_start_date=None,
                 slice_end_date=None, ccf_lag_thr=30, k=5,
                 normalized=False, selected_lag=None, corr_th=None):
    # print(type(ts_exogs))
    # if slice_start_date is None:
    #     slice_start_date = ts_ref.index.min().date()
    # if slice_end_date is None:
    #     slice_end_date = ts_ref.index.max().date()
    print("Corr analysis period:")
    print("From", slice_start_date, "To", slice_end_date)

    ts_ref_sliced = ts_ref[slice_start_date:slice_end_date]
    ccf_vals = list()
    cols = ['feature']
    for i in range(1, k + 1):
        cols.append("max_val_" + str(i) + "_lag")
        cols.append("max_val_" + str(i))

    ts_dic = dict()
    for ts_exog_name in ts_exogs:
        # print(ts_exogs)
        ts_exog = ts_exogs[ts_exog_name]
        # print(type(ts_exog))
        # print(type(ts_exogs))
        # print(ts_exog)
        # print(ts_exog.head())
        # input("press a key")

        ts_candidate_sliced = ts_exog[slice_start_date:slice_end_date]
        if normalized:
            ts_ref_sliced = (ts_ref_sliced - ts_ref_sliced.values.min()) / \
                            (ts_ref_sliced.values.max() -
                             ts_ref_sliced.values.min())
            ts_candidate_sliced = (ts_candidate_sliced -
                                   ts_candidate_sliced.values.min()) / (
                                      ts_candidate_sliced.values.max() -
                                      ts_candidate_sliced.values.min())
        res_ccf = ccf(ts_ref_sliced.values, ts_candidate_sliced.values,
                      unbiased=False)
        # print(ts_ref_sliced.shape, ts_candidate_sliced.shape)

        res_ccf_sub = res_ccf[:ccf_lag_thr]
        # print("ccf:")
        # print(np.around(res_ccf_sub, decimals=2))
        res_ccf_sub_abs = np.abs(res_ccf_sub)
        indx = np.argsort(-res_ccf_sub)[:k]
        val = res_ccf_sub[indx]

        temp_list = [ts_exog.name]
        for i in range(k):
            temp_list.append(indx[i])
            temp_list.append(val[i])
        ccf_vals.append(temp_list)
        ts_dic[ts_exog.name] = ts_exog

    df_cc = pd.DataFrame(ccf_vals, columns=cols)

    # create external signals for all of the keywords
    lagged_ts = pd.DataFrame(data=None)

    if corr_th is None:
        for key in ts_dic:
            row = df_cc[df_cc['feature'] == key]
            if selected_lag is None:
                lag = row['max_val_1_lag'].values[0]
            else:
                lag = selected_lag[key]
            ts = ts_dic[key]
            ts_candidate_sliced = ts[slice_start_date:]
            ts_exog = ts_candidate_sliced.shift(periods=lag)
            ts_exog.fillna(inplace=True, value=0)
            lagged_ts[key] = ts_exog
    else:
        for key in ts_dic:
            row = df_cc[df_cc['feature'] == key]
            if selected_lag is None:
                lag = row['max_val_1_lag'].values[0]
                corr_val = row['max_val_1'].values[0]
                if corr_val < corr_th:
                    continue
            else:
                lag = selected_lag[key]
            ts = ts_dic[key]
            ts_candidate_sliced = ts[slice_start_date:]
            ts_exog = ts_candidate_sliced.shift(periods=lag)
            ts_exog.fillna(inplace=True, value=0)
            lagged_ts[key] = ts_exog

    return df_cc, lagged_ts


def get_aligned_external_signals(ts_ref, ts_exogs, slice_start_date=None,
                                 slice_end_date=None, lag=None):
    if slice_start_date is None:
        slice_start_date = ts_ref.index.min().date()

    lagged_ts = pd.DataFrame(data=None)
    for indx, ts in enumerate(ts_exogs):
        ts = ts.shift(periods=lag[indx])
        ts.fillna(inplace=True, value=0)
        lagged_ts["feature_" + str(indx)] = ts[slice_start_date:]

    return lagged_ts


def main(argv):
    # [] load ground truth data
    ts_dexter_epmal, query = dp.get_ground_truth_ts_es(
        company="dexter", event_type="endpoint-malware")
    print("Num of dexter ep malware =", ts_dexter_epmal.sum())

    ts_armstrong_epmal, query = dp.get_ground_truth_ts_es(
        company="armstrong", event_type="endpoint-malware")
    print("Num of armstrong ep malware =", ts_armstrong_epmal.sum())

    ts_knox_epmal = dp.load_knox_endpoint_malware()
    print("Num of armstrong ep malware =", ts_knox_epmal.sum())

    # ============== load external signals ======================
    ts_absusech = dp.get_ts_malware()
    # ts_absusech = dp.get_ts_malware_exclue_Locky()
    ts_absusech.name = "abusech"
    print("Num of abuse.ch ep malware =", ts_absusech.sum())
    print(ts_absusech.index.min())
    print(ts_absusech.index.max())

    ts_phistank = dp.get_ts_malware(publisher="hg-taxii")
    ts_phistank.name = "phistank"
    print("Num of phistank malware =", ts_phistank.sum())
    print(ts_phistank.index.min())
    print(ts_phistank.index.max())

    # ============== linear correlation analysis ===============
    ts_exogs = [ts_absusech, ts_phistank]

    print("dexter:")
    start_date = ts_dexter_epmal.index.min().date() - pd.Timedelta(days=30)
    print("start_date =", start_date)
    df_cc, lagged_ts = pairwise_ccf(ts_dexter_epmal, ts_exogs,
                                    slice_start_date=start_date,
                                    slice_end_date="2016-06-30")

    filedir = os.path.join("output", "dexter_endpoint-malware", "correlation")
    os.makedirs(filedir, exist_ok=True)
    filepath = os.path.join(filedir, "cross_corr.csv")
    df_cc = np.around(df_cc, decimals=2)
    df_cc.to_csv(filepath, index=False)

    print("armstrong:")
    start_date = ts_armstrong_epmal.index.min().date() - pd.Timedelta(days=30)
    print("start_date =", start_date)
    df_cc, lagged_ts = pairwise_ccf(ts_armstrong_epmal, ts_exogs,
                                    slice_start_date=start_date,
                                    slice_end_date="2016-08-31")
    filedir = os.path.join("output", "armstrong_endpoint-malware",
                           "correlation")
    os.makedirs(filedir, exist_ok=True)
    filepath = os.path.join(filedir, "cross_corr.csv")
    df_cc = np.around(df_cc, decimals=2)
    df_cc.to_csv(filepath, index=False)

    print("knox:")
    print("Start Date =", ts_knox_epmal.index.min())
    print("End Date =", ts_knox_epmal.index.max())

    start_date = ts_knox_epmal.index.min().date() - pd.Timedelta(days=30)
    print("External signals start_date =", start_date)
    df_cc, lagged_ts = pairwise_ccf(ts_knox_epmal, ts_exogs,
                                    slice_start_date=start_date,
                                    slice_end_date="2017-06-30")
    filedir = os.path.join("output", "knox_endpoint-malware", "correlation")
    os.makedirs(filedir, exist_ok=True)
    filepath = os.path.join(filedir, "cross_corr.csv")
    df_cc = np.around(df_cc, decimals=2)
    df_cc.to_csv(filepath, index=False)


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
