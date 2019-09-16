#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import pandas as pd
import glob
import numpy as np

from utils import dataprocessing as dp

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def save_ts(ts):
    pass


def load_mirai_scan():
    filedir = "/Users/tozammel/cause/isi-code/effect-forecasting-models/data/mirai"
    filename = "miraiScan_ts.h5"
    filepath = os.path.join(filedir, filename)

    data = pd.read_hdf(filepath)
    # data.columns = ['date', 'count']
    print(type(data))
    print(data.head())


def load_deep_exploit():
    """
    all.csv: contains 278 unique CVEs
    armstrong.csv: contains 86 CVEs
    """
    filedir = "/Users/tozammel/cause/isi-code/effect-forecasting-models/data/deep-exploit"
    filename1 = "all.csv"
    filename2 = "armstrong.csv"
    filepath = os.path.join(filedir, filename1)
    filepath = os.path.join(filedir, filename2)

    data = pd.read_csv(filepath)


def security_keywords_ts():
    dirpath = "/Users/tozammel/cause/isi-code/effect-forecasting-models/data/security-keyword-counts"
    ext = "csv"
    files = [file for file in
             glob.glob(dirpath + '/**/*.' + ext, recursive=True)]
    tslist = [pd.read_csv(file, header=-1, parse_dates=True, index_col=0,
                          squeeze=True, names=['date', 'count']) for file in
              files]

    print(tslist[0].head())


    # for f in filelist:
    #     fname = os.path.split(f)[1]
    #     fname_wo_ext = os.path.splitext(fname)[0]
    #     ts = pd.read_csv(f, header=header, parse_dates=True, index_col=0,
    #                      squeeze=True)
    #     ts.name = name
    #     ts.index.name = index_name
    #     cts = TimeSeries(ts, name=fname_wo_ext)
    #     tslist.append(cts)


def load_abusech_exclude_locky():
    # ts_absusech = dp.get_ts_malware(publisher="hg-abusech")
    ts_absusech = dp.get_ts_malware_exclue_Locky()
    ts_absusech.name = "abusech-wo-locky"
    return ts_absusech

def load_abusech():
    ts_absusech = dp.get_ts_malware(publisher="hg-abusech")
    # ts_absusech = dp.get_ts_malware_exclue_Locky()
    ts_absusech.name = "abusech"
    return ts_absusech

def load_phistank():
    ts_phistank = dp.get_ts_malware(publisher="hg-taxii")
    ts_phistank.name = "phistank"
    return ts_phistank


def load_d2web(dirpath=None, ext="csv", header=0, save=True):
    if dirpath is None:
        dirpath = "/Users/tozammel/cause/isi-code/effect-forecasting/data/timeseries_for_expertiments/d2web/TS"

    files = [file for file in
             glob.glob(dirpath + '/**/*.' + ext, recursive=True)]
    dflist = [pd.read_csv(file, header=header, parse_dates=True, index_col=1,
                          squeeze=True, names=['indx', 'date', 'count'])
              for file in files]
    tslist = [df['count'] for df in dflist]
    # print("#size = {}".format(len(tslist)))
    # print(tslist[0].head())

    colnames = [os.path.splitext(os.path.split(x)[1])[0] for x in files]
    df = pd.concat(tslist, axis=1)
    df.columns = colnames
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(value=0, inplace=True)
    # print(df.head())
    if save:
        dirpath = "/Users/tozammel/cause/isi-code/effect-forecasting/data/timeseries_for_expertiments/d2web"
        filepath = os.path.join(dirpath, "d2web.csv")
        print("Saving {}".format(filepath))
        df.to_csv(filepath)
    return df


def load_d2web_df(filepath=None, header=0):
    if filepath is None:
        filepath = "/Users/tozammel/cause/isi-code/effect-forecasting/data/timeseries_for_expertiments/d2web/d2web.csv"
    return pd.read_csv(filepath, header=0, index_col=0, parse_dates=True)


def load_external_data(name):
    if name == "d2web":
        return load_d2web_df()


def main(argv):
    """
External data sources:
1. Abuse.ch
> All types
> All but locky
> Cerber only
> Representative  signals
2. Raptor
3. Discover
4. 


:param argv: 
:return: 
"""
    pass
    # load_mirai_scan()
    # security_keywords_ts()

    # d2wb
    load_d2web()
    load_d2web_df()


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
