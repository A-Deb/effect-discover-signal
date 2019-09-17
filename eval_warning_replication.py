#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warning_replication
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from argparse import ArgumentParser
import pandas as pd
from datetime import datetime, timedelta

def main(args):
    external_signal_name = args.ext
    if args.method =='arimax':
        methods = ['arimax']
    else:
        methods = ['gru']

    dset= args.target

    if args.event =='endpoint-malware':
        etypes = ['endpoint-malware']

    data_sources = ["armstrong_endpoint-malware"]


   



    args_wsd='2018-06-01'
    args_wed='2018-09-16'
    freq='W'
    warn_start_date = pd.to_datetime(args_wsd)
    warn_end_date = pd.to_datetime(args_wed)
    warn_start_dates = pd.date_range(warn_start_date, warn_end_date, freq=freq,closed='left') + pd.Timedelta(days=1)
    warn_end_dates = pd.date_range(warn_start_date, warn_end_date, freq=freq, closed='right')
    if freq == 'M':
        warn_start_dates = warn_start_dates.insert(0, warn_start_date)
    else:
        warn_end_dates = warn_end_dates.delete(0)
        warn_start_dates = warn_start_dates.delete(-1)
    warn_start_dates=warn_start_dates.strftime('%Y%m%d')
    warn_end_dates=warn_end_dates.strftime('%Y%m%d')

    train_start_dates = ['20180101']
    
    time_freqs = ['D', '12H', '6H', '3H']
    time_freq = time_freqs[0]

    prediction_types = ['static', 'stochastic']
    prediction_type = prediction_types[0]
    n_trials = 20


    for eid, etype in enumerate(etypes):
        print("\n{} Event type: {} {} ".format("[]" * 20, etype, "[]" * 20))

        train_start_date = train_start_dates[eid]

        for method in methods:
            print("\n{} Method: {} {} ".format("=" * 20, method, "=" * 20))
            
            outdir_root = './score_warnings/'
            for i in range(len(warn_start_dates)):
                print("\n{} Start date: {} {} ".format("-" * 20,
                                                       warn_start_dates[i],
                                                       "-" * 20))
                warn_start_date = warn_start_dates[i]
                warn_end_date = warn_end_dates[i]

                params = []
                # params.append("-d{}_{}".format(dset, etype))
                params.extend(["-external_source", external_signal_name])
                params.extend(["-d", "{}_{}".format(dset, etype)])
                params.extend(["-e", etype])
                params.extend(["--train-start-date", train_start_date])
                params.extend(["--warn-start-date", warn_start_date])
                params.extend(["--warn-start-date", warn_start_date])
                params.extend(["--warn-end-date", warn_end_date])
                params.extend(["-m", method])
                params.extend(["--sensor", "antivirus"])
                params.extend(["--warn-version", "2"])
                params.extend(["--look-ahead", "7"])
                params.extend(["--time-freq", time_freq])
                params.extend(["--prediction-type", prediction_type])
                # params.extend(["--measure-performance", True])
                
                

                if method == "arimax":
                    params.extend(["--model-config-file",
                                   "data/config/{}_config.json".format(method)])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    params.extend(["--warning-dir", outdir_root])
                    print(params)
                    print('params loaded')
                    warning_replication.main(params)


if __name__ == "__main__":
    import sys

    parser = ArgumentParser()
    parser.add_argument('-ext', '--ext')
    parser.add_argument('-method','--method')
    parser.add_argument('-target','--target')
    parser.add_argument('-event','--event')
   

    args = parser.parse_args()
    main(args)

