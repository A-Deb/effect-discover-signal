#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
from utils import custom_plot
from utils import dataprocessing as dp
from utils import model_wrapper
from utils import model_output

import csv
import json
import codecs
import warnings



import gen_warning_arimax as arimax


from util.measures import measures
import data_external as external_source
import lin_corr
import sys
import datetime
import pdb
import io
from scoring.metrics import Metrics
from scoring.metrics_objects import MetricGroundTruth, MetricWarning
from scoring.formatting_functions import format_gt, format_warn
from scoring.pair_objects_for_notebook import Pair
from munkres import Munkres
from score_gt import load_warnings
from score_gt import load_gt 

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"



def valid_date(s):
    try:
        return pd.datetime.strptime(s, "%Y%m%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def load_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)

    # data source group
    datagrp = parser.add_argument_group("[*] Data Source Options")
    datagrp.add_argument(
        '-d', '--data-source',
        default='ransomware_cerber',
        help='Data soucres for learning the daywise baserate model. \
        Possible options: \
        ransomware_locky, \
        ransomware_cerber, \
        dexter_endpoint-malware,\
        dexter_malicious-email,\
        dexter_malicious-destination, \
        armstrong_endpoint-malware,\
        armstrong_malicious-email,\
        knox_endpoint-malware,\
        knox_malicious-destination, and \
        knox_malicious-email.\
        Default value is ransomware_cerber, which denotes ransomware malware \
        with Cerber types.')
    datagrp.add_argument(
        '-w', '--window-size',
        default=100, type=int,
        help='Training window size. Default = 50 days.')
    datagrp.add_argument(
        '--data-hostname',
        default='http://cloudweb01.isi.edu/es/',
        help='Hostname for ES')
    datagrp.add_argument(
        '--data-replication',
        default=False, type=bool,
        help='Replication of warning')

    # model group
    modelgrp = parser.add_argument_group("[*] Model Options")
    modelgrp.add_argument('-m', '--model')
    modelgrp.add_argument('--model-config-file', type=str, default=None)

    # warning group
    warngrp = parser.add_argument_group("[*] Warning Generation Options")
    warngrp.add_argument(
        '--warn-start-date',
        default=pd.datetime.now().date() + pd.Timedelta(days=1),
        help='Warning start date. If None, tomorrow will\
        be the warning start date.', type=valid_date)
    warngrp.add_argument(
        '--warn-end-date',
        default=None,
        help='Warning end date. If None, tomorrow will\
        be the warning start date.')
    warngrp.add_argument(
        '-l', '--look-ahead',
        default=1, type=int,
        help='Number of days to be forecasted without GT data. Default value \
        is 1.')
    warngrp.add_argument(
        '-external_source', '--external_source',
        default='iot.csv',
        help='Specify external signal.')
    warngrp.add_argument(
        '--event-type-broad',
        default='malware',
        help='Generic class for event types.')
    warngrp.add_argument(
        '-e', '--event-type',
        default='endpoint-malware',
        help='Generate warnings for event-type. Possible options: \
        endpoint-malware, malicious-destination, malicious-email.')
    warngrp.add_argument(
        '--warning-dir',
        default='warnings_hourly_17',
        help='Save json warning into this directory.')

    warngrp.add_argument(
        '--model-metadata-dir',
        default='model_metadata/baserate',
        help='Save model metadata in json format into this directory.')

    warngrp.add_argument(
        '--sensor',
        default='ransomwaretracker.abuse.ch',
        help='Sensor field for the warning.')

    parser.add_argument(
        '--feature-dist-file', type=str, default=None)

    warngrp.add_argument(
        '--warn-version',
        default=2,
        help='Warning format version')
    warngrp.add_argument(
        '--train-start-date',
        help='Warning start date', default=None)
    warngrp.add_argument(
        '--time-freq',
        help='Time unit', default='12H')
    warngrp.add_argument(
        '--prediction-type',
        default='stochastic',
        help='Number of events is drawn id two ways: a) static: estimated mean \
        is given as output and b) stochastic: estimated mean is used draw an int from a \
        Poisson distribution')
    warngrp.add_argument(
        '--sample-warning',
        help='Sample warnings', default=True)

    # optional group
    othergrp = parser.add_argument_group("[*] Other Options")
    othergrp.add_argument(
        '-p', '--show-plot',
        default=True, type=bool,
        help='Draw plots')
    othergrp.add_argument(
        '--measure-performance',
        default=False, type=bool,
        help='Draw plots')

    return parser.parse_args(argv)


def load_external_signals(ext_source_name):
    print("External source name = {}".format(ext_signal_name))
    pass


def run_arimax(args, ts, es_query, train_start_date, train_end_date,
               look_ahead):
    print(">>Running {}".format(args.model))
    model_param={"max_p": 12,"max_d": 2,"max_q": 12}
    print("Model param:", model_param)
    d = vars(args)
    d.update(model_param)

    # <editor-fold desc="load external data">
    if args.external_source is not None:
        print("Loading external signals: %s" % args.external_source)
        ex_index = "data"
        if args.data_replication:
            ex_index = ex_index + '-' + (
                args.warn_start_date +
                pd.Timedelta(days=-1)).strftime("%Y%m%d") + '/malware'
            # print("Replication index: ", ex_index)
        else:
            ex_index = 'data/malware'
        # print("\tindex: ", ex_index)
        # ext_sources = args.external_source.split(";")
        ext_source_name = args.external_source

        
        
        filename = './data/ext_sig/%s' % ext_source_name
        ts_ext = pd.read_csv(filename)
        ts_ext['date'] = pd.to_datetime(ts_ext['date'])
        ts_ext.set_index('date', inplace=True)
        ts_ext.name = ext_source_name.split('.')[0]
        ts_exogs = pd.DataFrame(ts_ext)
       


        slice_start_date = ts.index.min().date() - pd.Timedelta(days=30)
        slice_end_date = train_end_date

        print("External signals:")
        print(ts_exogs.index.min())
        print(ts_exogs.index.max())
        print(args.data_source)
        # exit(1)
        df_cc, lagged_ts = lin_corr.pairwise_ccf(
            ts, ts_exogs, slice_start_date=slice_start_date,
            slice_end_date=slice_end_date)

        print("Cross Correlation:")
        print(df_cc)

        ts_external_shifted = lagged_ts[train_start_date:]

        print("External Data:")
        print("\tStart date:", ts_external_shifted.index.min())
        print("\tEnd date:", ts_external_shifted.index.max())
    else:
        ts_external_shifted = None
    # </editor-fold>

    ts_pred, arimax_order = arimax.learn_arimax_parameters(
        ts, ts_external_shifted, args, look_ahead=look_ahead,
        train_end_date=train_end_date,
        test_start_date=args.warn_start_date)
    ts_pred_rounded = np.around(ts_pred)
    model_id = None
    output_name = None
    if ts_pred_rounded.sum() > 0:
        # generate model metadata
        model_id = arimax.write_model_metadata(args, es_query, arimax_order)
        output_name = "arimax_warning"

    return ts_pred, ts_pred_rounded, model_id, output_name



def main(argv):
    
    args = load_arguments(argv)
    # Load ground truth data
    # ts, es_query = dp.gt_time_series(args)
    
    #ts = dp.load_armstrong_internal()
    ts = dp.load_armstrong_endpoint_malware()
    es_query={}
    if ts is None:
        # print("Error: time series is not loaded properly")
        sys.exit("Error: time series data is not loaded properly.")

    if args.train_start_date is None:
        train_start_date = min(ts.index)
    else:
        train_start_date = pd.to_datetime(args.train_start_date)

    if ts.index.max() >= args.warn_start_date:
        train_end_date = args.warn_start_date - pd.Timedelta(days=1)
    else:
        train_end_date = ts.index.max()

    warn_end_date = args.warn_end_date
    if warn_end_date is None:
        warn_end_date = ts.index.max()

    look_ahead = len(pd.date_range(args.warn_start_date, warn_end_date))
    d = vars(args)
    #d['look-ahead'] = look_ahead
   

    # <editor-fold desc="show stats">
    print("Training Data:")
    print("\tStart date:", train_start_date)
    print("\tEnd date:", train_end_date)
    print("\tWarning start date:", args.warn_start_date)
    print("\tForecasting look ahead days:", look_ahead)
    print("\tNumber of days:", ts.shape[0])
    print("\tNumber of data points:", ts.sum())
    print("\tMax num of events occurred in a day:", max(ts))
    print("\tMin num of events occurred in a day:", min(ts))
    print(d)
    # </editor-fold>


    print('which model')


    if args.model == "arima":
        print(">>Running {}".format(args.model))
        # model_param = json.load(codecs.open("data/config/arima_config.json"))
        model_param = json.load(codecs.open(args.model_config_file))
        print("Model param:", model_param)
        d = vars(args)
        d.update(model_param)

        ts_pred, arima_order = arima.learn_arima_parameters(
            ts, args, train_end_date=train_end_date,
            test_start_date=args.warn_start_date)

        ts_pred_rounded = np.around(ts_pred)
        if ts_pred_rounded.sum() > 0:
            # generate model metadata
            model_id = arima.write_model_metadata(args, es_query, arima_order)
            output_name = "arima_warning"

    elif args.model == "arimax":
        print('running arimax')
        
        ts_pred, ts_pred_rounded, model_id, output_name = run_arimax(args, ts, es_query, train_start_date, train_end_date, look_ahead)
        
   
    else:
        msg = "Not a valid model {}".format(args.model)
        raise argparse.ArgumentTypeError(msg)

    # generate warnings

    if ts_pred_rounded.sum() > 0:

        company = model_output.get_target_orgnization(args)
        if args.sample_warning:
            if company == "knox":
                warning_obj = model_output.write_model_output_v2(
                    args, ts_pred_rounded, model_id,
                    output_name=output_name, save_warning=False)
            else:
                warning_obj = model_output.write_model_output_v2(
                    args, ts_pred_rounded, model_id,
                    output_name=output_name, save_warning=False)
            # save warning obj
            id_prefix = warning_obj.warnings[0]['warning']['id'].rsplit("T", 1)[
                0]
            # print(id_prefix)
            ext_name = args.external_source.split('.')[0]
            filename = "warning_" + ext_name + "_" + id_prefix + ".json"
            filepath = os.path.join(args.warning_dir, filename)
            os.makedirs(os.path.split(filepath)[0], exist_ok=True)
            warning_obj.save(filepath)

        filename = "ts_" + ext_name + "_" + model_id + ".csv"
        filepath = os.path.join(args.warning_dir, filename)
        os.makedirs(os.path.split(filepath)[0], exist_ok=True)
        print("Saving :{}".format(filepath))
        ts_pred_rounded.to_csv(filepath, header=True)
    else:
        print("Model did not predict any attacks. " +
              "There will be no warnings.")

    print("Total = ", ts_pred_rounded.sum())
    target_org='armstrong'

    print("=" * 10, "Performance Measure", "=" * 10)
    print("true val:")
    print(ts.index.min())
    print(ts.index.max())

    print("pred val:")
    print(ts_pred.index.min(), ts_pred.index.max())

    #ps, pe = ts_pred.index.min(), ts_pred.index.max()
    ps, pe = ts_pred_rounded.index.min(), ts_pred_rounded.index.max()
    s, e = ts_pred_rounded.index.min().date(), ts_pred_rounded.index.max().date()
    print(s, e)


    
    #s, e = ts_pred_rounded.index.min().date(), ts_pred_rounded.index.max().date()
    print(s)
    print(e)
    # if ts.index.max().date() < e:
        # e = ts.index.max().date()
    mae = measures.get_mae(ts[s:e], ts_pred[ps:pe])
    mae_round = measures.get_mae(ts[s:e], ts_pred_rounded[ps:pe])
    rmse = measures.get_rmse(ts[s:e], ts_pred[ps:pe])
    rmse_round = measures.get_rmse(ts[s:e], ts_pred_rounded[ps:pe])
    mase = measures.get_mase(ts[s:e], ts_pred[ps:pe])
    mase_round = measures.get_mase(ts[s:e], ts_pred_rounded[ps:pe])

    firsta=np.max(ts[s:e])
    firstb=np.max(ts_pred[ps:pe])
    first=max(firsta,firstb)
    seconda=np.min(ts[s:e])
    secondb=np.min(ts_pred[ps:pe])
    second=min(seconda, secondb)
    third=first-second
    nrmse = rmse / third

    # first=np.max(ts[s:e], ts_pred_rounded[ps:pe])
    # second=np.min(ts[s:e], ts_pred_rounded[ps:pe])
    # third=first-second
    # nrmse_round = rmse_round / third
    firsta=np.max(ts[s:e])
    firstb=np.max(ts_pred_rounded[ps:pe])
    first=max(firsta,firstb)
    seconda=np.min(ts[s:e])
    secondb=np.min(ts_pred_rounded[ps:pe])
    second=min(seconda, secondb)
    third=first-second
    nrmse_round = rmse_round / third


    mae = np.around(mae,decimals=2)
    mae_round = np.around(mae_round,decimals=2)
    rmse = np.around(rmse,decimals=2)
    rmse_round = np.around(rmse_round,decimals=2)
    mase = np.around(mase,decimals=2)
    mase_round = np.around(mase_round,decimals=2)
    nrmse = np.around(nrmse,decimals=2)
    nrmse_round = np.around(nrmse_round,decimals=2)



   

    evt=args.data_source.split('_')[1]
    event_type=evt
    # method='ARIMA'
    ext_name = args.external_source.split('.')[0]
    external_signal_name=ext_name
    target=args.data_source.split('_')[0]
    results2=[target,evt,ext_name,s,e, ts_pred_rounded.sum(),mae, mae_round, rmse, rmse_round, mase, mase_round, nrmse,nrmse_round]

    with open('./output/model_stats.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(results2)
    

    
    #scoring code goes here:


    
    return


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
