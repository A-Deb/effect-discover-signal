#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import argparse
import warnings

# our modules
from utils import custom_plot
from utils import dataprocessing as dp
from utils import warning
from utils import model_wrapper
from utils import model_output
# from model.arima import auto_arima
from model.arima import arimax
from util.measures import measures

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

warnings.filterwarnings("ignore")
def learn_arimax_parameters(ts, ts_exogs, options, train_start_date=None,
                            train_end_date=None, test_start_date=None,
                            test_end_date=None):
    if train_start_date is None:
        train_start_date = min(ts.index)

    if train_end_date is None:
        train_end_date = max(ts.index)

    if test_start_date is None:
        test_start_date = pd.datetime.now().date() + pd.Timedelta(days=1)

    if test_end_date is None:
        test_end_date = options.warn_start_date + pd.Timedelta(
            days=options.look_ahead - 1)

    # test_end_date = test_start_date + pd.Timedelta(days=options.look_ahead - 1)

    gap_start_date = train_end_date + pd.Timedelta(days=1)
    gap_dates = pd.date_range(train_end_date + pd.Timedelta(days=1),
                              test_start_date, closed="left")

    test_dates = pd.date_range(test_start_date, test_end_date)
    ts_train = ts[train_start_date: train_end_date]

    if ts_exogs is not None:
        ts_exogs_train = ts_exogs[train_start_date: train_end_date]
        ts_exogs_gap_test = ts_exogs[gap_start_date:test_end_date]
    else:
        ts_exogs_train = None
        ts_exogs_gap_test = None

    arimax_instance = arimax.ARIMAX()
    min_aic_fit_res, min_aic_fit_order = arimax_instance.fit2(ts_train,
                                                              ts_exogs_train,
                                                              options.max_p,
                                                              options.max_d,
                                                              options.max_q,
                                                              trend='c')
    print("Selected Model Order=", min_aic_fit_order)
    # print("AR params:", min_aic_fit_res.arparams())
    # print("AIC score for selected model=", min_aic)
    print("Number of AR terms =", min_aic_fit_res.k_ar)
    print("Number of exog terms =", min_aic_fit_res.k_exog)
    print("Number of MA terms =", min_aic_fit_res.k_ma)
    print("Trend term included =", min_aic_fit_res.k_trend)
    print("Parameters =", min_aic_fit_res.params)

    forecast_length = len(gap_dates) + len(test_dates)
    forecast_res = min_aic_fit_res.forecast(steps=forecast_length,
                                            exog=ts_exogs_gap_test)
    list_pred = forecast_res[0][-len(test_dates):]
    print("predictions")
    print(list_pred)
    ts_pred = pd.Series(list_pred, index=test_dates)
    ts_pred[ts_pred < 0] = 0
    ts_pred.name = 'count'
    ts_pred.index.name = 'date'

    return ts_pred, min_aic_fit_order


def write_model_metadata(options, es_query, arima_order):
    model_metadata = model_wrapper.modelWrapper()

    model_metadata.id = "ARIMA_p_" + str(arima_order[0]) + "_d_" + \
                        str(arima_order[1]) + "_q_" + str(arima_order[2]) + \
                        "_" + options.data_source

    model_metadata.model_name = "ARIMA"
    model_metadata.model_version = "1.0"
    model_metadata.model_type = "forecast_model"
    model_metadata.repository = \
        'https://github.com/usc-isi-i2/effect-forecasting-models/tree/master/model/arima'
    model_metadata.author = "USC-ISI"
    model_metadata.model_input = list()
    model_metadata.model_input.append(es_query)
    model_metadata.parameters = list()
    model_metadata.parameters.append(("p", arima_order[0]))
    model_metadata.parameters.append(("d", arima_order[1]))
    model_metadata.parameters.append(("q", arima_order[2]))
    model_metadata.parameters.append(("max_p", options.max_p))
    model_metadata.parameters.append(("max_d", options.max_d))
    model_metadata.parameters.append(("max_q", options.max_q))
    model_metadata.parameters.append(("look_ahead", options.look_ahead))
    model_metadata.model_description = "This is ARIMA model, " + \
                                       "which predicts the expected number of cyber attacks for the next " + \
                                       "day given a time series of attacks till today."
    model_metadata.templated_narrative_hypothesis = None
    model_metadata.templated_narrative_context = None

    filename = model_metadata.id + ".json"
    os.makedirs(options.model_metadata_dir, exist_ok=True)
    model_matadata_fpath = os.path.join(options.model_metadata_dir, filename)
    model_metadata.save(model_matadata_fpath)
    return model_metadata.id


def valid_date(s):
    try:
        return pd.datetime.strptime(s, "%Y%m%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def main(args):
    # <editor-fold desc="Arguments">
    parser = argparse.ArgumentParser(description=__doc__)
    # <editor-fold desc="Meta data">
    datagrp = parser.add_argument_group("[*] Data Source Options")
    datagrp.add_argument(
        '-d', '--data-source',
        default='armstrong_endpoint-malware',
        help='Data soucres for learning the daywise baserate model. \
        Possible options: \
        ransomware_locky, \
        ransomware_cerber, \
        dexter_endpoint-malware,\
        dexter_malicious-email,\
        dexter_malicious-destination, \
        armstrong_endpoint-malware,\
        and armstrong_malicious-email.\
        Default value is ransomware_cerber, which denotes ransomware malware \
        with Cerber types.')
    datagrp.add_argument(
        '-w', '--window-size',
        default=100, type=int,
        help='Sliding window size. Default = 50 days.')
    datagrp.add_argument(
        '--data-hostname',
        default='http://cloudweb01.isi.edu/es/',
        help='Hostname for ES')
    datagrp.add_argument(
        '--external-source',
        default=None,
        help='Ransomware attacks reported on abuse.ch.')
    datagrp.add_argument(
        '--data-replication',
        default=False, type=bool,
        help='Replication of warning')

    # </editor-fold>

    # <editor-fold desc="Model parameters">
    arimaxgrp = parser.add_argument_group("[*] ARIMA Parameter Options")
    arimaxgrp.add_argument(
        '--max-p',
        default=7, type=int,
        help='Number of autoregressive terms. Default value is 2.')
    arimaxgrp.add_argument(
        '--max-d',
        default=2, type=int,
        help='Number of moving average terms. Default value is 1.')
    arimaxgrp.add_argument(
        '--max-q',
        default=7, type=int,
        help='Number of moving average terms. Default value is 0.')
    # </editor-fold>

    # <editor-fold desc="warning parameters">
    warngrp = parser.add_argument_group("[*] Warning Generation Options")
    warngrp.add_argument(
        '--warn-start-date',
        default=pd.datetime.now().date() + pd.Timedelta(days=1),
        help='Warning start date. If None, tomorrow will\
        be the warning start date.', type=valid_date)
    warngrp.add_argument(
        '-l', '--look-ahead',
        default=7, type=int,
        help='Number of days to be forecasted without GT data. Default value \
        is 1.')
    warngrp.add_argument(
        '--event-type-broad',
        default='malware',
        help='Generic class for event types.')
    warngrp.add_argument(
        '--event-type',
        default='endpoint-malware',
        help='Generate warnings for event-type. Possible options: \
        endpoint-malware, malicious-destination, malicious-email.')
    warngrp.add_argument(
        '--warning-dir',
        default='warnings',
        help='Save json warning into this directory.')

    warngrp.add_argument(
        '--model-metadata-dir',
        default='model_metadata/baserate',
        help='Save model metadata in json format into this directory.')

    warngrp.add_argument(
        '--sensor',
        default='ransomwaretracker.abuse.ch',
        help='Sensor field for the warning.')

    warngrp.add_argument(
        '--warn-version',
        default=1,
        help='Warning format version')
    # </editor-fold>

    # optional group
    othergrp = parser.add_argument_group("[*] Other Options")
    othergrp.add_argument(
        '--model-name',
        default="ARIMAX",
        help='Model Name')
    othergrp.add_argument(
        '--train-proportion',
        default=0.6, type=float,
        help='Model Name')
    othergrp.add_argument(
        '-p', '--show-plot',
        default=True, type=bool,
        help='Draw plots')

    options = parser.parse_args()

    # </editor-fold>
    #######################################################################
    # load GT data
    #######################################################################
    ts, es_query = dp.gt_time_series(options)
    if ts is None:
        sys.exit("Error: data is not loaded properly.")
    learn_start_date = min(ts.index)
    learn_end_date = max(ts.index)

    print("Training Data:")
    print("\tStart date:", learn_start_date)
    print("\tEnd date:", learn_end_date)
    print("\tNumber of days:", ts.shape[0])
    print("\tNumber of data points:", ts.sum())
    print("\tMax num of events occurred in a day:", max(ts))
    print("\tMin num of events occurred in a day:", min(ts))
    print("\tWarning start date:", options.warn_start_date)
    print("\tForecasting look ahead days:", options.look_ahead)

    train_proportion = options.train_proportion

    dirpath = os.path.join(
        "output", options.data_source.lower(), options.model_name.lower())
    os.makedirs(dirpath, exist_ok=True)

    if options.show_plot:
        p1 = custom_plot.plot_ts(ts)
        filename = "Time_Series_" + options.data_source + "_" + \
                   learn_end_date.strftime("%Y-%m-%d") + \
                   "_window_" + str(options.window_size) + "_days.pdf"
        filepath = os.path.join(dirpath, filename)
        print("Saving plot:", filepath)
        p1.savefig(filepath, format="pdf")

    #######################################################################
    # load external data
    #######################################################################
    if options.data_source == "dexter_endpoint-malware":
        lag = 29
        train_start_date = pd.to_datetime("2016-04-05").date()
        train_end_date = pd.to_datetime("2016-06-23").date()
        test_start_date = pd.to_datetime("2016-06-24").date()
        test_end_date = pd.to_datetime("2016-07-26").date()
    elif options.data_source == "armstrong_endpoint-malware":
        lag = 8
        train_start_date = pd.to_datetime("2016-04-06").date()
        train_end_date = pd.to_datetime("2016-08-23").date()
        test_start_date = pd.to_datetime("2016-08-24").date()
        test_end_date = pd.to_datetime("2016-09-27").date()

    if options.external_source is not None:
        print("Loading external signals.")
        # ts_absusech = dp.get_ts_malware(publisher="hg-abusech")
        #ts_absusech = dp.get_ts_malware_exclue_Locky(publisher="hg-abusech")
        ts_absusech=pd.read_csv('/Users/ashokdeb/timeseries/forum6_SentiStrength.csv' )
        min_date = ts_absusech.index.min()
        max_date = ts_absusech.index.max()
        ts_absusech = ts_absusech.reindex(
            index=pd.date_range(min_date, max_date),
            fill_value=0)

        ts_external_shifted = ts_absusech.shift(lag, freq=1)
        ts_external_shifted = ts_external_shifted[learn_start_date:]
        # ts_exog_end_date = ts_absusech_shifted.index.max().date()
        # time_now = pd.datetime.now().date()
    else:
        ts_external_shifted = None

    # ts_train = ts[train_start_date:train_end_date]
    #######################################################################
    # learn ARIMAX model
    #######################################################################
    # pred = []
    # for test_sd in pd.date_range(test_start_date, test_end_date):
    #     ts_pred, arima_order = learn_rare_parameters(
    #         ts, ts_absusech_shifted, options,
    #         train_start_date=train_start_date,
    #         train_end_date=test_sd - pd.Timedelta(days=1),
    #         test_start_date=test_sd)
    #     print("Forecast:")
    #     print(ts_pred)
    #     print(arima_order)
    #     ts_pred = np.around(ts_pred)
    #     print(ts_pred)
    #     pred.append(ts_pred)

    ts_pred, arima_order = learn_arimax_parameters(
        ts, ts_external_shifted, options,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date)

    # pred_ts = pd.concat(pred)
    pred_ts = ts_pred
    pred_ts.name = 'count'
    pred_ts.index.name = 'date'
    if options.external_source is not None:
        filename = options.data_source + "_pred.csv"
    else:
        filename = options.data_source + "_wo_external_pred.csv"
    filepath = os.path.join(dirpath, filename)
    pred_ts.to_csv(filepath, header=True)

    s, e = pred_ts.index.min(), pred_ts.index.max()
    print(s)
    print(e)
    mae = measures.get_mae(ts[s:e], pred_ts)
    rmse = measures.get_rmse(ts[s:e], pred_ts)
    mase = measures.get_mase(ts[s:e], pred_ts)

    score_df = pd.DataFrame([[options.model_name, mae, rmse, mase]],
                            columns=["Method", "MAE", "RMSE", "MASE"])
    print(score_df)
    if options.external_source is not None:
        filename = options.data_source + "_score.csv"
    else:
        filename = options.data_source + "_wo_external_score.csv"
    filepath = os.path.join(dirpath, filename)
    print("Saving:", filepath)
    score_df.to_csv(filepath, header=True, index=False)

    from plotting import tsplot

    if options.external_source is not None:
        filename = options.data_source + "_pred.png"
    else:
        filename = options.data_source + "_wo_external_pred.png"
    filepath = os.path.join(dirpath, filename)
    plt = tsplot.plot_list_of_lines([ts, pred_ts])
    # filepath = os.path.join(dirpath, options.data_source + "_pred.png")
    plt.savefig(filepath)
    plt.clf()

    if options.external_source is not None:
        filename = options.data_source + "_pred_test.png"
    else:
        filename = options.data_source + "_wo_external_pred_test.png"
    filepath = os.path.join(dirpath, filename)
    plt2 = tsplot.plot_list_of_lines([ts[s:e], pred_ts])
    plt2.savefig(filepath)
    plt.clf()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
