#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd

# our modules
from utils import custom_plot
from utils import dataprocessing as dp
from utils import warning
from utils import model_wrapper
from utils import model_output
# from model.arima import auto_arima
from model.arima import arima

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def learn_arima_parameters(ts, options, train_start_date=None,
                           train_end_date=None, test_start_date=None):
    if train_start_date is None:
        train_start_date = min(ts.index)
    if train_end_date is None:
        train_end_date = max(ts.index)

    if test_start_date is None:
        test_start_date = pd.datetime.now().date() + pd.Timedelta(days=1)
    test_end_date = test_start_date + pd.Timedelta(days=options.look_ahead - 1)

    gap_dates = pd.date_range(train_end_date + pd.Timedelta(days=1),
                              test_start_date, closed="left")
    test_dates = pd.date_range(test_start_date, test_end_date)

    ts_train = ts[train_start_date: train_end_date]

    # print("Learning ARIMA orders:")
    # ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
    #     auto_arima.iterative_ARIMA_fit(ts_train, options.max_p,
    #                                    options.max_d,
    #                                    options.max_q)
    arima_instance = arima.ARIMA()
    min_aic_fit_res, min_aic_fit_order = arima_instance.fit2(ts_train,
                                                             options.max_p,
                                                             options.max_d,
                                                             options.max_q)
    print("Selected Model Order=", min_aic_fit_order)
    # print("AIC score for selected model=", min_aic)

    forecast_length = len(gap_dates) + len(test_dates)
    forecast_res = min_aic_fit_res.forecast(steps=forecast_length)
    list_pred = forecast_res[0][-len(test_dates):]

    print("prediction =", list_pred)
    ts_pred = pd.Series(list_pred, index=test_dates + pd.Timedelta(hours=12))
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
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    datagrp = parser.add_argument_group("[*] Data Source Options")
    datagrp.add_argument(
        '-d', '--data-source',
        default='dexter_endpoint-malware',
        help='Data soucres for learning the daywise baserate model. \
        Possible options: \
        ransomware_locky, \
        ransomware_cerber, \
        dexter_endpoint-malware,\
        dexter_malicious-email,\
        dexter_malicious-url, \
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
        '--data-replication',
        default=False, type=bool,
        help='Replication of warning')

    arimagrp = parser.add_argument_group("[*] ARIMA Parameter Options")
    arimagrp.add_argument(
        '--max-p',
        default=7, type=int,
        help='Number of autoregressive terms. Default value is 2.')
    arimagrp.add_argument(
        '--max-d',
        default=2, type=int,
        help='Number of moving average terms. Default value is 1.')
    arimagrp.add_argument(
        '--max-q',
        default=7, type=int,
        help='Number of moving average terms. Default value is 0.')

    # warning group
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

    parser.add_argument(
        '--feature-dist-file', type=str, default=None)

    warngrp.add_argument(
        '--warn-version',
        default=2,
        help='Warning format version')

    othergrp = parser.add_argument_group("[*] Other Options")
    othergrp.add_argument(
        '--show-plot',
        default=False, type=bool,
        help='Draw plots')

    options = parser.parse_args()

    #######################################################################
    # load data
    #######################################################################
    ts, es_query = dp.gt_time_series(options)
    if ts is None:
        # print("Error: time series is not loaded properly")
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

    if options.show_plot:
        p1 = custom_plot.plot_ts(ts)
        dirpath = "output/arima"
        os.makedirs(dirpath, exist_ok=True)
        filename = "time_series_" + options.data_source + "_" + \
                   learn_end_date.strftime("%Y-%m-%d") + \
                   "_window_" + str(options.window_size) + "_days.pdf"
        filepath = os.path.join(dirpath, filename)
        print("Saving plot:", filepath)
        p1.savefig(filepath, format="pdf")

    ts_pred, arima_order = learn_arima_parameters(
        ts, options, test_start_date=options.warn_start_date)
    print("Forecast:")
    print(ts_pred)
    print(arima_order)
    ts_pred = np.around(ts_pred[-1:])
    print(ts_pred)

    if ts_pred.sum() > 0:
        # generate model metadata
        model_id = write_model_metadata(options, es_query, arima_order)

        # generate warnings
        if options.warn_version == 1:
            model_output.write_model_output(options, ts_pred, model_id,
                                            output_name="arima_warning")
        else:
            company = model_output.get_target_orgnization(options)
            if company == "knox":
                model_output.write_model_output_v3(
                    options, ts_pred, model_id,
                    output_name="daywise_baserate_warning")
            else:
                model_output.write_model_output_v2(
                    options, ts_pred, model_id,
                    output_name="daywise_baserate_warning")
    else:
        print("Model did not predict any attacks. There will be no warnings.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
