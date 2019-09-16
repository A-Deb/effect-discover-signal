#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")

# our modules
from utils import custom_plot
from utils import dataprocessing as dp
from utils import warning
from utils import model_wrapper
from utils import model_output
# from model.arima import auto_arima
from model.arima import arimax

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def learn_arimax_parameters(ts, ts_exogs, options, look_ahead=1,
                            train_start_date=None,
                            train_end_date=None, test_start_date=None,
                            save_plots=False):
    if train_start_date is None:
        if ts_exogs is None:
            train_start_date = ts.index.min()
        else:
            train_start_date = max(ts.index.min(), ts_exogs.index.min())

    if train_end_date is None:
        train_end_date = max(ts.index)

    test_end_date = options.warn_start_date + pd.Timedelta(days=look_ahead - 1)

    gap_dates = pd.date_range(train_end_date + pd.Timedelta(days=1),
                              test_start_date, closed="left")

    test_dates = pd.date_range(test_start_date, test_end_date)

    gap_and_test_dates = gap_dates.append(test_dates)

    ts_train = ts[train_start_date: train_end_date]

    if ts_exogs is not None:
        ts_exogs_train = ts_exogs[train_start_date: train_end_date]
        ts_exogs_gap_test = ts_exogs[
                            min(gap_and_test_dates):max(gap_and_test_dates)]
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

    if save_plots:
        outdir = os.path.join("output", "arimax")
        os.makedirs(outdir, exist_ok=True)
        filepath = os.path.join(outdir, options.data_source + "_fitted_ts.png")
        save_ts_plots(ts_train, min_aic_fit_res.fittedvalues, ts_exogs_train,
                      ts_true_name="observed", ts_pred_name="fitted",
                      xlabel="Day", ylabel="#Events",
                      filepath=filepath)
        filepath = os.path.join(outdir,
                                options.data_source + "_predicted_ts.png")
        ts_gap_test = ts[min(gap_and_test_dates):max(gap_and_test_dates)]
        ts_pred_gap_test = pd.Series(forecast_res[0],
                                     index=pd.date_range(
                                         min(gap_and_test_dates),
                                         max(gap_and_test_dates)))
        save_ts_plots(ts, ts_pred_gap_test, ts_exogs,
                      ts_true_name="observed", ts_pred_name="predicted",
                      xlabel="Day", ylabel="#Events",
                      filepath=filepath)

    print("Last", look_ahead, "predictions:")
    print(list_pred)
    ts_pred = pd.Series(list_pred, index=test_dates + pd.Timedelta(hours=12))
    ts_pred[ts_pred < 0] = 0
    ts_pred.name = 'count'
    ts_pred.index.name = 'date'

    return ts_pred, min_aic_fit_order


def save_ts_plots(ts_true, ts_pred, ts_exog, ts_true_name="observed",
                  ts_pred_name="predicted", ts_exog_name="external",
                  xlabel="Day", ylabel="#Events", show_external=False,
                  filepath="output/arimax/sample_plot.png"):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # print(ts_true)
    # print(ts_pred)
    model_colors = ['k', 'b', 'r', '#8E44AD', '#F39C12', 'g', '#3498DB']
    ax = plt.subplot(111)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.plot(ts_true, '-', label=ts_true_name, color=model_colors[0],
             linewidth=2.5)
    plt.plot(ts_pred, '-', label=ts_pred_name, color=model_colors[1],
             linewidth=2.5)
    if show_external and ts_exog is not None:
        plt.plot(ts_exog, '-', label=ts_exog_name, color=model_colors[2],
                 linewidth=2.5)

    plt.gcf().set_size_inches(15, 8)
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.legend(loc='best', prop={'size': 18},
               edgecolor="0.9", facecolor="0.9",
               bbox_to_anchor=(1.0, 0.5))
    print("Saving:", filepath)
    plt.savefig(filepath, format='png', bbox_inches='tight')
    plt.clf()


def write_model_metadata(options, es_query, arima_order):
    model_metadata = model_wrapper.modelWrapper()

    model_metadata.id = "ARIMAX_p_" + str(arima_order[0]) + "_d_" + \
                        str(arima_order[1]) + "_q_" + str(arima_order[2]) + \
                        "_" + options.data_source

    model_metadata.model_name = "ARIMAX"
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
    model_metadata.model_description = "This is ARIMAX model, " + \
                                       "which predicts the expected number of cyber attacks for the next " + \
                                       "day given a time series of attacks and external signals."
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
    parser = argparse.ArgumentParser(description=__doc__)

    datagrp = parser.add_argument_group("[*] Data Source Options")
    datagrp.add_argument(
        '-d', '--data-source',
        default='armstrong_endpoint-malware',
        help='Data soucres for learning the daywise baserate model. \
        Possible options: \
        dexter_endpoint-malware and \
        armstrong_endpoint-malware.\
        Default value is dexter_endpoint-malware.')
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
        help='Possible values: hg-abusech.')
    datagrp.add_argument(
        '--data-replication',
        default=False, type=bool,
        help='Replication of warning')

    arimaxgrp = parser.add_argument_group("[*] ARIMAX Parameter Options")
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
        endpoint-malware.')
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
    # load GT data
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
        dirpath = "output/arimax"
        os.makedirs(dirpath, exist_ok=True)
        filename = "time_series_" + options.data_source + "_" + \
                   learn_end_date.strftime("%Y-%m-%d") + \
                   "_window_" + str(options.window_size) + "_days.pdf"
        filepath = os.path.join(dirpath, filename)
        print("Saving plot:", filepath)
        p1.savefig(filepath, format="pdf")

    #######################################################################
    # load external data
    #######################################################################

    if options.external_source is not None:
        print("Loading external signals.")

        ex_index = "effect"
        if options.data_replication:
            ex_index = ex_index + '-' + (
                options.warn_start_date +
                pd.Timedelta(days=-1)).strftime("%Y%m%d") + '/malware'
            print("Replication index: ", ex_index)
        else:
            ex_index = 'effect/malware'
        print("\tindex: ", ex_index)
        ts_external = dp.get_ts_malware(publisher=options.external_source,
                                        hostname=options.data_hostname,
                                        index=ex_index)
        # ts_external = dp.get_ts_malware_exclue_Locky(
        #     publisher=options.external_source, hostname=options.data_hostname,
        #     index=ex_index)
        min_date = ts_external.index.min()
        max_date = ts_external.index.max()
        ts_external = ts_external.reindex(
            index=pd.date_range(min_date, max_date),
            fill_value=0)

        if options.data_source == "dexter_endpoint-malware":
            lag = 29
        elif options.data_source == "armstrong_endpoint-malware":
            lag = 12
        elif options.data_source == "knox_endpoint-malware":
            lag = 25
        else:
            print("Not a valid data source.")
            exit(1)

        ts_external_shifted = ts_external.shift(lag, freq=1)
        ts_external_shifted = ts_external_shifted[learn_start_date:]
        # ts_exog_end_date = ts_absusech_shifted.index.max().date()

        print("External Data:")
        print("\tStart date:", ts_external_shifted.index.min())
        print("\tEnd date:", ts_external_shifted.index.max())
        # time_now = pd.datetime.now().date()
        #
        # if time_now >= ts_exog_end_date:
        #     raise ValueError(
        #         "Lack of external signals for out-of-sample predictions.")
        # print("end date of external ts=", ts_exog_end_date)
        # input("Press a key")
    else:
        ts_external_shifted = None
    #######################################################################
    # learn ARIMAX model
    #######################################################################
    # ts_pred, arima_order = learn_rare_parameters(ts, ts_absusech_shifted,
    #                                               options)
    ts_pred, arimax_order = learn_arimax_parameters(
        ts, ts_external_shifted, options, look_ahead=options.look_ahead,
        test_start_date=options.warn_start_date)

    print("Forecast:")
    print(ts_pred)
    print("Arimax order =", arimax_order)
    ts_pred = np.around(ts_pred[-1:])
    print("Rounded prediction =", ts_pred)

    if ts_pred.sum() > 0:
        # generate model metadata
        model_id = write_model_metadata(options, es_query, arimax_order)

        # generate warnings
        if options.warn_version == 1:
            model_output.write_model_output(options, ts_pred, model_id,
                                            output_name="arimax_warning")
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
