#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import argparse
import os
import pandas as pd
import numpy as np

from model.arima import auto_arima

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def sliding_prediction_pyflux(ts_df, winsize=28):
    start_indx = 0
    predictions = []
    for start_indx in range(0, ts_df.size - winsize + 1):
        # for start_indx in range(0, 2):
        end_indx = start_indx + winsize
        ts_sliced_df = ts_df[start_indx: end_indx]
        min_aic, min_aic_param, model, nextday = auto_arima.auto_arima_pyflux(
            ts_sliced_df)
        predictions.append(nextday)
        # print("Min AIC =", min_aic)
        # print("Min AIC param =", min_aic_param)
        # print("Nextday prediction:")
        # print(nextday)
        print(nextday.index[0], nextday[[0]].values[0][0],
              ts_df.iloc[end_indx][0], min_aic, min_aic_param)
    pred_df = pd.concat(predictions)
    outfile = "data/isis_forecast_auto_w_" + str(winsize) + ".csv"
    print("Saving prediction:", outfile)
    pred_df.to_csv(outfile)


def sliding_prediction_fixed_arma(ts, winsize=28, show_convg_info=False):
    import pyflux as pf
    from statsmodels.tsa.arima_model import ARIMA

    ts = ts.astype(float)  # statsmodel
    # ts = pd.DataFrame(ts)  # pyflux

    start_indx = 0
    predictions = []

    for start_indx in range(0, ts.size - winsize):
        # for start_indx in range(0, 1):
        end_indx = start_indx + winsize
        ts_sliced = ts[start_indx: end_indx]
        date = ts.index[end_indx].date()

        # statsmodel
        # model = ARIMA(ts_sliced, (7, 1, 0))
        # model_fit = model.fit(disp=show_convg_info)
        # nextday_pred = model_fit.forecast(steps=1)

        # pyflux
        ts_sliced = pd.DataFrame(ts_sliced)
        model = pf.ARIMA(data=ts_sliced, ar=7, integ=0, ma=1)
        model_fit = model.fit("MLE")  # M-H
        nextday_pred = model.predict(h=1, intervals=True)
        pred_count = nextday_pred['count'][0]
        # print(model_fit)
        # print(nextday_pred['count'][0])
        print(nextday_pred['count'], date)
        predictions.append((date, pred_count))
    return pd.DataFrame(predictions)


def sliding_prediction_statsmodel(ts, winsize=28):
    ts = ts.astype(float)
    start_indx = 0
    predictions = []

    for start_indx in range(0, ts.size - winsize):
        # for start_indx in range(0, 1):
        end_indx = start_indx + winsize
        ts_sliced = ts[start_indx: end_indx]
        res = auto_arima.auto_arima_statsmodels(ts_sliced)

        date = ts.index[end_indx].date()
        if res is None:
            pred_count = np.nan
            print(date, pred_count)
        else:
            min_aic, min_aic_param, model, nextday = res
            pred_count = nextday[0][0]
            print(date, pred_count, min_aic, min_aic_param)

        predictions.append((date, pred_count))
        # print("Min AIC =", min_aic)
        # print("Min AIC param =", min_aic_param)
        # print("Nextday prediction:")
        # print(nextday[0][0])
        # print(date, pred_count, min_aic, min_aic_param)
    return pd.DataFrame(predictions)


def sliding_arima_simple(ts, ts_name, n_ar=7, n_d=0, n_ma=0,
                         train_set_prop=0.66, show_details=0):
    from statsmodels.tsa.arima_model import ARIMA
    X = ts.values
    size = int(len(X) * train_set_prop)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    arima_order = (n_ar, n_d, n_ma)
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=show_details)
        if show_details:
            print(model_fit.summary())
        output = model_fit.forecast()
        yhat = output[0][0]
        date = ts.index[size + t].date()
        predictions.append((date, yhat))
        obs = test[t]
        history.append(obs)
        print(date, 'predicted=%f, expected=%f' % (yhat, obs))
    # from sklearn.metrics import mean_squared_error
    # error = mean_squared_error(test, predictions)
    # print('Test MSE: %.3f' % error)
    # plot
    # import custom_plot
    # p = custom_plot.plot_true_pred(test, predictions)
    # pyplot.plot(test)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()
    # p.show()

    filename = ts_name + "_pred_simple_sm_w_" + str(size) + \
               "_p_" + str(arima_order[0]) + "_d_" + str(arima_order[1]) + \
               "_q_" + str(arima_order[2]) + ".csv"
    return pd.DataFrame(predictions), filename


def sliding_arima_simple_v2(ts, ts_name="ts", n_ar=7, n_d=0, n_ma=0,
                            train_set_prop=0.66, show_details=0):
    from statsmodels.tsa.arima_model import ARIMA
    X = ts.values
    size = int(len(X) * train_set_prop)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    arima_order = (n_ar, n_d, n_ma)
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=show_details)
        if show_details:
            print(model_fit.summary())
        output = model_fit.forecast()
        yhat = output[0][0]
        date = ts.index[size + t].date()
        predictions.append((date, yhat))
        obs = test[t]
        history.append(obs)
        history.pop(0)
        print(date, 'predicted=%f, expected=%f' % (yhat, obs))
    # from sklearn.metrics import mean_squared_error
    # error = mean_squared_error(test, predictions)
    # print('Test MSE: %.3f' % error)
    # plot
    # import custom_plot
    # p = custom_plot.plot_true_pred(test, predictions)
    # pyplot.plot(test)
    # pyplot.plot(predictions, color='red')
    # pyplot.show()
    # p.show()

    filename = ts_name + "_pred_simple_v2_sm_w_" + str(size) + \
               "_p_" + str(arima_order[0]) + "_d_" + str(arima_order[1]) + \
               "_q_" + str(arima_order[2]) + ".csv"
    return pd.DataFrame(predictions), filename

def valid_date(s):
    try:
        return pd.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


# def main(args):
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument('--package', default='stats-model',
#                         help='package info')
#     parser.add_argument('--data-dir', default="../data/usenix")
#     parser.add_argument('--out-dir', default="../output/isis")
#     # parser.add_argument('--out-dir', default="../output/usenix")
#     parser.add_argument(
#         '--ts-start-date', default="2016-07-01", type=valid_date)
#     parser.add_argument(
#         '--ts-end-date', default="2016-12-15", type=valid_date)
#     parser.add_argument('-w', '--window_size', default=28,
#                         type=int, help='Sliding window size')
#     parser.add_argument('-r', '--train-portion', default=0.30,
#                         type=float, help='Sliding window size')
#     parser.add_argument('-p', '--n-ar', default=1,
#                         type=int, help='Num of autoregressive terms')
#     parser.add_argument('-d', '--n-diff', default=1,
#                         type=int, help='Difference orders')
#     parser.add_argument('-q', '--n-ma', default=0,
#                         type=int, help='Num of moving avg terms')
#     options = parser.parse_args()
#
#     # ts = dp.get_ransom_locky()
#     # ts = dp.get_ransom_cerber()
#     ############################################################################
#     # load data
#     ############################################################################
#     # cerber domain registration
#     # filename = "gt_plus_likely_domains.csv"
#     filename = "gt_plus_likely_domains_2016-08-30_2017-01-18.csv"
#     filepath = os.path.join(options.data_dir, filename)
#
#     ts_cerber_dr = pd.read_csv(filepath, header=0, index_col=0,
#                                parse_dates=True, squeeze=True)
#
#     # cerber locky and cerber
#
#     filepath = os.path.join("../data/ransomware", "20170215.csv")
#     ts_locky = dp.get_ransom_series(datafile=filepath, malware_type="Locky",
#                                     start_date="2016-07-01",
#                                     end_date="2016-12-15")
#     ts_locky.name = 'count'
#     ts_locky.index.name = 'date'
#     ts_locky.to_csv("../data/ransomware/locky_20170215.csv", header=True)
#     ts_cerber = dp.get_ransom_series(datafile=filepath, malware_type="Cerber",
#                                      start_date="2016-07-01",
#                                      end_date="2016-12-15")
#     ts_cerber.name = 'count'
#     ts_cerber.index.name = 'date'
#     ts_cerber.to_csv("../data/ransomware/cerber_20170215.csv",header=True)
#
#     ts = ts_cerber.astype('float')
#     ts_name = "cerber"
#
#     ts = ts_locky.astype('float')
#     ts_name = "locky"
#
#     # ts = ts_cerber_dr.loc[options.ts_start_date:options.ts_end_date]
#     ts = ts_cerber_dr
#     ts_name = "cerber_dom_reg_set2"
#
#
#     if options.package == "stats-model":
#         print("Package =", options.package)
#         # pred_df = sliding_prediction_statsmodel(ts,
#         #                                         winsize=options.window_size)
#         # pred_df = sliding_prediction_fixed_arma(ts,
#         #   winsize=options.window_size)
#
#         # wit sliding arima simple
#         # input: ts, ts_name, n_ar=7, n_d=0, n_ma=0, train_set_prop=0.66,
#         # show_details=0
#         n_ar = options.n_ar
#         n_d = options.n_diff
#         n_ma = options.n_ma
#         pred_df, filename = sliding_arima_simple_v2(
#             ts, ts_name, n_ar=n_ar, n_d=n_d, n_ma=n_ma,
#             train_set_prop=options.train_portion)
#
#         pred_df.columns = ['date', 'count']
#         # filename = ts_name + "_pred_simple_sm_w_" + \
#         #     str(options.window_size) + ".csv"
#         outfile = os.path.join(options.out_dir, filename)
#         print("Saving:", outfile)
#         pred_df.to_csv(outfile, index=False)
#
#     elif options.package == "pyflux":
#         print("Package =", options.package)
#         # ts_df = pd.DataFrame(ts)
#         # sliding_prediction_pyflux(ts_df, winsize=wsize)
#     else:
#         pass
#
#
# if __name__ == "__main__":
#     sys.exit(main(sys.argv))
