#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
import pandas as pd
import argparse

# our modules
from utils import custom_plot
from utils import dataprocessing as dp
from utils import model_wrapper
from utils import model_output
# from model.arima import auto_arima

import _pickle as cPickle
import pandas as pd
import numpy as np
# from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.api import VAR
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, GRU
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D
from time import time
import datetime
import pytz
import pdb
from itertools import product

from utils.sensor_mapping import SensorMapping
import lin_corr

__author__ = "Palash Goyal"
__email__ = "goyal@isi.edu"


def construct_rnn_model(look_back,
                        d,
                        n_units=[20, 20],#[500, 500],
                        dense_units=[50, 10],#[1000, 200, 50, 10],
                        filters=64,
                        kernel_size=5,
                        pool_size=4,
                        method='sgru',
                        bias_reg=None,
                        input_reg=None,
                        recurr_reg=None):
    model = Sequential()
    if method == 'lstm':
        model.add(LSTM(n_units[0],
                       input_shape=(look_back, d),
                       return_sequences=True,
                       bias_regularizer=bias_reg,
                       kernel_regularizer=input_reg,
                       recurrent_regularizer=recurr_reg))
        for n_unit in n_units[1:]:
            model.add(LSTM(n_unit,
                           bias_regularizer=bias_reg,
                           kernel_regularizer=input_reg,
                           recurrent_regularizer=recurr_reg))
    elif method == 'gru':
        model.add(GRU(n_units[0],
                      input_shape=(look_back, d),
                      return_sequences=True,
                      bias_regularizer=bias_reg,
                      kernel_regularizer=input_reg,
                      recurrent_regularizer=recurr_reg))
        for n_unit in n_units[1:]:
            model.add(GRU(n_unit,
                          bias_regularizer=bias_reg,
                          kernel_regularizer=input_reg,
                          recurrent_regularizer=recurr_reg))
    elif method == 'sgru':
        model.add(sGRU(n_units[0],
                       input_shape=(look_back, d),
                       return_sequences=True))
        for n_unit in n_units[1:]:
            model.add(sGRU(n_unit))
    elif method == 'plstm':
        model.add(PLSTM(n_units[0],
                        input_shape=(look_back, d),
                        return_sequences=True))
        for n_unit in n_units[1:]:
            model.add(PLSTM(n_unit))
    elif method == 'bi-lstm':
        model.add(Bidirectional(LSTM(n_units[0],
                                input_shape=(look_back, d),
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(Bidirectional(LSTM(n_unit)))
    elif method == 'bi-gru':
        model.add(Bidirectional(GRU(n_units[0],
                                input_shape=(look_back, d),
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(Bidirectional(GRU(n_unit)))
    elif method == 'bi-sgru':
        model.add(Bidirectional(sGRU(n_units[0],
                                input_shape=(look_back, d),
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(Bidirectional(sGRU(n_unit)))
    elif method == 'bi-plstm':
        model.add(Bidirectional(PLSTM(n_units[0],
                                input_shape=(look_back, d),
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(Bidirectional(PLSTM(n_unit)))
    elif method == 'lstm-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(n_units[0],
                       return_sequences=True))
        for n_unit in n_units[1:]:
            model.add(LSTM(n_unit))
    elif method == 'gru-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(GRU(n_units[0],
                      return_sequences=True))
        for n_unit in n_units[1:]:
            model.add(GRU(n_unit))
    elif method == 'sgru-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(sGRU(n_units[0],
                       return_sequences=True))
        for n_unit in n_units[1:]:
            model.add(sGRU(n_unit))
    elif method == 'plstm-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(PLSTM(n_units[0],
                        return_sequences=True))
        for n_unit in n_units[1:]:
            model.add(PLSTM(n_unit))
    elif method == 'bi-lstm-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Bidirectional(LSTM(n_units[0],
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(LSTM(n_unit))
    elif method == 'bi-gru-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Bidirectional(GRU(n_units[0],
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(GRU(n_unit))
    elif method == 'bi-sgru-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Bidirectional(sGRU(n_units[0],
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(Bidirectional(sGRU(n_unit)))
    elif method == 'bi-plstm-cnn':
        model.add(Conv1D(filters,
                  kernel_size,
                  input_shape=(look_back, d),
                  padding='valid',
                  activation='relu',
                  strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Bidirectional(PLSTM(n_units[0],
                                return_sequences=True)))
        for n_unit in n_units[1:]:
            model.add(Bidirectional(PLSTM(n_unit)))
    for dense_n_unit in dense_units:
        model.add(Dense(dense_n_unit, activation='relu'))
    model.add(Dense(d + 1))
    if 'plstm' in method:
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss='mean_squared_error', optimizer=adam)
    else:
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# convert an array of values into a dataset matrix
def create_training_samples(ts_train, look_back=5, d=2):
    T = len(ts_train)
    train_size = T - look_back
    trainX = np.zeros((train_size, look_back, d - 1))
    trainY = np.zeros((train_size, d))
    n_samples_train = 0
    for t in range(T - look_back):
        for tau in range(look_back):
            trainX[n_samples_train, tau, :] = ts_train.iloc[t + tau, 1:]
        trainY[n_samples_train, :] = ts_train.iloc[t + look_back, :]
        n_samples_train += 1
    return trainX, trainY


def learn_rnn_parameters(ts, ts_exogs, options, look_ahead=1,
                         train_start_date=None,
                         train_end_date=None, test_start_date=None,
                         save_plots=False, method='gru'):
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

    if ts_exogs_train is not None:
        ts_concat = pd.concat([ts_train, ts_exogs_train], axis=1)
    else:
      ts_concat = pd.DataFrame(ts_train)
    ts_concat = ts_concat.dropna(axis=0)
    look_back = 5
    d = len(ts_concat.columns)
    model = construct_rnn_model(look_back=look_back, d=d - 1, method=method)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    trainX, trainY = create_training_samples(
        ts_concat,
        look_back=look_back,
        d=d
    )
    t1 = time()
    model.fit(trainX,
              trainY,
              nb_epoch=2000,
              batch_size=100,
              validation_split=0.2,
              callbacks=[early_stop],
              verbose=2)
    t2 = time()
    print('Training time: %fsec' % (t2 - t1))
    forecast_length = len(gap_dates) + len(test_dates)
    predictions = []
    for pred_day in range(forecast_length):
        try:
          testX = np.array(ts_concat[-look_back:])[:, 1:].reshape((1, look_back, d - 1))
        except:
          pdb.set_trace()
        prediction = model.predict(testX, batch_size=100, verbose=0)
        ts_concat = ts_concat.append(pd.DataFrame(prediction, columns=ts_concat.columns))
        predictions.append(prediction[0])
    print('Test time: %fsec' % (time() - t2))
    list_pred = np.array(predictions)[-len(test_dates):, 0]

    # if save_plots:
    #     outdir = os.path.join("output", "arimax")
    #     os.makedirs(outdir, exist_ok=True)
    #     filepath = os.path.join(outdir, options.data_source + "_fitted_ts.png")
    #     save_ts_plots(ts_train, min_aic_fit_res.fittedvalues, ts_exogs_train,
    #                   ts_true_name="observed", ts_pred_name="fitted",
    #                   xlabel="Day", ylabel="#Events",
    #                   filepath=filepath)
    #     filepath = os.path.join(outdir,
    #                             options.data_source + "_predicted_ts.png")
    #     ts_gap_test = ts[min(gap_and_test_dates):max(gap_and_test_dates)]
    #     ts_pred_gap_test = pd.Series(forecast_res[0],
    #                                  index=pd.date_range(
    #                                      min(gap_and_test_dates),
    #                                      max(gap_and_test_dates)))
    #     save_ts_plots(ts, ts_pred_gap_test, ts_exogs,
    #                   ts_true_name="observed", ts_pred_name="predicted",
    #                   xlabel="Day", ylabel="#Events",
    #                   filepath=filepath)

    print("Last", look_ahead, "predictions:")
    print(list_pred)
    ts_pred = pd.Series(list_pred, index=test_dates + pd.Timedelta(hours=12))
    ts_pred[ts_pred < 0] = 0
    ts_pred.name = 'count'
    ts_pred.index.name = 'date'

    return ts_pred


def save_ts_plots(ts_true, ts_pred, ts_exog, ts_true_name="observed",
                  ts_pred_name="predicted", ts_exog_name="external",
                  xlabel="Day", ylabel="#Events", show_external=False,
                  filepath="output/rnn_only_es/sample_plot.png"):
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


def write_model_metadata(options, es_query):
    model_metadata = model_wrapper.modelWrapper()

    model_metadata.id = "RNN" + \
                        "_" + options.data_source + "_" + options.external_source + "_" + options.external_keyword


    model_metadata.model_name = "RNN"
    model_metadata.model_version = "1.0"
    model_metadata.model_type = "forecast_model"
    model_metadata.repository = \
        'https://github.com/usc-isi-i2/effect-forecasting-models/tree/master/model/rnn'
    model_metadata.author = "USC-ISI"
    model_metadata.model_input = list()
    model_metadata.model_input.append(es_query)
    model_metadata.parameters = list()
    model_metadata.parameters.append(("look_ahead", options.look_ahead))
    model_metadata.model_description = "This is RNN model, " + \
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
        help='Source possible values: blogs, twitter, d2web, and vulnerability.')
    datagrp.add_argument(
        '--external-keyword',
        default=None,
        help='Source keyword values. Examples: account, cve, dns.')
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
        '--train-start-date',
        help='Warning start date', default=None)
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
    
    parser.add_argument("-a", "--ablate", help="comma separated list of sources to ablate, eg: hg-abusech,gt,hg-cve",
                        required=False)
    parser.add_argument("-b", "--ablateSensor",
                        help="comma separated list of sensors to ablate. eg: raptor,dark-embed,company-vulnerability",
                        required=False)

    options = parser.parse_args()
    sources_ablated = SensorMapping.get_ablated_sources(options)
    sensors_ablated = SensorMapping.get_ablated_sensors(options)
    
    #######################################################################
    # reproducible results - set seeds and single CPU thread for session
    #######################################################################
    warn_timestamp = options.warn_start_date
    warn_date = datetime.datetime(warn_timestamp.year, warn_timestamp.month, warn_timestamp.day)
    pacific = pytz.timezone('US/Pacific')     # seed always using same timezone
    seed = int(pacific.localize(warn_date).timestamp()) 
    
    np.random.seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(seed)
    rn.seed(seed)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

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
        dirpath = "output/rnn_only_es"
        os.makedirs(dirpath, exist_ok=True)
        filename = "time_series_" + options.data_source + "_" + \
                   learn_end_date.strftime("%Y-%m-%d") + \
                   "_window_" + str(options.window_size) + "_days.pdf"
        filepath = os.path.join(dirpath, filename)
        print("Saving plot:", filepath)
        p1.savefig(filepath, format="pdf")
        
        
    if options.train_start_date is None:
        train_start_date = min(ts.index)
    else:
        train_start_date = pd.to_datetime(args.train_start_date)

    if ts.index.max() >= options.warn_start_date:
        train_end_date = options.warn_start_date - pd.Timedelta(days=1)
    else:
        train_end_date = ts.index.max()

    #######################################################################
    # load external data
    #######################################################################

    if options.external_source is not None and options.external_keyword is not None:
        print("Loading external signals.")

        ex_index = "effect"
        if options.data_replication:
            ex_index = ex_index + '-' + (
                options.warn_start_date +
                pd.Timedelta(days=-1)).strftime("%Y%m%d")
            print("Replication index: ", ex_index)
        else:
            ex_index = 'effect'

        #ts_external = dp.get_ts_malware(publisher=options.external_source,
        
        
        ext_source_name = options.external_source
        ext = ext_source_name
        sig = ""
        for keyword in options.external_keyword.split("_"):
            sig += (keyword + " ")
        sig = sig[0:len(sig)-1]
        dateAttr='datePublished'

        if ext == 'blogs':
            pub = 'hg-blogs'
            ind = ex_index + '/blog'
        elif ext == 'twitter':
            pub = 'asu-twitter'
            ind = ex_index + '/socialmedia'
        elif ext == 'd2web':
            pub = 'asu-hacking-posts'
            ind = ex_index + '/post'
        elif ext == 'vulnerability':
            pub = 'hg-cve'
            ind = ex_index + '/vulnerability'
            dateAttr='startDate'
        else:
            print('External Signal not supported')
        
        print("\tindex: ", ind)
        ts_external = dp.get_ts_external(end_date=warn_timestamp,
                                        publisher=pub,
                                        hostname=options.data_hostname,
                                        index=ind,
                                        keyword=sig, dateAttr=dateAttr,
                                        sources_ablated=sources_ablated)
        # ts_external = dp.get_ts_malware_exclue_Locky(
        #     publisher=options.external_source, hostname=options.data_hostname,
        #     index=ex_index)
        if (ts_external is None):
            print('No data for query; external data source not used.')
            ts_external_shifted = None
        else:
            min_date = ts_external.index.min()
            max_date = ts_external.index.max()
            ts_external = ts_external.reindex(
                index=pd.date_range(min_date, max_date),
                fill_value=0)

            # if options.data_source == "dexter_endpoint-malware":
            #     lag = 29
            # elif options.data_source == "armstrong_endpoint-malware":
            #     lag = 12
            # elif options.data_source == "knox_endpoint-malware":
            #     lag = 25
            # else:
            #     print("Not a valid data source.")
            #     exit(1)


            slice_start_date = ts.index.min().date() - pd.Timedelta(days=30)
            slice_end_date = train_end_date

            ts_exogs = ts_external
            df_cc, lagged_ts = lin_corr.pairwise_ccf(
                ts, ts_exogs, slice_start_date=slice_start_date,
                slice_end_date=slice_end_date)



            #ts_external_shifted = ts_external.shift(lag, freq=1)
            #ts_external_shifted = ts_external_shifted[learn_start_date:]
            ts_external_shifted = lagged_ts[learn_start_date:]
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
    ts_pred = learn_rnn_parameters(
        ts, ts_external_shifted, options, look_ahead=options.look_ahead,
        test_start_date=options.warn_start_date)

    print("Forecast:")
    print(ts_pred)
    ts_pred = np.around(ts_pred[-1:])
    print("Rounded prediction =", ts_pred)

    if ts_pred.sum() > 0:
        # generate model metadata
        model_id = write_model_metadata(options, es_query)

        # generate warnings
        if options.warn_version == 1:
            model_output.write_model_output(options, ts_pred, model_id,
                                            output_name="rnn_warning")
        else:
            company = model_output.get_target_orgnization(options)
            if company == "knox":
                model_output.write_model_output_v3(
                    options, ts_pred, model_id,
                    output_name="rnn_warning_only_es")
            else:
                model_output.write_model_output_v2(
                    options, ts_pred, model_id,
                    output_name="rnn_warning_only_es")
    else:
        print("Model did not predict any attacks. There will be no warnings.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
