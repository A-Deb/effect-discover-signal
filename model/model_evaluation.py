#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
from util.measures import measures
from plotting import custom_plot
from model.arima import auto_arima

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class MetaModelClass:
    def __init__(self, name, param):
        self.name = name
        self.param = param


def get_files_in_dir(dirpath):
    import os
    filepaths = []
    for dirName, subdirList, fileList in os.walk(dirpath):
        # print('Found directory: %s' % dirName)
        # print(subdirList)
        for fname in fileList:
            filepaths.append(os.path.join(dirpath, fname))
    return filepaths


def load_json(fpath, flat=False):
    with open(fpath, 'rb') as fh:
        decoded = json.loads(fh.read().decode('utf-8-sig'))
        return decoded if flat else decoded['values']


def load_mansa_gsr(gsrdir=None):
    if gsrdir is None:
        gsr_data_dir = "/Users/tozammel/safe/mercury/data/gsr/"
        mansa_gsr_data_dir = gsr_data_dir + "mansa_gsr/"
        gsrdir = mansa_gsr_data_dir

    gsr_data = []
    mansa_gsr_filepaths = get_files_in_dir(gsrdir)
    # print(mansa_gsr_filepaths)
    for filepath in mansa_gsr_filepaths:
        temp_gsr_data = load_json(filepath, flat=True)
        gsr_data.extend(temp_gsr_data)

    # columns:
    # ['Actor', 'Approximate_Location', 'Casualties', 'City', 'Country',
    #  'Earliest_Reported_Date', 'Encoding_Comment', 'Event_Date', 'Event_ID',
    #  'Event_Subtype', 'Event_Type', 'First_Reported_Link', 'GSS_Link',
    #  'Latitude', 'Longitude', 'News_Source', 'Other_Links', 'Revision_Date',
    #  'State', 'Target', 'Target_Name', 'Target_Status'],

    mansa_gsr_df = pd.DataFrame(gsr_data)
    # change the dtypes of the some columns
    mansa_gsr_df['Event_Date'] = pd.to_datetime(mansa_gsr_df['Event_Date'])
    mansa_gsr_df['Earliest_Reported_Date'] = pd.to_datetime(
        mansa_gsr_df['Earliest_Reported_Date'])
    mansa_gsr_df['Revision_Date'] = pd.to_datetime(
        mansa_gsr_df['Revision_Date'])
    # NOTE: the column Casualties has str + int data. Not sure how to handle it
    # mansa_gsr_df['Casualities'] = mansa_gsr_df['Casualities'].astype(int)
    return mansa_gsr_df


def learn_param(hmm_data, num_states, emission, n_iter=50):
    models = {'Gaussian': GaussianHMM, 'Poisson': PoissonHMM,
              'Geometric': GeometricHMM,
              'HurdleGeometric': HurdleGeometricHMM}

    hmm_model = models[emission]
    model = hmm_model(n_components=num_states,
                      algorithm='viterbi',
                      n_iter=50).fit(hmm_data)

    e = []  # expected number of events

    for i in range(num_states):
        if hmm_model == GaussianHMM or hmm_model == PoissonHMM:
            e.append(model.means_[i][0])
        elif hmm_model == GeometricHMM:
            e.append(1.0 / (1.0 - model.means_[i][0]) - 1.0)
        elif hmm_model == HurdleGeometricHMM:
            e.append(1.0 / (1.0 - model.mu_[i][0]) * model.gamma_[i][0])

    # states = model.predict(hmm_data)
    # last_day_state = states[-1]
    # idx = np.argsort(e)
    # # states in the order of 0, 1, 2, ...
    # true_last_state = np.where(idx == last_day_state)[0][0]
    #
    # n_pred_events = 0
    # for i in range(num_states):
    #     p = model.transmat_[last_day_state, i]  # P(last state = i)
    #     n_pred_events += p * e[i]

    # return hmm_model, model, e, true_last_state, n_pred_events
    return hmm_model, model, e


def fit_hmm(model_param, ts_train, ts_test):
    print("Fitting with HMM")
    history = [[x] for x in ts_train]
    hmm_model, model, e = learn_param(history, model_param['num_states'],
                                      model_param['emission'])
    print("Expected num of events =", e)

    train_end_date = ts_train.index.max()
    test_start_date = ts_test.index.min()
    # print(train_end_date)
    # print(test_start_date)

    gap_dates = pd.date_range(train_end_date + pd.Timedelta(days=1),
                              test_start_date, closed="left")
    # predict events for the gaps
    for adate in gap_dates:
        states = model.predict(history)
        last_day_state = states[-1]
        idx = np.argsort(e)
        # # states in the order of 0, 1, 2, ...
        true_last_state = np.where(idx == last_day_state)[0][0]

        n_pred_events = 0
        for i in range(model_param['num_states']):
            p = model.transmat_[last_day_state, i]  # P(last state = i)
            n_pred_events += p * e[i]
        history.append([n_pred_events])

    predictions = []
    print(len(ts_test.index))
    for adate in ts_test.index:
        states = model.predict(history)
        last_day_state = states[-1]
        idx = np.argsort(e)
        # # states in the order of 0, 1, 2, ...
        true_last_state = np.where(idx == last_day_state)[0][0]

        n_pred_events = 0
        for i in range(model_param['num_states']):
            p = model.transmat_[last_day_state, i]  # P(last state = i)
            n_pred_events += p * e[i]
        history.append([n_pred_events])
        predictions.append((adate.date(), n_pred_events))
        print(adate.date(),
              'predicted=%f, expected=%f' % (n_pred_events, ts_test.ix[adate]))

        # filename = ts_name + "_pred_hmm_v2_w_" + str(size) + \
        #            "_z_" + str(num_states) + "_e_" + emission + ".csv"
        # return pd.DataFrame(predictions), filename
    df_pred = pd.DataFrame(predictions, columns=['date', 'count'])
    df_pred.set_index('date', inplace=True)
    return df_pred['count']


def fit_arima(model_param, ts_train, ts_test):
    ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
        auto_arima.iterative_ARIMA_fit(ts_train, model_param['max_p'],
                                       model_param['max_d'],
                                       model_param['max_q'])
    # print("Min AIC=", min_aic)
    # print("Order=", min_aic_fit_order)

    train_end_date = ts_train.index.max()
    test_end_date = ts_test.index.max()
    gap_and_test_dates = pd.date_range(train_end_date + pd.Timedelta(days=1),
                                       test_end_date)
    forecast_res = min_aic_fit_res.forecast(steps=len(gap_and_test_dates))
    list_pred = forecast_res[0][-len(ts_test):]
    ts_pred = pd.Series(list_pred, index=ts_test.index)
    ts_pred.name = 'count'
    ts_pred.index.name = 'date'
    return ts_pred


class ModelEvaluation(object):
    @staticmethod
    def get_time_series(gsr_dir=None, country=None, actor=None, city=None,
                        event_type=None):
        mansa_gsr_df = load_mansa_gsr(gsrdir=None)

        # remove duplicate events by Event ID
        mansa_gsr_df_by_unique_event_id = mansa_gsr_df.groupby(
            'Event_ID', group_keys=False, as_index=False).apply(
            lambda x: x.ix[x.Event_Date.idxmax()])

        if country:
            df_subset = mansa_gsr_df_by_unique_event_id[
                mansa_gsr_df_by_unique_event_id['Country'] == country
                ]
        elif actor:
            df_subset = mansa_gsr_df_by_unique_event_id[
                mansa_gsr_df_by_unique_event_id['Actor'] == actor
                ]

        ts = df_subset.groupby('Event_Date').size()
        idx = pd.date_range(min(ts.index), max(ts.index))
        ts = ts.reindex(idx, fill_value=0)
        return ts

    def __init__(self, train_start_date, train_end_date,
                 test_start_date, test_end_date, model_name, model_param=None,
                 time_series=None, gsr_dir=None, dataset_name='Syria',
                 country='Syria', actor=None, city=None, event_type=None,
                 verbose=False, exog=None, normalized_ts=False,
                 test_param=None):
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.model_name = model_name
        self.model_param = model_param
        self.dataset_name = dataset_name
        self.gsr_dir = gsr_dir
        self.country = country
        self.event_type = event_type
        self.actor = actor
        self.city = city
        self.test_param = test_param
        self.verbose = verbose

        if time_series is not None:
            self.ts = time_series
        else:
            self.ts = self.get_time_series(gsr_dir=self.gsr_dir,
                                           country=self.country,
                                           actor=self.actor, city=self.city,
                                           event_type=self.event_type)
        if normalized_ts:
            self.ts = (self.ts - self.ts.values.min()) / (
                self.ts.values.max() - self.ts.values.min())

        if test_param['train_portion'] > 0:
            train_size = int(np.around(
                self.ts.size * test_param['train_portion']))
            self.ts_train = self.ts[:train_size]
            self.ts_test = self.ts[train_size:]
        else:
            self.ts_train = self.ts[self.train_start_date:self.train_end_date]
            self.ts_test = self.ts[self.test_start_date:self.test_end_date]

        # print(self.ts.head())
        # print(self.ts.tail())
        # print(self.ts_train.index.min())
        # print(self.ts_train.index.max())
        # print(self.ts_test.index.min())
        # print(self.ts_test.index.max())

        if exog is not None and exog is not '':
            exog_df = pd.read_csv(exog, header=0, parse_dates=True, index_col=0)
            # print(exog_df.head())
            # print(exog_df.tail())
            if test_param['train_portion'] > 0:
                train_size = int(np.around(
                    self.ts.size * test_param['train_portion']))
                self.ts_exog_train = exog_df[:train_size]
                self.ts_exog_test = exog_df[train_size:]
            else:
                self.ts_exog_train = exog_df[
                                     self.train_start_date:self.train_end_date]
                self.ts_exog_test = exog_df[self.test_start_date:self.test_end_date]

            # print(self.ts_exog_train.index.min())
            # print(self.ts_exog_train.index.max())
            # print(self.ts_exog_test.index.min())
            # print(self.ts_exog_test.index.max())
            # input("press a key")

            if normalized_ts:
                self.ts_exog_train = (self.ts_exog_train -
                                      self.ts_exog_train.values.min()) / (
                                         self.ts_exog_train.values.max() -
                                         self.ts_exog_train.values.min())
                self.ts_exog_test = (self.ts_exog_test -
                                     self.ts_exog_test.values.min()) / (
                                        self.ts_exog_test.values.max() -
                                        self.ts_exog_test.values.min())

    def describe(self):
        print("Dataset Name =", self.dataset_name)
        print("Train start date =", self.train_start_date)
        print("Train end date =", self.train_end_date)
        print("Test start date =", self.test_start_date)
        print("Test end date =", self.test_end_date)
        print("Country =", self.country)
        print("City =", self.city)
        print("Actor =", self.actor)
        print("Event type =", self.event_type)
        print("Time series shape =", self.ts.shape)
        print("Train:")
        print("\t", self.ts_train.index.min())
        print("\t", self.ts_train.index.max())
        print("\t#days =", self.ts_train.size)
        print("Test:")
        print("\t", self.ts_test.index.min())
        print("\t", self.ts_test.index.max())
        print("\t#days =", self.ts_test.size)
        print("Model name =", self.model_name)
        print("Model param =", self.model_param)

    def plot(self):
        # plt = custom_plot.plot_ts([self.ts, self.ts_train, self.ts_test])
        plt = custom_plot.plot_ts(self.ts)
        plt.show()
        plt = custom_plot.plot_ts(self.ts_train)
        plt.show()
        plt = custom_plot.plot_ts(self.ts_test)
        plt.show()

    def fit(self):
        if self.model_name not in ['hmm', 'arima', 'fb_prophet']:
            raise Exception("Not a valid model name")
        if self.model_param is None:
            raise Exception("Model param is not given")
        print("Model =", self.model_name)
        print("Param =", self.model_param)

        if self.model_name == "hmm":
            ts_pred = fit_hmm(self.model_param, self.ts_train, self.ts_test)
            self.evaluate(self.ts_test, ts_pred)
            print("RMSE:", self.rmse)
        elif self.model_name == "arima":
            ts_pred = fit_arima(self.model_param, self.ts_train, self.ts_test)
            self.evaluate(self.ts_test, ts_pred)
            print("RMSE:", self.rmse)
        else:
            raise Exception("Given model is not implemented")

    def evaluate(self, true_ts, pred_ts):
        self.ts_true = true_ts
        self.ts_pred = pred_ts
        self.mae = measures.get_mae(true_ts, pred_ts)
        self.rmse = measures.get_rmse(true_ts, pred_ts)
        self.mase = measures.get_mase(true_ts, pred_ts)
        self.nash_sutcliffe = measures.get_nash_sutcliffe_score(true_ts,
                                                                pred_ts)
        self.rate = measures.get_rate(true_ts, pred_ts)
        self.precision_recall = measures.get_precision_recall(true_ts, pred_ts)

        if self.verbose:
            print("MAE:", self.mae)
            print("RMSE:", self.rmse)
            print("MASE:", self.mase)
            print("Nash-Sutcliffe:", self.nash_sutcliffe)
            print("RATE:", self.rate)
            print("PRECISION RECALL", self.precision_recall)
