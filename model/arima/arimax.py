#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from model.model_evaluation import ModelEvaluation
from model.arima import auto_arima

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class ARIMAX:
    def __init__(self, ts=None, train_start_date=None, train_end_date=None):
        self.ts = ts
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

    def do_evaluation(self, modeleval: ModelEvaluation, round_value=False,
                      make_nonnegative=False):
        ts_pred = self.fit_and_forecast(modeleval)
        if round_value:
            ts_pred = np.around(ts_pred)
        if make_nonnegative:
            ts_pred[ts_pred < 0] = 0
        modeleval.evaluate(modeleval.ts_test, ts_pred)
        # print("RMSE:", modeleval.rmse)

    def fit_and_forecast(self, modeleval: ModelEvaluation):
        model_param = modeleval.model_param
        test_param = modeleval.test_param
        ts_train = modeleval.ts_train
        ts_test = modeleval.ts_test
        ts_exog_train = modeleval.ts_exog_train
        ts_exog_test = modeleval.ts_exog_test

        print("Length of train =", ts_train.size)
        print("Length of test =", ts_test.size)
        print("Length of train external=", ts_exog_train.size)
        print("Length of test external=", ts_exog_test.size)

        print("train")
        print(ts_train.head(2))
        print(ts_train.tail(2))

        print("test")
        print(ts_test.head(2))
        print(ts_test.tail(2))

        print("train exog")
        print(ts_exog_train.head(2))
        print(ts_exog_train.tail(2))

        print("test exog")
        print(ts_exog_test.head(2))
        print(ts_exog_test.tail(2))

        list_pred = []

        if test_param['sliding_window']:
            print("Sliding window approach")
            if test_param['window_size']:
                window_size = test_param['window_size']
            else:
                window_size = len(ts_train)

            if not test_param["look_ahead_step"]:
                look_ahead_step = 1
            else:
                look_ahead_step = test_param["look_ahead_step"]
            cur_window_data_points = ts_train[-window_size:]
            # cur_window_data_points = list(ts_train[-window_size:].values)
            # TO DO: need to fix this
            cur_window_data_points_exog = ts_exog_train[-window_size:]

            for i in range(0, len(ts_test), look_ahead_step):
                ARIMAX_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
                    auto_arima.iterative_ARIMAX_fit(cur_window_data_points,
                                                    model_param['max_p'],
                                                    model_param['max_d'],
                                                    model_param['max_q'],
                                                    exog=cur_window_data_points_exog)
                print("i =", i)
                print("Selected Model Order=", min_aic_fit_order)
                print("AIC score for selected model=", min_aic)
                print(ts_exog_test[i:i + look_ahead_step])

                if i + look_ahead_step > len(ts_test):
                    forecast_res = min_aic_fit_res.forecast(
                        steps=len(ts_exog_test[i:i + look_ahead_step]),
                        exog=ts_exog_test[
                             i:i + look_ahead_step].values)
                else:
                    forecast_res = min_aic_fit_res.forecast(
                        steps=look_ahead_step, exog=ts_exog_test[
                                                    i:i + look_ahead_step].values)

                list_pred.extend(forecast_res[0])
                cur_window_data_points = cur_window_data_points[
                                         look_ahead_step:]
                temp_val = ts_test[i:i + look_ahead_step]
                cur_window_data_points = cur_window_data_points.append(temp_val)

                cur_window_data_points_exog = cur_window_data_points_exog[
                                              look_ahead_step:]
                temp_val = ts_exog_test[i:i + look_ahead_step]
                cur_window_data_points_exog = cur_window_data_points_exog.append(
                    temp_val)
        else:
            ARIMAX_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
                auto_arima.iterative_ARIMAX_fit(ts_train, model_param['max_p'],
                                                model_param['max_d'],
                                                model_param['max_q'],
                                                exog=ts_exog_train.values)
            # print("Min AIC=", min_aic)
            # print("Order=", min_aic_fit_order)

            print("ts")
            print(ts_train.index.min())
            print(ts_train.index.max())
            print(ts_test.index.min())
            print(ts_test.index.max())
            print(ts_exog_train.index.min())
            print(ts_exog_train.index.max())
            print(ts_exog_test.index.min())
            print(ts_exog_test.index.max())

            print("ts_exog")
            train_end_date = ts_train.index.max()
            test_end_date = ts_test.index.max()
            gap_and_test_dates = pd.date_range(
                train_end_date + pd.Timedelta(days=1),
                test_end_date)
            forecast_res = min_aic_fit_res.forecast(
                steps=len(gap_and_test_dates),
                exog=ts_exog_test.values)
            list_pred = forecast_res[0][-len(ts_test):]
        ts_pred = pd.Series(list_pred[:ts_test.size], index=ts_test.index)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        # ts_pred = pd.Series(list_pred, index=ts_test.index)
        # ts_pred.name = 'count'
        # ts_pred.index.name = 'date'

        return ts_pred

    def fit2(self, ts, ts_exog, max_p, max_d, max_q, trend='c'):
        ARIMAX_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
            auto_arima.iterative_ARIMAX_fit(ts, max_p, max_d, max_q,
                                            exog=ts_exog, trend=trend)
        print("Selected Model Order=", min_aic_fit_order)
        print("AIC score for selected model=", min_aic)
        return min_aic_fit_res, min_aic_fit_order
