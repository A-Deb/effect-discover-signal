#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from model.model_evaluation import ModelEvaluation
from model.arima import auto_arima
from model.arguments import ModelArguments
from model.model_results import ModelResults

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class ARIMA:
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

            print(len(ts_test))
            for i in range(0, len(ts_test), look_ahead_step):
                ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
                    auto_arima.iterative_ARIMA_fit(cur_window_data_points,
                                                   model_param['max_p'],
                                                   model_param['max_d'],
                                                   model_param['max_q'])
                print("Selected Model Order=", min_aic_fit_order)
                print("AIC score for selected model=", min_aic)
                forecast_res = min_aic_fit_res.forecast(steps=look_ahead_step)
                list_pred.extend(forecast_res[0])
                cur_window_data_points = cur_window_data_points[
                                         look_ahead_step:]
                temp_val = ts_test[i:i + look_ahead_step]
                cur_window_data_points = cur_window_data_points.append(temp_val)

        else:
            ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
                auto_arima.iterative_ARIMA_fit(ts_train, model_param['max_p'],
                                               model_param['max_d'],
                                               model_param['max_q'])
            print("Selected Model Order=", min_aic_fit_order)
            print("AIC score for selected model=", min_aic)
            train_end_date = ts_train.index.max()
            test_end_date = ts_test.index.max()
            gap_and_test_dates = pd.date_range(
                train_end_date + pd.Timedelta(days=1),
                test_end_date)
            forecast_res = min_aic_fit_res.forecast(
                steps=len(gap_and_test_dates))
            list_pred = forecast_res[0][-len(ts_test):]
            # ts_pred = pd.Series(list_pred, index=ts_test.index)

        print(len(list_pred))
        print(len(ts_test))

        ts_pred = pd.Series(list_pred[:ts_test.size], index=ts_test.index)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'

        return ts_pred

    def fit(self, modelarg: ModelArguments):
        model_param = modelarg.model_param
        ts_train = modelarg.endog[
                   modelarg.train_start_date:modelarg.train_end_date]
        if 'window_size' in model_param:
            window_size = model_param['window_size']
        else:
            window_size = len(ts_train)

        ts_train = ts_train[-window_size:]
        ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
            auto_arima.iterative_ARIMA_fit(ts_train, model_param['max_p'],
                                           model_param['max_d'],
                                           model_param['max_q'])
        print("Selected Model Order=", min_aic_fit_order)
        print("AIC score for selected model=", min_aic)

        est_param = {"train_size": window_size, "order": min_aic_fit_order,
                     "aic": min_aic, "result": min_aic_fit_res}
        modelres = ModelResults(model_name="ARIMA",
                                estimated_param=est_param)
        return modelres

    def forecast(self, modelarg: ModelArguments, modelres: ModelResults):
        gap_dates = pd.date_range(
            modelarg.train_end_date + pd.Timedelta(days=1),
            modelarg.test_start_date, closed='left')
        concerned_dates = pd.date_range(modelarg.test_start_date,
                                        modelarg.test_end_date)
        gap_and_concerned_dates = gap_dates.append(concerned_dates)
        est_param = modelres.learned_param

        list_pred = [est_param['rate']] * gap_and_concerned_dates.size

        ts_pred = pd.Series(list_pred[-concerned_dates.size:],
                            index=concerned_dates)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        return ts_pred

    def fit2(self, ts, max_p, max_d, max_q):
        ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
            auto_arima.iterative_ARIMA_fit(ts, max_p, max_d, max_q)
        print("Selected Model Order=", min_aic_fit_order)
        print("AIC score for selected model=", min_aic)
        return min_aic_fit_res, min_aic_fit_order


class DaywiseARIMA:
    def __init__(self, ts=None, train_start_date=None, train_end_date=None):
        self.ts = ts
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

    def do_evaluation(self, modeleval: ModelEvaluation, round_value=False,
                      make_nonnegative=False):
        ts_pred = self.fit(modeleval)
        if round_value:
            ts_pred = np.around(ts_pred)
        if make_nonnegative:
            ts_pred[ts_pred < 0] = 0
        modeleval.evaluate(modeleval.ts_test, ts_pred)

    def fit(self, modeleval: ModelEvaluation):
        model_param = modeleval.model_param
        ts_train = modeleval.ts_train
        ts_test = modeleval.ts_test

        # create daywise data
        ts_dayofweek = ts_train.groupby(ts_train.index.dayofweek)
        train_end_date = ts_train.index.max()
        test_end_date = ts_test.index.max()

        gap_and_test_dates = pd.date_range(
            train_end_date + pd.Timedelta(days=1),
            test_end_date)
        gap_and_test_dates_dayofweek = gap_and_test_dates.groupby(
            gap_and_test_dates.dayofweek
        )

        test_dates_dayofweek = ts_test.index.groupby(ts_test.index.dayofweek)
        ts_pred = pd.Series()

        for day, ts_day in ts_dayofweek:
            ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
                auto_arima.iterative_ARIMA_fit(ts_day, model_param['max_p'],
                                               model_param['max_d'],
                                               model_param['max_q'])
            print("Selected Model Order=", min_aic_fit_order)
            print("AIC score for selected model=", min_aic)
            indx_gap_test_dayofweek = gap_and_test_dates_dayofweek[day]
            indx_test_dayofweek = test_dates_dayofweek[day]
            forecast_res = min_aic_fit_res.forecast(
                steps=len(indx_gap_test_dayofweek))
            list_pred = forecast_res[0][-len(indx_test_dayofweek):]
            ts_pred_sub = pd.Series(list_pred, index=indx_test_dayofweek)
            ts_pred = ts_pred.append(ts_pred_sub)

            print("Model info:")
            print("Day =", day)
            print("Min aic order =", min_aic_fit_order)

        ts_pred = ts_pred.sort_index()

        if "pred_days" in model_param:
            pass

        ts_pred.name = 'count'
        ts_pred.index.name = 'date'
        return ts_pred

    def fit2(self, ts, max_p, max_d, max_q):
        ts_dayofweek = ts.groupby(ts.index.dayofweek)
        train_end_date = ts.index.max()
        test_end_date = ts.index.max()

        estimated_models = {}
        for day, ts_day in ts_dayofweek:
            ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
                auto_arima.iterative_ARIMA_fit(ts_day, max_p, max_d, max_q)
            print("Selected Model Order=", min_aic_fit_order)
            print("AIC score for selected model=", min_aic)
            # print("TS: ", ts_day)
            estimated_models[day] = {}
            estimated_models[day]['min_aic_fit_order'] = min_aic_fit_order
            estimated_models[day]['min_aic_fit_res'] = min_aic_fit_res

        return estimated_models
