#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from model.model_evaluation import ModelEvaluation
from model.arima import auto_arima

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class SARIMA:
    def __init__(self, ts=None, train_start_date=None, train_end_date=None):
        self.ts = ts
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

    def do_evaluation(self, modeleval: ModelEvaluation):
        ts_pred = self.fit(modeleval)
        modeleval.evaluate(modeleval.ts_test, ts_pred)
        # print("RMSE:", modeleval.rmse)

    def fit(self, modeleval: ModelEvaluation):
        model_param = modeleval.model_param
        ts_train = modeleval.ts_train
        ts_test = modeleval.ts_test

        SARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
            auto_arima.iterative_SARIMA_fit(
                ts_train, model_param['max_p'], model_param['max_d'],
                model_param['max_q'], model_param['s_max_p'],
                model_param['s_max_d'], model_param['s_max_q'], model_param['s']
            )
        print("Selected Model Order=", min_aic_fit_order)
        print("AIC score for selected model=", min_aic)

        train_end_date = ts_train.index.max()
        test_end_date = ts_test.index.max()
        gap_and_test_dates = pd.date_range(
            train_end_date + pd.Timedelta(days=1),
            test_end_date)
        forecast_res = min_aic_fit_res.predict(steps=len(gap_and_test_dates))
        # start : int, str, or datetime, optional
        # end : int, str, or datetime, optional
        # dynamic : boolean, int, str, or datetime, optional
        list_pred = forecast_res[0][-len(ts_test):]
        ts_pred = pd.Series(list_pred, index=ts_test.index)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'

        if "pred_days" in model_param:
            pass

        return ts_pred
