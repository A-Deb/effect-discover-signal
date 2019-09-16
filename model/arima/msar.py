#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from model.model_evaluation import ModelEvaluation
from model.arima import auto_arima
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class MSAR:
    def __init__(self, ts=None, train_start_date=None, train_end_date=None):
        self.ts = ts
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date

    def do_evaluation(self, modeleval: ModelEvaluation):
        ts_pred = self.fit(modeleval)
        modeleval.evaluate(modeleval.ts_test, ts_pred)

    def fit(self, modeleval: ModelEvaluation):
        model_param = modeleval.model_param
        ts_train = modeleval.ts_train
        ts_test = modeleval.ts_test
        ts_exog_train = modeleval.ts_exog_train
        ts_exog_test = modeleval.ts_exog_test

        ARIMAX_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res = \
            auto_arima.iterative_ARIMAX_fit(ts_train, model_param['max_p'],
                                            model_param['max_d'],
                                            model_param['max_q'],
                                            exog=ts_exog_train)
        # print("Min AIC=", min_aic)
        # print("Order=", min_aic_fit_order)

        train_end_date = ts_train.index.max()
        test_end_date = ts_test.index.max()
        gap_and_test_dates = pd.date_range(
            train_end_date + pd.Timedelta(days=1),
            test_end_date)
        forecast_res = min_aic_fit_res.forecast(steps=len(gap_and_test_dates),
                                                exog=ts_exog_test)
        list_pred = forecast_res[0][-len(ts_test):]
        ts_pred = pd.Series(list_pred, index=ts_test.index)
        ts_pred.name = 'count'
        ts_pred.index.name = 'date'

        if "pred_days" in model_param:
            pass

        return ts_pred
