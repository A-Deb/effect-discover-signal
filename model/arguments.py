#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


class ModelArguments:
    def __init__(self, time_series, train_start_date=None, train_end_date=None,
                 test_start_date=None, test_end_date=None, model_name=None,
                 model_param={}, nowcast=True, forecast_size=7, gsr_dir=None,
                 dataset_name='Syria',
                 country='Syria', actor=None, city=None, event_type=None,
                 verbose=False, exog=None):
        if time_series is None:
            raise Exception("Time series is None")
        self.endog = time_series
        if train_start_date is None:
            self.train_start_date = time_series.index.min()
        else:
            self.train_start_date = pd.to_datetime(train_start_date).date()

        if train_end_date is None:
            # use all of the data for training
            self.train_end_date = time_series.index.max()
        else:
            self.train_end_date = pd.to_datetime(train_end_date).date()

        if test_start_date is None:
            if nowcast:
                self.test_start_date = pd.datetime.now().date()
            else:
                self.test_start_date = self.train_end_date + pd.Timedelta(
                    days=1)
        else:
            self.test_start_date = pd.to_datetime(test_start_date).date()
            if self.test_start_date < self.train_end_date:
                self.test_start_date = self.train_end_date + pd.Timedelta(
                    days=1)

        if test_end_date is None:
            self.test_end_date = self.test_start_date + pd.Timedelta(
                days=forecast_size - 1)
        else:
            self.test_end_date = pd.to_datetime(test_end_date).date()
            if self.test_end_date < self.test_start_date:
                self.test_end_date = self.test_start_date

        self.model_name = model_name
        self.model_param = model_param
        self.dataset_name = dataset_name
        self.gsr_dir = gsr_dir
        self.country = country
        self.event_type = event_type
        self.actor = actor
        self.city = city
        self.verbose = verbose
        self.exog = exog
