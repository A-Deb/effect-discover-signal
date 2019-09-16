#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from scipy.optimize import brute

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

# import dataprocessing as dp

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


# Cannot cast ufunc subtract output from dtype('float64') to dtype('int64') with
# casting rule 'same_kind'

def objfunc(order, endog, exog):
    """
    order: (p,d,q)
    endog: endogenous variable
    exog: exogenous variables
    """
    print(endog, order, exog)
    fit = ARIMA(endog, order, exog).fit()
    return fit.aic()


def objfunc2(order, endog):
    """
    order: (p,d,q)
    endog: endogenous variable
    exog: exogenous variables
    """
    model = ARIMA(endog, order)
    res = model.fit()
    return res.aic


def auto_arima(ts):
    grid = (slice(0, 3, 1), slice(0, 2, 1), slice(0, 3, 1))
    brute(objfunc2, grid, args=(ts,), finish=None)


def auto_arima_statsmodels(ts, show_convg_info=False):
    min_aic = np.inf
    min_aic_param = None

    for p in range(3):
        for d in range(3):
            for q in range(3):
                # if (p, d, q) != (0, 0, 0):
                try:
                    # aic = objfunc2((p, q, d), ts)
                    # print(p, d, q)
                    model = ARIMA(ts, (p, d, q))
                    model_fit = model.fit(disp=show_convg_info)
                    # print(model_fit.aic)
                    if model_fit.aic < min_aic:
                        min_aic = model_fit.aic
                        min_aic_param = (p, d, q)
                except:
                    # print("------------", p, d, q)
                    pass

    if min_aic_param is None:
        print("Not successful in fitting ARIMA")
        return None
    else:
        model = ARIMA(ts, min_aic_param)
        model_fit = model.fit(disp=show_convg_info)  # M-H
        return (min_aic, min_aic_param, model, model_fit.forecast(steps=1))


# print("Min value")
# d = dict(zip(params, aics))
# minaic = min(d, key=d.get)
# print(minaic)
# print(d)


def auto_arima_pyflux(ts_df):
    import pyflux as pf
    min_aic = np.inf
    min_aic_param = None

    for p in range(3):
        for d in range(3):
            for q in range(3):
                if (p, d, q) != (0, 0, 0):
                    model = pf.ARIMA(data=ts_df, ar=p, integ=d, ma=q)
                    model_fit = model.fit("MLE")  # M-H
                    if model_fit.aic < min_aic:
                        min_aic = model_fit.aic
                        min_aic_param = (p, d, q)
    if min_aic_param is None:
        print("Not successful in fitting ARIMA")
        return -1
    else:
        model = pf.ARIMA(data=ts_df, ar=min_aic_param[0],
                         integ=min_aic_param[1], ma=min_aic_param[2])
        model_fit = model.fit("MLE")  # M-H
        return min_aic, min_aic_param, model, model.predict(h=1, intervals=True)


def iterative_ARIMA_fit(ts, max_ar=2, max_dff=2, max_ma=2):
    """ Iterates within the allowed values of the p and q parameters

    Returns a dictionary with the successful fits.
    Keys correspond to models.
    """
    ts = ts.astype('float')
    ARIMA_fit_results = {}

    min_aic = np.inf
    min_aic_fit_order = None
    min_aic_fit_res = None

    for AR in range(max_ar + 1):
        for MA in range(max_ma + 1):
            for Diff in range(max_dff + 1):
                model = ARIMA(ts, order=(AR, Diff, MA))
                try:
                    results_ARIMA = model.fit(disp=False, method='css')
                    fit_is_available = True
                except:
                    # print("\tDidn't find a fit")
                    continue

                if fit_is_available:
                    # print("\tFound a fit (%d,%d,%d)" % (AR, Diff, MA))
                    # print("\tAIC score =", results_ARIMA.aic)
                    ARIMA_fit_results['%d-%d-%d' % (AR, Diff, MA)] = \
                        results_ARIMA
                    if results_ARIMA.aic < min_aic:
                        min_aic = results_ARIMA.aic
                        min_aic_fit_order = (AR, Diff, MA)
                        # min_aic_fit_res = ARIMA_fit_results
                        min_aic_fit_res = results_ARIMA

    return ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res


def iterative_SARIMA_fit(ts, max_ar=2, max_dff=1, max_ma=2, s_max_ar=2,
                         s_max_diff=1, s_max_ma=2, s=7):
    """ Iterates within the allowed values of the p and q parameters
    Returns a dictionary with the successful fits.
    Keys correspond to models.
    """
    ts = ts.astype('float')
    SARIMA_fit_results = {}

    min_aic = np.inf
    min_aic_fit_order = None
    min_aic_fit_res = None

    for AR in range(max_ar + 1):
        for Diff in range(max_dff + 1):
            for MA in range(max_ma + 1):
                for sAR in range(s_max_ar + 1):
                    for sDiff in range(s_max_diff + 1):
                        for sMA in range(s_max_ma + 1):
                            model = smt.SARIMAX(
                                ts, order=(AR, Diff, MA),
                                seasonal_order=(sAR, sDiff, sMA, s))
                            try:
                                results_SARIMA = model.fit(disp=False,
                                                           method='lbfgs')
                                fit_is_available = True
                            except:
                                # print("\tDidn't find a fit")
                                continue

                            if fit_is_available:
                                # print("\tFound a fit (%d,%d,%d)" % (AR, Diff, MA))
                                # print("\tAIC score =", results_ARIMA.aic)
                                SARIMA_fit_results[
                                    '%d-%d-%d--%d-%d-%d-%d' % (
                                        AR, Diff, MA, sAR, sDiff, sMA, s)] = \
                                    results_SARIMA
                                if results_SARIMA.aic < min_aic:
                                    min_aic = results_SARIMA.aic
                                    min_aic_fit_order = (
                                        AR, Diff, MA, sAR, sDiff, sMA, s)
                                    # min_aic_fit_res = ARIMA_fit_results
                                    min_aic_fit_res = results_SARIMA

    return SARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res


def iterative_ARIMAX_fit(ts, max_ar=2, max_dff=2, max_ma=2, exog=None,
                         trend='c', verbose=False):
    """ Iterates within the allowed values of the p and q parameters

    Returns a dictionary with the successful fits.
    Keys correspond to models.
    """
    ts = ts.astype('float')
    ARIMA_fit_results = {}

    min_aic = np.inf
    min_aic_fit_order = None
    min_aic_fit_res = None

    for AR in range(max_ar + 1):
        for MA in range(max_ma + 1):
            for Diff in range(max_dff + 1):
                model = ARIMA(ts, order=(AR, Diff, MA), exog=exog)
                try:
                    results_ARIMA = model.fit(disp=False, method='css',
                                              trend=trend)
                    fit_is_available = True
                except:
                    if verbose:
                        print("\tNo fit is found")
                    continue

                if fit_is_available:
                    if verbose:
                        print("\tFound a fit (%d,%d,%d)" % (AR, Diff, MA))
                        print("\tAIC score =", results_ARIMA.aic)
                    ARIMA_fit_results['%d-%d-%d' % (AR, Diff, MA)] = \
                        results_ARIMA
                    if results_ARIMA.aic < min_aic:
                        min_aic = results_ARIMA.aic
                        min_aic_fit_order = (AR, Diff, MA)
                        # min_aic_fit_res = ARIMA_fit_results
                        min_aic_fit_res = results_ARIMA

    return ARIMA_fit_results, min_aic, min_aic_fit_order, min_aic_fit_res


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-f', '--data-file',
        default='ts_syria.csv',
        help='Files containing time series of counts')
    # options = parser.parse_args()


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
