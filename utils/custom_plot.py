#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
# %matplotlib inline
from matplotlib.pylab import rcParams

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


rcParams['figure.figsize'] = 15, 6


def plot_ts(ts, xlabel='Days', ylabel='Num of Attacks', legend='', title='',
            col='blue', lstyle='-', lw=2, marker='o'):
    plt.plot(ts, color=col, label=legend, linestyle=lstyle, linewidth=lw)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title)
    return plt


def plot_true_pred(ts_true, ts_pred, xlabel='Days', ylabel='Num of Attacks',
                   leg_true='Observed', leg_pred='Predicted', lstyle='-',
                   lw=2, marker=''):
    plt.plot(ts_true, color='blue', label=leg_true, linestyle=lstyle,
             linewidth=lw, marker=marker)
    plt.plot(ts_pred, color='red', label=leg_pred, linestyle=lstyle,
             linewidth=lw, marker=marker)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    return plt


def plot_list_of_lines(ts_list, legends=None, xlabel='Days',
                       ylabel='Num of Attacks', lstyle='-', lw=2, marker='',
                       title=""):
    colors = ['black', 'blue', 'green', 'magenta', 'red']
    if legends is None:
        legends = []
        for i in range(len(ts_list)):
            legends.append("Line " + str(i + 1))

    print(colors)
    print(legends)

    for idx, ts in enumerate(ts_list):
        plt.plot(ts, color=colors[idx], label=legends[idx], linestyle=lstyle,
                 linewidth=lw, marker=marker)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(loc="upper left", prop={'size': 18})
    plt.title(title)

    return plt
