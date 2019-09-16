#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np


def plot_custom_hist(listX=None, bins=None):
    if listX is None:
        np.random.seed(1)
        values_A = np.random.choice(np.arange(600), size=200,
                                    replace=True).tolist()
        values_B = np.random.choice(np.arange(600), size=200,
                                    replace=True).tolist()
        bins = np.arange(0, 350, 25)
    else:
        values_A, values_B = listX

    """
    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    _, bins, patches = plt.hist([np.clip(values_A, bins[0], bins[-1]),
                                 np.clip(values_B, bins[0], bins[-1])],
                                normed=1,
                                # normed is deprecated and will be replaced by density
                                bins=bins, color=['#3782CC', '#AFD5FA'],
                                label=['A', 'B'])

    xlabels = [str(b) for b in bins[1:]]
    xlabels[-1] = '300+'

    N_labels = len(xlabels)
    plt.xlim([0, 325])
    plt.xticks(25 * np.arange(N_labels) + 12.5)
    ax.set_xticklabels(xlabels)

    plt.yticks([])
    plt.title('')
    plt.setp(patches, linewidth=0)
    plt.legend(loc='upper left')

    fig.tight_layout()

    return plt


def main(argv):
    plt = plot_custom_hist()
    plt.show()

    values_A = np.random.choice(np.arange(600), size=200,
                                replace=True).tolist()
    values_B = np.random.choice(np.arange(600), size=200,
                                replace=True).tolist()
    bins = np.arange(0, 20, 5)

    plt = plot_custom_hist([values_A, values_B], bins)
    plt.show()


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
