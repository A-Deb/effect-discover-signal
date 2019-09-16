#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib.pylab import rcParams
from collections import OrderedDict

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

# https://tableaufriction.blogspot.com/2012/11/finally-you-can-use-tableau-data-colors.html
"""
Plotting colors
"""


def get_line_styles():
    return ['-', '--', '-.', ':']


def get_different_dashed_line():
    linestyles = OrderedDict(
        [('solid', (0, ())),
         ('loosely dotted', (0, (1, 10))),
         ('dotted', (0, (1, 5))),
         ('densely dotted', (0, (1, 1))),

         ('densely dashed', (0, (5, 1))),
         ('loosely dashed', (0, (5, 10))),
         ('dashed', (0, (5, 5))),
         # ('densely dashed', (0, (5, 1))), # changed by me

         ('loosely dashdotted', (0, (3, 10, 1, 10))),
         ('dashdotted', (0, (3, 5, 1, 5))),
         ('densely dashdotted', (0, (3, 1, 1, 1))),

         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
         ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
         ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

    return linestyles


def get_marker_styles():
    return ['o', 's', 'D', '*', '1', '2']


def convert_rgb_to_01_range(tableau):
    for i in range(len(tableau)):
        r, g, b = tableau[i]
        tableau[i] = (r / 255., g / 255., b / 255.)
    return tableau


def tableau_10_colors():
    # filepath = "color/tableau_10.csv"
    # df = pd.read_csv(filepath, header=0, sep="\t", encoding='latin1')
    # print(df)
    # colors = df['RGB']
    # print(colors)

    # tableau10 = [(255, 127, 14), (214, 39, 40), (148, 103, 189), (44, 160, 44),
    #              (31, 119, 180), (227, 119, 194), (188, 189, 34), (140, 86, 75),
    #              (127, 127, 127),
    #              (23, 190, 207)]
    # reorder color
    tableau10 = [(255, 127, 14), (44, 160, 44), (214, 39, 40),
                 (148, 103, 189), (31, 119, 180), (140, 86, 75),
                 (227, 119, 194), (188, 189, 34), (127, 127, 127),
                 (23, 190, 207)]

    return convert_rgb_to_01_range(tableau10)


def tableau_20_colors():
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14),
                 (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75),
                 (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127),
                 (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207),
                 (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    return tableau20


def get_plot(yticks_range=range(0, 11, 1)):
    import matplotlib.pyplot as plt

    # You typically want your plot to be ~1.33x wider than tall.
    # Common sizes: (10, 7.5) and (12, 9)
    plt.figure(figsize=(12, 9))

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.xticks(fontsize=14)
    # plt.yticks(range(5000, 30001, 5000), fontsize=14)
    plt.yticks(yticks_range, fontsize=14)


def figsize(scale=1.5):
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(
        5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def set_rcparam(rcParams, scale=1.5):
    rcParams['figure.figsize'] = figsize(scale)
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['xtick.major.size'] = 6
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.size'] = 6
    rcParams['ytick.major.width'] = 1


def get_minimal_plot(width=10, height=7.5):
    import matplotlib.pyplot as plt

    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare
    # exception because of the number of lines being plotted on it.
    # Common sizes: (10, 7.5) and (12, 9)
    # plt.figure(figsize=(12, 14))
    plt.figure(figsize=(width, height))

    # Remove the plot frame lines. They are unnecessary chartjunk.
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Ensure that the axis ticks only show up on the bottom and left of the plot.
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.yaxis.grid(True, ls="--")

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    plt.tick_params(axis="both", which="both", bottom="on", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    return plt, ax


def show_tableu10():
    cols = tableau_10_colors()
    plt, ax = get_minimal_plot()

    x = list(range(10))

    for i, col in enumerate(cols):
        print(i, col)
        y = [i + 10] * len(x)
        plt.plot(x, y, label=str(i), color=col)

    plt.legend(prop={'size': 10})
    plt.show()


def main(argv):
    show_tableu10()
    # tableau_10_colors()

    plt, ax = get_minimal_plot()
    x = list(range(10))
    y = [i * i for i in x]
    plt.plot(x, y)
    # plt.show()

    import pandas as pd


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
