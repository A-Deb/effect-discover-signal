#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pylab as plt
import matplotlib.pyplot as plt

# %matplotlib inline
from matplotlib.pylab import rcParams

__version__ = 1.0
__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"

###########################################################################
# Publication quality plot
###########################################################################
"""

We do not really care about the exact size when our goal is to export a pdf or
an eps (for LaTeX) because they are vector formats. However, we care about the
ratio (length/width). Moreover, changing the size of the figure will affect the
relative size of the fonts.

As a rule of thumb, the font size of your labels should be close to the font
size of the figure's caption.

It is also a good idea to increase the linewidths, to be able to make the figure
small:


linestyle:
=========
-, --, :-, *-, o-
[‘solid’ | ‘dashed’, ‘dashdot’, ‘dotted’ | (offset, on-off-dash-seq)
| '-' | '--' | '-.' | ':' | 'None' | ' ' | '']

marker:
=======
http://matplotlib.org/api/markers_api.html
http://matplotlib.org/examples/lines_bars_and_markers/marker_reference.html

no marker: '', 'o'

color:
=======
color='#B22400', '#006BB2',

Try to go away from the classic 100% red/100% blue/etc. There are many color
scheme generators on the web: you select a nice color, and they generate
matching colors, given the number of colors you need.

http://www.paletton.com/
https://kuler.adobe.com/
http://www.perbang.dk/color+scheme/
http://www.colorsontheweb.com/colorwizard.asp

An interesting alternative is to use the python package brewer2mpl, which
implements the guidelines published by C. Brewer and colleagues for coloring
maps with sequential, divergent, and qualitative colors:
http://colorbrewer2.org/

> pip install brewer2mpl


legend:
=======
The parameter loc determines the position (1=top right, 2=top left,
3=bottom left, 4=bottom right). We can make the legend a bit prettier by
removing the frame and putting a gray background.


    legend = legend(["Low mutation rate", "High Mutation rate"], loc=4);
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')

Ticks:
======
to remove some ticks, to lighten the figure:
    xticks(np.arange(0, 500, 100))


Last, we can make the figure a bit lighter by removing the frame and adding a
light grid:
# put this _before_ the calls to plot and fill_between
axes(frameon=0)
grid()

Text:
====
plt.text(-9, 100, r'$\mu$ = 0.0, $\sigma$ = 2.0', size=16)

"""


def figsize(scale):
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(
        5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


params = {
    'pgf.texsystem': 'pdflatex',
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Palatino',
    # 'font.serif': [],
    # blank entries should cause plots to inherit fonts from the document
    # 'font.sans-serif': [],
    # 'font.monospace': [],
    'axes.labelsize': 10,
    'font.size': 10,
    # 'text.fontsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    # 'figure.figsize': [6.5, 4.0]
    'figure.figsize': figsize(0.9),
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",
        # plots will be generated using this preamble
    ]

}


def test_plot():
    params = {
        'axes.labelsize': 8,
        'text.fontsize': 8,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'figure.figsize': [6.5, 4.0]
    }
    rcParams.update(params)


def newfig(scale):
    plt.clf()
    fig = plt.figure(figsize=figsize(scale))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filepath):
    # plt.savefig('{}.pgf'.format(filepath))
    plt.savefig('{}.pdf'.format(filepath))


def plot_ts_pub(ts, xlabel=r'\textit{time} (day)',
                ylabel=r'\textit(num of attacks)', legend='',
                title='', col='blue', lstyle='-',
                lw=2, marker='', outfile=None):
    # rcParams['figure.figsize'] = 15, 6
    rcParams.update(params)

    plt.plot(ts, color=col, label=legend, linestyle=lstyle, linewidth=lw,
             marker=marker)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    plt.legend(loc="upper left")
    # plt.title(title)
    if outfile is not None:
        plt.savefig(outfile, format="pdf")
    return plt


###########################################################################
# Time Series Plots
###########################################################################

def plot_ts(ts, xlabel='Day', ylabel='#Events', legend='',
            title='', col='blue', lstyle='-',
            lw=2, marker='', outfile=None):
    # rcParams['figure.figsize'] = 15, 6
    rcParams['figure.figsize'] = 15, 7.5
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['xtick.major.size'] = 6
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.size'] = 6
    rcParams['ytick.major.width'] = 1
    plt.plot(ts, color=col, label=legend, linestyle=lstyle, linewidth=lw,
             marker=marker)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=20)
    if outfile is not None:
        plt.savefig(outfile, format="pdf")
    return plt


def plot_true_pred(ts_true, ts_pred, xlabel='Days', ylabel='Num of Attacks',
                   leg_true='Observed', leg_pred='Predicted', lstyle='-',
                   lw=2, marker='', legend_loc="upper left",
                   legend_font_size=16):
    rcParams['figure.figsize'] = 10, 6
    plt.plot(ts_true, color='blue', label=leg_true, linestyle=lstyle,
             linewidth=lw, marker=marker)
    plt.plot(ts_pred, color='red', label=leg_pred, linestyle=lstyle,
             linewidth=lw, marker=marker)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(loc=legend_loc, prop={'size': legend_font_size},
               edgecolor="0.9", facecolor="0.9")
    return plt


"""
'best'         : 0, (only implemented for axes legends)
'upper right'  : 1,
'upper left'   : 2,
'lower left'   : 3,
'lower right'  : 4,
'right'        : 5,
'center left'  : 6,
'center right' : 7,
'lower center' : 8,
'upper center' : 9,
'center'       : 10,
"""


def plot_list_of_lines(ts_list, legends=None, xlabel='Day',
                       ylabel='Num of Attacks', lstyle='-', lw=2, marker='',
                       title="", legend_loc="upper left", legend_font_size=16):
    rcParams['figure.figsize'] = 15, 6
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['xtick.major.size'] = 6
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.size'] = 6
    rcParams['ytick.major.width'] = 1
    # import brewer2mpl
    # bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
    # colors = bmap.mpl_colors
    # colors = ['black', 'blue', 'green', 'magenta', 'red']
    # colors = ['black', 'blue', 'green', '#B22400', '#006BB2', 'magenta', 'red']
    colors = ['black', 'BrBG', 'PRGn', 'bwr', 'seismic']
    # colors = plt.cm.Set3(np.linspace(0, 1, 12))
    # colors = ['black', 'green', 'magenta', 'red', 'blue']
    # colors = ['black', 'purple', 'seagreen', 'darkred', 'mediumblue']
    # colors = ['black', 'purple', 'forestgreen', 'mediumblue', 'darkred']
    colors = ['black', 'purple', 'green', 'darkred', 'blue']
    lw = [1.5, 2, 3, 3, 3]

    linestyles = ['-', ':', '--', '-', '-.']
    markers = ['', 'o', 'd', 's', 'X']
    # markers = ['', '', '', '', '']
    markevery = [None, 2, 2, 2, 2]

    if legends is None:
        legends = []
        for i in range(len(ts_list)):
            legends.append("Line " + str(i + 1))

    # print(colors)
    # print(legends)

    for idx, ts in enumerate(ts_list):
        plt.plot(ts,
                 color=colors[idx],
                 label=legends[idx],
                 linestyle=linestyles[idx],
                 linewidth=lw[idx],
                 marker=markers[idx],
                 markevery=markevery[idx],
                 alpha=0.9)
        # markevery="stride")
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    # plt.legend(loc=legend_loc, prop={'size': legend_font_size},
    #            edgecolor="white")
    plt.legend(loc=legend_loc, prop={'size': legend_font_size},
               edgecolor="0.9", facecolor="0.9")
    plt.title(title)

    return plt
