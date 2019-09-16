#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from plotting import plot_config

tableau20 = plot_config.tableau_20_colors()
tableau10 = plot_config.tableau_10_colors()


def plot_ts_lines(df, majors):
    """
    Each column is a time series and the index the time 
    :param df: 
    :return: 
    """
    p, fig = get_desired_plot()
    # colorpalette = tableau10
    colorpalette = tableau20

    ymin = df.min()[0]
    ymax = df.max()[0]
    # xmin = df.index.min()
    xmin = df.index.min() - pd.Timedelta(days=2)
    # xmax = df.index.max() + pd.Timedelta(days=3)
    xmax = df.index.max()
    ytickfreq = 25

    p.ylim(ymin, ymax)
    p.xlim(xmin, xmax)

    p.yticks(range(0, ymax + 1, ytickfreq),
             [str(x) for x in range(0, ymax + 1, ytickfreq)],
             fontsize=14)
    p.xticks(fontsize=14)

    for y in range(ytickfreq, ymax + 1, ytickfreq):
        p.plot(df.index, [y] * len(df.index), "--", lw=0.5,
               color="black", alpha=0.3)

    p.tick_params(axis="both", which="both", bottom="on", top="off",
                  labelbottom="on", left="off", right="off", labelleft="on")

    for rank, column in enumerate(majors):
        # Plot each line separately with its own color, using the Tableau 20
        # color set in order.
        # key = column.lower().replace("\n", " ")[0]
        key = column
        p.plot(df.index.values, df[key].values, lw=2.5,
               color=colorpalette[rank])
        print(key)
        y_pos = df[key].values[-1] - 0.5
        # if column == "A":
        #     y_pos += 0.5
        # elif column == "B":
        #     y_pos -= 0.5
        p.text(xmax, y_pos, column, fontsize=14, color=colorpalette[rank])

    return p, fig


def plot_lines_df(df):
    p, fig = get_desired_plot()

    # Limit the range of the plot to only where the data is.
    # Avoid unnecessary whitespace.
    p.ylim(0, 90)
    p.xlim(1968, 2014)

    # Make sure your axis ticks are large enough to be easily read.
    # You don't want your viewers squinting to read your plot.
    p.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)],
             fontsize=14)
    p.xticks(fontsize=14)

    # Provide tick lines across the plot to help your viewers trace along
    # the axis ticks. Make sure that the lines are light and small so they
    # don't obscure the primary data lines.
    for y in range(10, 91, 10):
        p.plot(range(1968, 2012), [y] * len(range(1968, 2012)), "--", lw=0.5,
               color="black", alpha=0.3)

    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
    p.tick_params(axis="both", which="both", bottom="off", top="off",
                  labelbottom="on", left="off", right="off", labelleft="on")

    # Now that the plot is prepared, it's time to actually plot the data!
    # Note that I plotted the majors in order of the highest % in the final year.
    majors = ['Health Professions', 'Public Administration', 'Education',
              'Psychology',
              'Foreign Languages', 'English', 'Communications\nand Journalism',
              'Art and Performance', 'Biology', 'Agriculture',
              'Social Sciences and History', 'Business', 'Math and Statistics',
              'Architecture', 'Physical Sciences', 'Computer Science',
              'Engineering']

    for rank, column in enumerate(majors):
        # Plot each line separately with its own color, using the Tableau 20
        # color set in order.
        p.plot(df.Year.values, df[column.replace("\n", " ")].values,
               lw=2.5, color=tableau20[rank])

        # Add a text label to the right end of every line. Most of the code below
        # is adding specific offsets y position because some labels overlapped.
        y_pos = df[column.replace("\n", " ")].values[-1] - 0.5
        if column == "Foreign Languages":
            y_pos += 0.5
        elif column == "English":
            y_pos -= 0.5
        elif column == "Communications\nand Journalism":
            y_pos += 0.75
        elif column == "Art and Performance":
            y_pos -= 0.25
        elif column == "Agriculture":
            y_pos += 1.25
        elif column == "Social Sciences and History":
            y_pos += 0.25
        elif column == "Business":
            y_pos -= 0.75
        elif column == "Math and Statistics":
            y_pos += 0.75
        elif column == "Architecture":
            y_pos -= 0.75
        elif column == "Computer Science":
            y_pos += 0.75
        elif column == "Engineering":
            y_pos -= 0.25

            # Again, make sure that all labels are large enough to be easily read
        # by the viewer.
        p.text(2011.5, y_pos, column, fontsize=14, color=tableau20[rank])

    # matplotlib's title() call centers the title on the plot, but not the graph,
    # so I used the text() call to customize where the title goes.

    # Make the title big enough so it spans the entire plot, but don't make it
    # so big that it requires two lines to show.

    # Note that if the title is descriptive enough, it is unnecessary to include
    # axis labels; they are self-evident, in this plot's case.
    p.text(1995, 93,
           "Percentage of Bachelor's degrees conferred to women in the U.S.A."
           ", by major (1970-2012)", fontsize=17, ha="center")

    # Always include your data source(s) and copyright notice! And for your
    # data sources, tell your viewers exactly where the data came from,
    # preferably with a direct link to the data. Just telling your viewers
    # that you used data from the "U.S. Census Bureau" is completely useless:
    # the U.S. Census Bureau provides all kinds of data, so how are your
    # viewers supposed to know which data set you used?
    p.text(1966, -8,
           "Data source: nces.ed.gov/programs/digest/2013menu_tables.asp"
           "\nAuthor: Randy Olson (randalolson.com / @randal_olson)"
           "\nNote: Some majors are missing because the historical data "
           "is not available for them", fontsize=10)

    # Finally, save the figure as a PNG.
    # You can also save it as a PDF, JPEG, etc.
    # Just change the file extension in this call.
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.
    filepath = "fig/percent-bachelors-degrees-women-usa.png"
    print("Saving:", filepath)
    p.savefig(filepath, bbox_inches="tight")


def get_desired_plot(size=(15, 7.5)):
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare
    # exception because of the number of lines being plotted on it.
    # Common sizes: (10, 7.5) and (12, 9)
    # fig = plt.figure(figsize=(12, 14))
    fig = plt.figure(figsize=size)

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

    return plt, fig


def plot_ts(ts, xlabel='Day', ylabel='Count', legend='',
            title='', col='blue', lstyle='-',
            lw=2, marker='', outfile=None, outfilefrmt="png",
            scale=1.5):
    plot_config.set_rcparam(rcParams, scale)
    plt.plot(ts, color=col, label=legend, linestyle=lstyle, linewidth=lw,
             marker=marker)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title)
    if outfile is not None:
        plt.savefig(outfile, format=outfilefrmt)
    plt.tight_layout()
    return plt


def plot_lines_w_pandas(ts_list, legends=None, xlabel='Day',
                        ylabel='Num of Attacks', lstyle='-', lw=2, marker='',
                        title="", legend_loc="upper left", legend_font_size=16):
    plot_config.set_rcparam(rcParams)

    colors = ['black', 'purple', 'green', 'darkred', 'blue']
    lw = [1.5, 2, 2, 2, 2]
    linestyles = ['-', ':', '--', '-', '-.']
    markers = ['', 'o', 'd', 's', 'X']
    markevery = [None, 2, 2, 2, 2]


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
    colors = ['black', 'blue', 'purple', 'green', 'darkred']
    lw = [1.5, 2, 3, 3, 3]

    linestyles = ['-', '-', '--', '-', '-.']
    # markers = ['', 'o', 'd', 's', 'X']
    markers = ['', '', '', '', '']
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


def main(argv):
    gender_degree_data = pd.read_csv(
        "http://www.randalolson.com/wp-content/uploads/percent-bachelors-degrees-women-usa.csv")
    # print(gender_degree_data.describe())
    # print(gender_degree_data.shape)
    # print(gender_degree_data.columns)

    # fig = plot_lines_df(gender_degree_data)
    # fig.show()

    # generate a set of random time series data
    np.random.seed(123456)
    ncol = 2
    nrow = 50
    cols = ['a', 'b']
    df = pd.DataFrame(np.random.randint(low=0, high=100, size=(nrow, ncol)),
                      columns=cols,
                      index=pd.date_range("2017-01-01", periods=nrow))

    majors = ['Argentian', 'Bangladesh']
    p, f = plot_ts_lines(df, majors)
    p.show()


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
