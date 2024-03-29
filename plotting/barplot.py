#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gen_synthetic_data():
    raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                'pre_score': [4, 24, 31, 2, 3],
                'mid_score': [25, 94, 57, 62, 70],
                'post_score': [5, 43, 23, 23, 51]}
    df = pd.DataFrame(raw_data, columns=['first_name', 'pre_score', 'mid_score',
                                         'post_score'])
    return df


def exp_barplot():
    df = gen_synthetic_data()
    # Setting the positions and width for the bars
    pos = list(range(len(df['pre_score'])))
    width = 0.25

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos,
            # using df['pre_score'] data,
            df['pre_score'],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#EE3224',
            # with label the first value in first_name
            label=df['first_name'][0])
    print(df['pre_score'])
    print(df['first_name'][0])

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos],
            # using df['mid_score'] data,
            df['mid_score'],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#F78F1E',
            # with label the second value in first_name
            label=df['first_name'][1])

    # Create a bar with post_score data,
    # in position pos + some width buffer,
    plt.bar([p + width * 2 for p in pos],
            # using df['post_score'] data,
            df['post_score'],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#FFC222',
            # with label the third value in first_name
            label=df['first_name'][2])

    # Set the y axis label
    ax.set_ylabel('Score')

    # Set the chart's title
    ax.set_title('Test Subject Scores')

    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df['first_name'])

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 8)
    plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])])

    # Adding the legend and showing the plot
    plt.legend(['Pre Score', 'Mid Score', 'Post Score'], loc='upper left')
    plt.grid()
    plt.show()


def make_barplot(data, title="", ylabel='Count', width=0.2):
    from plotting import plot_config

    tableau10 = plot_config.tableau_10_colors()

    xlabels = data['xlabel']
    barlabels = list(data.keys())
    barlabels.remove('xlabel')

    # print(xlabels)
    # print(barlabels)
    # Setting the positions and width for the bars
    pos = list(range(len(xlabels)))

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar with pre_score data,
    # in position pos,
    for indx, alabel in enumerate(barlabels):
        plt.bar([p + indx * width for p in pos],
                # using df['pre_score'] data,
                data[alabel],
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                # color='#EE3224',
                color=tableau10[indx],
                # with label the first value in first_name
                label=alabel)

    # Set the y axis label
    ax.set_ylabel(ylabel=ylabel)

    # Set the chart's title
    ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(xlabels)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)
    temp = []
    [temp.extend(data[x]) for x in barlabels]
    plt.ylim([0, max(temp)])

    # Adding the legend and showing the plot
    # plt.legend(barlabels, loc='upper left')
    plt.legend(barlabels, loc='best')
    # plt.grid()
    # plt.show()

    return plt


def main():
    exp_barplot()


if __name__ == '__main__':
    main()
