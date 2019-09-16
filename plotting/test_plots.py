#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def test_tsplot():
    gender_degree_data = pd.read_csv(
        "http://www.randalolson.com/wp-content/uploads/percent-bachelors-degrees-women-usa.csv")
    # print(gender_degree_data.describe())
    print(gender_degree_data.shape)
    print(gender_degree_data.columns)


def main(argv):
    test_tsplot()


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
