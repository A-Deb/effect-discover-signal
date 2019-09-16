#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

__author__ = "Tozammel Hossain"
__email__ = "tozammel@isi.edu"


def get_K_max_values_with_indx(arr, k):
    """
    Note: slow; use argpartiion
    array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
    ind = np.argpartition(a, -4)[-4:]
    ind, a[ind]
    ind[np.argsort(a[ind])]
    """
    indx = np.argsort(-arr)[:k]
    return indx, arr[indx]

def get_K_min_values_with_indx(arr, k):
    "Note: slow"
    indx = np.argsort(arr)[:k]
    return indx, arr[indx]

