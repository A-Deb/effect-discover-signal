import numpy as np

"""
https://stackoverflow.com/questions/24601405/why-do-statsmodelss-correlation-and-autocorrelation-functions-give-different-re


"""


def ccovf(x, y, unbiased=True, demean=True):
    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if unbiased:
        xi = np.ones(n)
        d = np.correlate(xi, xi, 'full')
    else:
        d = n
    return (np.correlate(xo, yo, 'full') / d)[n - 1:]


def ccf(x, y, unbiased=True):
    cvf = ccovf(x, y, unbiased=unbiased, demean=True)
    return cvf / (np.std(x) * np.std(y))
