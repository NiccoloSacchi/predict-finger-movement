# -*- coding: utf-8 -*-
""" Few utility functions. """

import matplotlib.pylab as plt

def plot_scores(param, paramName, tr_scores, va_scores, ylog_scale=False):
    plt.figure(figsize=(7, 5))
    plt.grid()
    plt.plot(param, tr_scores)
    plt.plot(param, va_scores)
    plt.legend(['train', 'validation'])
    plt.xlabel(paramName)
    plt.ylabel('Accuracy')
    if ylog_scale:
        plt.xscale('log')