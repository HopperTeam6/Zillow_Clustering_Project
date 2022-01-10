import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

sns.set()


def display_correlations_z(df):

    '''
    This function displays 4 scatterplots, a regression line, and titles each plot with respective R coefficients and p-values from a pearson's r test.
    This functions works with the current zillow data set.
    '''

    plt.figure(figsize = (10, 9))    #create figure

    #subplot 1
    plt.subplot(221)

    x = df.house_area     # set x

    y = df.logerror    # set y

    corr, p = stats.pearsonr(x, y)    # run pearson's

    m, b = np.polyfit(x, y, 1)    # fit regression line

    plt.scatter(x, y, color = 'indianred', s = 2)    # plot relationship

    plt.plot(x, m * x + b, color = 'rebeccapurple', lw = 2)    # plot regression line

    #title with pearson's r and p-value
    plt.title(f'R-value: {round(corr, 3)} | P-value: {round(p, 4)} \n -----------------')

    plt.xlabel('House Area')    #label x-axis

    plt.ylabel('Log Error');   #label y-axis



    #subplot 2
    plt.subplot(222)

    x = df.age    # set x
    y = df.logerror    # set y

    corr, p = stats.pearsonr(x, y)    # run pearson's

    m, b = np.polyfit(x, y, 1)    # fit regression line

    plt.scatter(x, y, color = 'indianred', s = 2)    # plot relationship

    plt.plot(x, m * x + b, color = 'rebeccapurple', lw = 2)    # plot regression line

    #title with pearson's r and p-value
    plt.title(f'R-value: {round(corr, 3)} | P-value: {round(p, 4)} \n -----------------')

    plt.xlabel('House Age')    #label x-axis

    plt.ylabel('Log Error');   #label y-axis



    #subplot 3
    plt.subplot(223)
    x = df.land_dollar_sqft
    y = df.logerror
    corr, p = stats.pearsonr(x, y)
    m, b = np.polyfit(x, y, 1)
    plt.scatter(x, y, color = 'indianred', s = 2)
    plt.plot(x, m * x + b, color = 'rebeccapurple', lw = 2)
    plt.title(f'R-value: {round(corr, 3)} | P-value: {round(p, 4)} \n -----------------')
    plt.xlabel('Land Dollars per Square Foot')
    plt.ylabel('Log Error');



    #subplot 4
    plt.subplot(224)
    x = df.bed_bath_ratio
    y = df.logerror
    corr, p = stats.pearsonr(x, y)
    m, b = np.polyfit(x, y, 1)
    plt.scatter(x, y, color = 'indianred', s = 2)
    plt.plot(x, m * x + b, color = 'rebeccapurple', lw = 2)
    plt.title(f'R-value: {round(corr, 3)} | P-value: {round(p, 4)} \n -----------------')
    plt.xlabel('Bed/Bath Ratio')
    plt.ylabel('Log Error')
    plt.tight_layout();


def show_long_lat(df):
    sns.relplot(x = 'longitude',
            y = 'latitude',
            data = df,
            hue = 'logerror_abs',
            height = 10,
            aspect = 1.5,
            palette = 'inferno');


def display_strips(df):
    #juxtapose log error through viz
    plt.figure(figsize = (13, 7))
    sns.stripplot(x = 'county' , y = 'logerror_abs', data = df, palette = 'inferno', size = 1.6)
    plt.title('LA County Seems to Have a Slight Lead in Log Error', size = 16, pad = 6);



def juxtapose_distributions(C1, C2, C3):
    #visualize distribution target variable across different counties
    plt.figure(figsize = (13, 7))
    plt.hist([C1.logerror_abs, C2.logerror_abs, C3.logerror_abs],
            label = ['LA', 'OC', 'Ventura'],
            color = ['rebeccapurple', 'indianred', 'darkorange'],
            bins = 22
            )
    plt.legend()
    plt.title('Absolute Log Error Is Right-Skewed For All Counties', size = 16, pad = 6)
    plt.xlabel('Log Error (Absolute Value)', size = 13)
    plt.ylabel('Frequency', size = 13);