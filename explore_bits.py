import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


sns.set()

#plot_categorical_and_continuous defines 3 parameters, a dataframe to pull data from, and x variable (categorical column) and a y variable (continuous value column), and returns visualizations of these relationships.
def plot_categorical_and_continuous(df, x, y):

    #plot 3 figures and 3 different plots for visualizing categorical-continuous relationships
    plt.figure(figsize = (8, 5))
    sns.boxplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.stripplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.violinplot(x = x, y = y, data = df, palette = 'inferno_r');



def plot_variable_pairs(df, quant_vars):

    #determine k
    k = len(quant_vars)

    #set up if-conditional to see how many features are being paired
    if k == 2:

        #determine correlation coefficient
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])

        #plot relationships between continuous variables
        sns.lmplot(x = quant_vars[1], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');


    #pair 3 features
    if k == 3:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])

        #plot relationships between continuous variables
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[0], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[1], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');


    #pair 4 features
    if k == 4:
        
        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])

        #plot relationships between continuous variables
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');


    #pair 5 features
    if k == 5:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        

        #plot relationships between continuous variables
        
        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[0], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[2], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[3], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

    #pair 6 features
    if k == 6:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])

        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[0], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[1], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[3], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[4], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');


    #pair 7 features
    if k == 7:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])


        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[0], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[1], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[2], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------');

        #plot XIX
        sns.lmplot(x = quant_vars[3], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------');

        #plot XX
        sns.lmplot(x = quant_vars[4], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------');

        #plot XXI
        sns.lmplot(x = quant_vars[5], y = quant_vars[6], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');


    #pair 8 features
    if k == 8:

        #determine correlation coefficients
        corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
        corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
        corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
        corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
        corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
        corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])
        corr7, p7 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[0]])
        corr8, p8 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[1]])
        corr9, p9 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[2]])
        corr10, p10 = stats.pearsonr(df[quant_vars[4]], df[quant_vars[3]])
        corr11, p11 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[0]])
        corr12, p12 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[1]])
        corr13, p13 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[2]])
        corr14, p14 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[3]])
        corr15, p15 = stats.pearsonr(df[quant_vars[5]], df[quant_vars[4]])
        corr16, p16 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[0]])
        corr17, p17 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[1]])
        corr18, p18 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[2]])
        corr19, p19 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[3]])
        corr20, p20 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[4]])
        corr21, p21 = stats.pearsonr(df[quant_vars[6]], df[quant_vars[5]])
        corr22, p22 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[0]])
        corr23, p23 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[1]])
        corr24, p24 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[2]])
        corr25, p25 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[3]])
        corr26, p26 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[4]])
        corr27, p27 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[5]])
        corr28, p28 = stats.pearsonr(df[quant_vars[7]], df[quant_vars[6]])


        #plot relationships between continuous variables

        #plot I
        sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr1, 3)} | P-value: {round(p1, 4)} \n -----------------');

        #plot II
        sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr2, 3)} | P-value: {round(p2, 4)} \n -----------------');

        #plot III
        sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr3, 3)} | P-value: {round(p3, 4)} \n -----------------');

        #plot IV
        sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr4, 3)} | P-value: {round(p4, 4)} \n -----------------');

        #plot V
        sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr5, 3)} | P-value: {round(p5, 4)} \n -----------------');

        #plot VI
        sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr6, 3)} | P-value: {round(p6, 4)} \n -----------------');

        #plot VII
        sns.lmplot(x = quant_vars[4], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr7, 3)} | P-value: {round(p7, 4)} \n -----------------');

        #plot VIII
        sns.lmplot(x = quant_vars[4], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr8, 3)} | P-value: {round(p8, 4)} \n -----------------');

        #plot IX
        sns.lmplot(x = quant_vars[4], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr9, 3)} | P-value: {round(p9, 4)} \n -----------------');

        #plot X
        sns.lmplot(x = quant_vars[4], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr10, 3)} | P-value: {round(p10, 4)} \n -----------------');

        #plot XI
        sns.lmplot(x = quant_vars[5], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr11, 3)} | P-value: {round(p11, 4)} \n -----------------');

        #plot XII
        sns.lmplot(x = quant_vars[5], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr12, 3)} | P-value: {round(p12, 4)} \n -----------------');

        #plot XIII
        sns.lmplot(x = quant_vars[5], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr13, 3)} | P-value: {round(p13, 4)} \n -----------------');

        #plot XIV
        sns.lmplot(x = quant_vars[5], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr14, 3)} | P-value: {round(p14, 4)} \n -----------------');

        #plot XV
        sns.lmplot(x = quant_vars[5], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr15, 3)} | P-value: {round(p15, 4)} \n -----------------');

        #plot XVI
        sns.lmplot(x = quant_vars[6], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr16, 3)} | P-value: {round(p16, 4)} \n -----------------');

        #plot XVII
        sns.lmplot(x = quant_vars[6], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr17, 3)} | P-value: {round(p17, 4)} \n -----------------');

        #plot XVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[2], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr18, 3)} | P-value: {round(p18, 4)} \n -----------------')

        #plot XIX
        sns.lmplot(x = quant_vars[6], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr19, 3)} | P-value: {round(p19, 4)} \n -----------------')

        #plot XX
        sns.lmplot(x = quant_vars[6], y = quant_vars[4], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr20, 3)} | P-value: {round(p20, 4)} \n -----------------')

        #plot XXI
        sns.lmplot(x = quant_vars[6], y = quant_vars[5], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr21, 3)} | P-value: {round(p21, 4)} \n -----------------');

        #plot XXII
        sns.lmplot(x = quant_vars[0], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr22, 3)} | P-value: {round(p22, 4)} \n -----------------');

        #plot XXIII
        sns.lmplot(x = quant_vars[1], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr23, 3)} | P-value: {round(p23, 4)} \n -----------------');

        #plot XXIV
        sns.lmplot(x = quant_vars[2], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr24, 3)} | P-value: {round(p24, 4)} \n -----------------');

        #plot XXV
        sns.lmplot(x = quant_vars[3], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr25, 3)} | P-value: {round(p25, 4)} \n -----------------');

        #plot XXVI
        sns.lmplot(x = quant_vars[4], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr26, 3)} | P-value: {round(p26, 4)} \n -----------------');

        #plot XXVII
        sns.lmplot(x = quant_vars[5], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr27, 3)} | P-value: {round(p27, 4)} \n -----------------');

        #plot XXVIII
        sns.lmplot(x = quant_vars[6], y = quant_vars[7], data = df, line_kws = {'color': 'purple'})
        plt.title(f'R-value: {round(corr28, 3)} | P-value: {round(p28, 4)} \n -----------------');
    

#<^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^> CUSTOMIZED RETURNS ON STATS TESTS FOUND HERE <^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^><^>#




#return_chi2 defines one parameter, an observed cross-tabulation, runs the stats.chi2_contingency function and returns the test results in a readable format.
def return_chi2(observed):
    
    #run the test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    #print the rest
    print('Observed')
    print('--------')
    print(observed.values)
    print('===================')
    print('Expected')
    print('--------')
    print(expected.astype(int))
    print('===================')
    print(f'Degrees of Freedom: {degf}')
    print('===================')
    print('Chi^2 and P')
    print('-----------')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p = {p:.4f}')

