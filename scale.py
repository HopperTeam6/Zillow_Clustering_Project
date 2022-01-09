import warnings
warnings.filterwarnings('ignore')
import sklearn.preprocessing

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import mason_functions as mf








def robust_scaler(train, validate, test, quant_vars):
    
    #creation
    scaler = sklearn.preprocessing.RobustScaler()
    #fitting
    scaler.fit(train[quant_vars])

    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #transforming
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler



def max_min_scaler(train, validate, test, quant_vars):

    #creation
    scaler = sklearn.preprocessing.MinMaxScaler()
    #fitting
    scaler.fit(train[quant_vars])

    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #autobots, roll out
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler



def standard_scaler(train, validate, test, quant_vars):

    #creation
    scaler = sklearn.preprocessing.StandardScaler()
    #fitting
    scaler.fit(train[quant_vars])
    
    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #autobots do it
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler


def quantile_norm_scaler(train, validate, test, quant_vars):

    #creation
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution = 'normal')
    #fitting
    scaler.fit(train[quant_vars])
    
    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #autobots do it
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler


def quantile_uniform_scaler(train, validate, test, quant_vars):

    #creation
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution = 'uniform')
    #fitting
    scaler.fit(train[quant_vars])
    #assign new dataframes
    train_scaled = train[quant_vars]
    validate_scaled = validate[quant_vars]
    test_scaled = test[quant_vars]

    #autobots do it
    train_scaled[quant_vars] = scaler.transform(train[quant_vars])
    validate_scaled[quant_vars] = scaler.transform(validate[quant_vars])
    test_scaled[quant_vars] = scaler.transform(test[quant_vars])

    #return scaled sets and the scaler
    return train_scaled, validate_scaled, test_scaled, scaler