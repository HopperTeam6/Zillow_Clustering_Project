import warnings
warnings.filterwarnings('ignore')
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import mason_functions as mf






def calculate_column_nulls(df):
    
    #set an empty list
    output = []
    
    #gather columns
    df_columns = df.columns.to_list()
    
    #commence for loop
    for column in df_columns:
        
        #assign variable to number of rows that have null values
        missing = df[column].isnull().sum()
        
        #assign variable to ratio of rows with null values to overall rows in column
        ratio = missing / len(df)
    
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100, 2)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    column_nulls = pd.DataFrame(output, index = df_columns)
    
    #return the dataframe
    return column_nulls


def calculate_row_nulls(df):
    
    #create an empty list
    output = []
    
    #gather values in a series
    nulls = df.isnull().sum(axis = 1)
    
    #commence 4 loop
    for n in range(len(nulls)):
        
        #assign variable to nulls
        missing = nulls[n]
        
        #assign variable to ratio
        ratio = missing / len(df.columns)
        
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    row_nulls = pd.DataFrame(output, index = df.index)
    
    #return the dataframe
    return row_nulls



def handle_nulls(df, cr, rr):
    '''
    This function defines 3 parameters, the dataframe you want to calculate nulls for (df), the column ratio (cr) and the row ratio (rr),
    ratios that define the threshold of having too many nulls, and returns your dataframe with columns and rows dropped if they are above their respective threshold ratios.
    Note: This function calculates the ratio of nulls missing for rows AFTER it drops the columns with null ratios above the cr threshold.
    TL; DR: This function handles nulls for dataframes.
    '''
    #set an empty list
    output = []
    
    #gather columns
    df_columns = df.columns.to_list()
    
    #commence for loop
    for column in df_columns:
        
        #assign variable to number of rows that have null values
        missing = df[column].isnull().sum()
        
        #assign variable to ratio of rows with null values to overall rows in column
        ratio = missing / len(df)
    
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100, 2)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    column_nulls = pd.DataFrame(output, index = df_columns)

    #set a list of columns to drop
    null_and_void = []
    
    #commence loop
    for n in range(len(column_nulls)):
        
        #set up conditional to see if the ratio of nulls to total columns is over the entered column ratio (.cr)
        if column_nulls.iloc[n].null_ratio > cr:
            
            #add columns over the threshold with nulls to the list of columns to drop
            null_and_void.append(column_nulls.index[n])
    
    #drop the columns
    df = df.drop(columns = null_and_void)

    #create another list
    output = []
    
    #gather values in a series
    nulls = df.isnull().sum(axis = 1)
    
    #commence 4 loop
    for n in range(len(nulls)):
        
        #assign variable to nulls
        missing = nulls[n]
        
        #assign variable to ratio
        ratio = missing / len(df.columns)
        
        #assign a dictionary for your dataframe to accept
        r_dict = {'nulls': missing,
                  'null_ratio': round(ratio, 5),
                  'null_percentage': f'{round(ratio * 100)}%'
                 }
        #add dictonaries to list
        output.append(r_dict)
        
    #design frame
    row_nulls = pd.DataFrame(output, index = df.index)

    #set an empty index of rows to drop
    ice_em = []

    #commence loop
    for n in range(len(row_nulls)):
        
        #set up conditional to see if the ratio of nulls to total columns is over the entered row ratio (.rr)
        if row_nulls.iloc[n].null_ratio > rr:
            
            #add rows to index
            ice_em.append(row_nulls.index[n])
    
    #drop rows where the percentage of nulls is over the threshold
    df = df.drop(index = ice_em)

    #return the df with preferred drop parameters
    return df