# import pandas and numpy to start coding
import pandas as pd
import numpy as np

# Scaler
from sklearn.preprocessing import MinMaxScaler

# RMSE
from sklearn.metrics import mean_squared_error

# Modeling Methods
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures






def create_dummies(train, validate, test):
    
    # hot one encode for cluster column using get_dummies for train, validate, test
    df_dummies_train = pd.get_dummies(data=train.cluster, prefix='cluster', drop_first=True)
    df_dummies_validate = pd.get_dummies(data=validate.cluster, prefix='cluster', drop_first=True)
    df_dummies_test = pd.get_dummies(data=test.cluster, prefix='cluster', drop_first=True)

    # concat df_dummies with train on columns
    train = pd.concat([train, df_dummies_train], axis=1)
    validate = pd.concat([validate, df_dummies_validate], axis=1)
    test = pd.concat([test, df_dummies_test], axis=1)
    
    return train, validate, test


def X_y_versions(train, validate, test, target='logerror_abs'):
    # select columns to model
    cols = ['land_dollar_sqft','house_dollar_sqft', 'age', 'longitude','latitude','tax_rate','bed_bath_ratio', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5','logerror_abs']
    
    # create X & y version of train, validate, test with y the target and X are the features. 
    X_train = train[cols].drop(columns=[target])
    y_train = train[[target]]

    X_validate = validate[cols].drop(columns=[target])
    y_validate = validate[[target]]

    X_test = test[cols].drop(columns=[target])
    y_test = test[[target]]
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test


def X_scaled_fe(X_train, X_validate, X_test):
    
    # Create the scale container
    scaler = MinMaxScaler()

    # Fit the scaler to the features
    scaler.fit(X_train)

    # create scaled X versions 
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # Convert numpy array to pandas dataframe for feature Engineering
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns.to_list())
    X_validate_scaled = pd.DataFrame(X_validate_scaled, index=X_validate.index, columns=X_validate.columns.to_list())
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns.to_list())
    
    return X_train_scaled, X_validate_scaled, X_test_scaled


def model_prep(train, validate, test):
    
    train, validate, test = create_dummies(train, validate, test)
    
    X_train, X_validate, X_test, y_train, y_validate, y_test = X_y_versions(train, validate, test, target='logerror_abs')
    
    X_train_scaled, X_validate_scaled, X_test_scaled = X_scaled_fe(X_train, X_validate, X_test)
    
    return train, validate, test, X_train, X_validate, X_test, y_train, y_validate, y_test, X_train_scaled, X_validate_scaled, X_test_scaled
    
    


#-------------------------------------------------------------------



def select_k_best(X_train_scaled, y_train):
    from sklearn.feature_selection import SelectKBest, f_regression

    # Use f_regression stats test each column to find best 3 features
    f_selector = SelectKBest(f_regression, k=3)

    # find tthe best correlations with y
    f_selector.fit(X_train_scaled, y_train)

    # Creaet boolean mask of the selected columns. 
    feature_mask = f_selector.get_support()

    # get list of top K features. 
    f_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()

    return f_feature



def rfe(X_train_scaled, y_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE

    # create the ML algorithm container
    lm = LinearRegression()

    # create the rfe container with the the number of features I want. 
    rfe = RFE(lm, n_features_to_select=3)

    # fit RFE to the data
    rfe.fit(X_train_scaled,y_train)  

    # get the mask of the selected columns
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()

    return rfe_feature


#---------------------------------------------------



def model_baseline(y_train, y_validate): 

    # Add target mean column as baseline check
    y_train['mean_pred'] = y_train.logerror_abs.mean()
    y_validate['mean_pred'] = y_validate.logerror_abs.mean()

    # Create Baseline RMSE of target mean
    rmse_train = mean_squared_error(y_train.logerror_abs, y_train.mean_pred) ** .5
    rmse_validate = mean_squared_error(y_validate.logerror_abs, y_validate.mean_pred) ** .5
 
    # Create df to hold rmse values
    metric_df = pd.DataFrame(data=[
                {
                    'model': 'mean_baseline', 
                    'RMSE_train': rmse_train,
                    'RMSE_validate': rmse_validate,
                    'RMSE_test:': 'none',
                    'RMSE_diff:': rmse_train - rmse_validate
                    }
                ])
    
    return y_train, y_validate, metric_df



#-----------------------------------------------------------

def model_ols_wo_cluster(train, validate, test, y_train, y_validate, metric_df):
    
    # select columns to model without cluster columns
    cols = ['land_dollar_sqft','house_dollar_sqft', 'age', 'bed_bath_ratio', 'logerror_abs']

    # establish target column
    target = 'logerror_abs'

    # create X & y version of train, validate, test with y the target and X are the features. 
    X_train = train[cols].drop(columns=[target])

    X_validate = validate[cols].drop(columns=[target])

    X_test = test[cols].drop(columns=[target])

    # Create the scale container
    scaler = MinMaxScaler()

    # Fit the scaler to the features
    scaler.fit(X_train)

    # create scaled X versions 
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # Convert numpy array to pandas dataframe for feature Engineering
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns.to_list())
    X_validate_scaled = pd.DataFrame(X_validate_scaled, index=X_validate.index, columns=X_validate.columns.to_list())
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns.to_list())

    # create, fit, predict ols model for train and validate
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train.logerror_abs)

    # predict on train
    y_train['ols_pred_wo_cluster'] = ols.predict(X_train_scaled)

    # predict validate
    y_validate['ols_pred_wo_cluster'] = ols.predict(X_validate_scaled)
    
    # evaluate rmse of train
    rmse_train = mean_squared_error(y_train.logerror_abs, y_train.ols_pred_wo_cluster) ** .5

    # evaluate rmse of validate
    rmse_validate = mean_squared_error(y_validate.logerror_abs, y_validate.ols_pred_wo_cluster) ** .5

    # add to eval to metric holder
    metric_df = metric_df.append({
        'model': 'ols_regressor_wo_cluster', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        'RMSE_test:': 'none',
        'RMSE_diff:': rmse_train - rmse_validate
        }, ignore_index=True)

    return y_train, y_validate, metric_df



#--------------------------------------


def model_osl_w_cluster(train, validate, test, y_train, y_validate, metric_df):
    
    # select columns to model including cluster columns
    cols = ['land_dollar_sqft','house_dollar_sqft', 'age', 'bed_bath_ratio','logerror_abs', 'cluster_1', 'cluster_5']

    # establish target column
    target = 'logerror_abs'

    # create X & y version of train, validate, test with y the target and X are the features. 
    X_train = train[cols].drop(columns=[target])

    X_validate = validate[cols].drop(columns=[target])

    X_test = test[cols].drop(columns=[target])

    # Create the scale container
    scaler = MinMaxScaler()


    # Fit the scaler to the features
    scaler.fit(X_train)

    # create scaled X versions 
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # Convert numpy array to pandas dataframe for feature Engineering
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns.to_list())
    X_validate_scaled = pd.DataFrame(X_validate_scaled, index=X_validate.index, columns=X_validate.columns.to_list())
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns.to_list())

    # create, fit, predict ols model for train and validate
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train.logerror_abs)

    # predict on train
    y_train['ols_pred_w_cluster'] = ols.predict(X_train_scaled)

    # predict validate
    y_validate['ols_pred_w_cluster'] = ols.predict(X_validate_scaled)
    
    # evaluate rmse for train
    rmse_train = mean_squared_error(y_train.logerror_abs, y_train.ols_pred_w_cluster) ** .5

    # evaluate rmse for validate
    rmse_validate = mean_squared_error(y_validate.logerror_abs, y_validate.ols_pred_w_cluster) ** .5

    # add to eval to metric holder
    metric_df = metric_df.append({
        'model': 'ols_regressor_w_cluster', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        'RMSE_test:': 'none',
        'RMSE_diff:': rmse_train - rmse_validate
        }, ignore_index=True)

    return y_train, y_validate, metric_df


#----------------------------------------------


def model_osl_w_cluster_more_features(train, validate, test, y_train, y_validate, metric_df):
    
    # select columns to model including cluster columns
    cols = ['land_dollar_sqft','house_dollar_sqft', 'age', 'bed_bath_ratio','logerror_abs', 'cluster_1', 'cluster_5', 'longitude', 'latitude','tax_rate']

    # establish target column
    target = 'logerror_abs'

    # create X & y version of train, validate, test with y the target and X are the features. 
    X_train = train[cols].drop(columns=[target])

    X_validate = validate[cols].drop(columns=[target])

    X_test = test[cols].drop(columns=[target])

    # Create the scale container
    scaler = MinMaxScaler()


    # Fit the scaler to the features
    scaler.fit(X_train)

    # create scaled X versions 
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # Convert numpy array to pandas dataframe for feature Engineering
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns.to_list())
    X_validate_scaled = pd.DataFrame(X_validate_scaled, index=X_validate.index, columns=X_validate.columns.to_list())
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns.to_list())

    # create, fit, predict ols model for train and validate
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train.logerror_abs)

    # predict on train
    y_train['ols_pred_w_cluster_and_features'] = ols.predict(X_train_scaled)

    # predict validate
    y_validate['ols_pred_w_cluster_and_features'] = ols.predict(X_validate_scaled)
    
    # evaluate rmse for train
    rmse_train = mean_squared_error(y_train.logerror_abs, y_train.ols_pred_w_cluster_and_features) ** .5

    # evaluate rmse for validate
    rmse_validate = mean_squared_error(y_validate.logerror_abs, y_validate.ols_pred_w_cluster_and_features) ** .5

    # add to eval to metric holder
    metric_df = metric_df.append({
        'model': 'ols_regressor_w_cluster_and_features', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        'RMSE_test:': 'none',
        'RMSE_diff:': rmse_train - rmse_validate
        }, ignore_index=True)

    return y_train, y_validate, metric_df


#------------------------------------------------------------



def model_ployreg_w_cluster_more_features(train, validate, test, y_train, y_validate, X_train_scaled, X_validate_scaled, X_test_scaled, metric_df):
    
    # create polynomial features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled to new sets
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 =  pf.transform(X_test_scaled)

    # create the model object
    ols2 = LinearRegression()

    # fit the model train data. Specify y_train columns since it was converted to dataframe  
    ols2.fit(X_train_degree2, y_train.logerror_abs)

    # predict train
    y_train['pr_pred'] = ols2.predict(X_train_degree2)

    # predict validate
    y_validate['pr_pred'] = ols2.predict(X_validate_degree2)
    
    # create rmse train
    rmse_train = mean_squared_error(y_train.logerror_abs, y_train.pr_pred) ** .5

    # evaluate rmse validate
    rmse_validate = mean_squared_error(y_validate.logerror_abs, y_validate.pr_pred) ** .5

    # add to metric holder
    metric_df = metric_df.append({
        'model': 'polyn_reg_cluster_and_features', 
        'RMSE_train': rmse_train,
        'RMSE_validate': rmse_validate,
        'RMSE_test:': 'none',
        'RMSE_diff:': rmse_train - rmse_validate
        }, ignore_index=True)

    return y_train, y_validate, X_test_degree2, ols2, rmse_train, metric_df




def test_polyreg(X_test_degree2, y_test, ols2, rmse_train, metric_df):
    
    # predict train
    y_test['pr_pred'] = ols2.predict(X_test_degree2)

    # create rmse
    rmse_test = mean_squared_error(y_test.logerror_abs, y_test.pr_pred) ** .5

    # add to metric holder
    metric_df = metric_df.append({
        'model': 'test_poly_reg_w_cluster_and_feature', 
        'RMSE_train': 'none',
        'RMSE_validate': 'none',
        'RMSE_test:': rmse_test,
        'RMSE_diff:': rmse_train - rmse_test
        }, ignore_index=True)

    return y_test, metric_df
































