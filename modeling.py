# import pandas and numpy to start coding
import pandas as pd
import numpy as np



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


def X_y_versions(target='logerror_abs'):
    # create X & y version of train, validate, test with y the target and X are the features. 
    X_train = train[cols].drop(columns=[target])
    y_train = train[[target]]

    X_validate = validate[cols].drop(columns=[target])
    y_validate = validate[[target]]

    X_test = test[cols].drop(columns=[target])
    y_test = test[[target]]
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test


def X_scaled_fe(X_train):
    
    # Create the scale container
    scaler = sklearn.preprocessing.MinMaxScaler()

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


def model_baseline(): 
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
y_train = pd.DataFrame(y_train)
y_validate = pd.DataFrame(y_validate)
y_test = pd.DataFrame(y_test)

# Add target mean column as baseline check
y_train['mean_pred'] = y_train.logerror_abs.mean()
y_validate['mean_pred'] = y_validate.logerror_abs.mean()

# Create Baseline RMSE of target mean
rmse_train = mean_squared_error(y_train.logerror_abs, y_train.mean_pred) ** .5
rmse_validate = mean_squared_error(y_validate.logerror_abs, y_validate.mean_pred) ** .5

































