# import pandas and numpy to start coding
import pandas as pd
import numpy as np




def filter_zillow(df):
    """
    Function removes propeties that have no bedrooms and no bathrooms and too small of an area
    """
    # remove propeties that have no bedrooms and no bathrooms and too small of an area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & (df.unitcnt <= 1) | df.unitcnt.isna() & (df.calculatedfinishedsquarefeet > 500) & (df.bedroomcnt > 0) & (df.bathroomcnt > 0)]

    return df


def handle_nulls(df, percent_required_cols = .5, percent_required_rows = .7):
    """
    - Drops column if it has more than 50% nulls. Drops row it has more than 30% nulls.
    - Drops columsn either no longer needed, not useful, or duplicate.
    - Replaced heating system description with None
    - Drops all other nulls
    """
    
    # set threshold for min of values in columns for dropping
    thresh_col = int(round(percent_required_cols * df.shape[0]))
    
    # drop columns that don't meed threshhold for non-null values (rows without nulls)
    df = df.dropna(axis=1, thresh=thresh_col)
    
    # set threshold for min non-null values for rows (cols without nulls)
    thresh_row = int(round(percent_required_rows * df.shape[1]))
    
    # drop rows with don't meet threshold for non-null values for columns
    df = df.dropna(axis=0, thresh=thresh_row)
    
    # remove columns that are not useful
    df = df.drop(columns=[
        # uniquie identifer to lot
        'parcelid',
        # uniquie identifer for table        
        'id',
         #Description of the allowed land uses (zoning) for that property
         'propertyzoningdesc', 
         # Finished living area
         'finishedsquarefeet12',
         #  Census tract and block ID combined - also contains blockgroup assignment by extension
             'censustractandblock',
          # Type of land use the property is zoned for
          'propertylandusetypeid',
          #  Type of home heating system
          'heatingorsystemtypeid',
        # unit type cnt: filtered out to only 1 cnt, no longer needed
        'unitcnt',
        # Census tract and block ID combined, not needed
        'rawcensustractandblock',
        # year assessed
        'assessmentyear',
        # date of transaction
        'transactiondate',
        #  Number of bathrooms in home including fractional bathroom. duplicate from bathroomcnt
        'calculatedbathnbr',
        #  Total number of rooms in the principal residence. Not collected for LA County
        'roomcnt',
        # descirpiton of land use (single family), no longer needed
        'propertylandusedesc',
        # duplicate id column
        'id.1'

        ])
    
    # relacing nulls with 'None', assuming null was for not having a heating system
    df.heatingorsystemdesc.fillna('None', inplace=True)
    
    # droping buildingqualitytypeid because they are not collected for Ventura and Orange
    df.drop(columns=['buildingqualitytypeid'], inplace=True)
    
    # dropping the rest of the nulls
    df = df.dropna()
    
    return df







def remove_outliers(df):
    """
    Function drops outliers that are above and below first and third quartile by 1.5 times the interquartile range
    """
    # List of columns to filter outliers
    cols = [col for col in df.columns.drop(['bedroomcnt',
                                            'fips',
                                            'propertycountylandusecode',
                                            'heatingorsystemdesc'
                                           ])]

    for col in cols:
        # get 1st and 3rd quartile
        q1, q3 = df[col].quantile([.25, .75])
        
        # set Interquartile Range
        iqr = q3 - q1

        # Set upper and lower bounderies
        upper_bound = q3 + 1.5 * iqr   # get upper bound
        lower_bound = q1 - 1.5 * iqr   # get lower bound

        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
        return df
    
    
    
    
def fix_cols(df):
    """
    Function renames columns for readability and creates new columns for dimension reduction
    """
    
    # rename for readability
    df = df.rename(columns = {'bathroomcnt':'bathrooms',
     'bedroomcnt':'bedrooms',
     'calculatedfinishedsquarefeet':'house_area',
     'fullbathcnt':'full_baths',
     'lotsizesquarefeet':'lot_area',
     'propertycountylandusecode':'land_use_code',
     'regionidcity':'city_id',
     'regionidcounty':'county_id',
     'regionidzip':'zip_id',
     'yearbuilt':'year_built',
     'structuretaxvaluedollarcnt':'tax_value_house',
     'taxvaluedollarcnt':'tax_value_total',
     'landtaxvaluedollarcnt':'tax_value_land',
     'taxamount':'tax_amount',
     'heatingorsystemdesc':'heating_sys',
    })



    # create column for age of house
    df['age'] = 2017 - df.year_built


    # Create column for absolute value of logerror
    df['logerror_abs'] = df.logerror.abs()


    # create columsn to match fips to county(LA, Orange County, Ventury County)
    df['county'] = df.fips.map({6037:'LA',
                6059:'OC',
                6111:'Ventura'})


    # create column for Dollar per square foot
    df['house_dollar_sqft'] = df.tax_value_total / df.house_area


    # Create column for land dollar per sqft
    df['land_dollar_sqft'] = df.tax_value_land / df.lot_area


    # create column for tax rate
    df['tax_rate'] = df.tax_value_house / df.tax_amount


    # create column for bed bath ratio
    df['bed_bath_ratio'] = df.bedrooms / df.bathrooms
    
    return df


def split(df):
    """
    Function splits data into 3 sets: train, vaidate, and test
    """
    from sklearn.model_selection import train_test_split
    
    # crerte tain, validate, test data sets
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test






def scale_cols(train, validate, test):
    """
    Function creates datafram of columns used to scale for clustering and modeling
    """
    # Scaler
    from sklearn.preprocessing import MinMaxScaler
    
    # select columns to cluster and explore
    cols = ['house_area', 'lot_area', 'tax_value_house', 'tax_value_total',
           'tax_value_land', 'age',
           'logerror_abs','house_dollar_sqft', 'land_dollar_sqft',
           'tax_rate', 'bed_bath_ratio', 'fips']

    # Columns used for further exploration and clustering
    train_cols = train[cols]
    validate_cols = validate[cols]
    test_cols = test[cols]

    # Make the scaler with MinMax
    scaler = MinMaxScaler()

    # Fit the scalter to X_train
    scaler.fit(train_cols)

    # Transform train, validate, test to scaled version
    train_scaled = scaler.transform(train_cols)
    validate_scaled = scaler.transform(validate_cols)
    test_scaled = scaler.transform(test_cols)

    # Make the scaled arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=train_cols.columns, index=train_cols.index)
    validate_scaled = pd.DataFrame(validate_scaled, columns=validate_cols.columns, index=validate_cols.index)
    test_scaled = pd.DataFrame(test_scaled, columns=test_cols.columns, index=test_cols.index)
    
    return train_scaled, validate_scaled, test_scaled



def wrangle(df):
    """
    Function merges all prepare functions into one
    """
    # removes propeties that have no bedrooms and no bathrooms and too small of an area
    df = filter_zillow(df)
    
    # removes columns and rows that meet null percent values
    df = handle_nulls(df, percent_required_cols = .5, percent_required_rows = .7)
    
    # remove outliers that 1.5 times IQR above or below first or third quartile
    df = remove_outliers(df)
    
    # Renames columns and add columns for dimension reduction
    df = fix_cols(df)
    
    # split data for train, validate, test
    train, validate, test = split(df)
    
    #scale train, valiate, test
    train_scaled, validate_scaled, test_scaled = scale_cols(train, validate, test)
    
    return df, train, validate, test, train_scaled, validate_scaled, test_scaled
    
    
    













































    
    
    
