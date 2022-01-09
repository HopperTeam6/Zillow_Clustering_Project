import pandas as pd
import numpy as np






def get_zillow():
    """
    Function imports MySQL Sever credential to pull Zillow data and creates a csv file if needed.
    """
    # import env file for hostname, username, password
    from env import host, user, password

    db_name = 'zillow'

    # Pass env file authentication to container 'url'
    url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

    # define sql search for all records from all tables
    sql = """
    SELECT *
    FROM properties_2017
    LEFT JOIN predictions_2017 pred USING(parcelid)
    LEFT JOIN airconditioningtype USING(airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING(buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    LEFT JOIN storytype USING(storytypeid)
    LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
    WHERE latitude IS NOT NULL 
    AND longitude IS NOT NULL
    AND propertylandusetypeid = 261
    AND transactiondate LIKE "2017%%"
    AND pred.id IN (SELECT MAX(id)
    FROM predictions_2017
    GROUP BY parcelid
    HAVING MAX(transactiondate))
    """

    # load zillow data from saved csv or pull from sql server and save to csv
    import os
    file = 'zillow_data.csv'
    if os.path.isfile(file):
        df = pd.read_csv(file, index_col=0)
    else:
        df = pd.read_sql(sql, url)
        df.to_csv(file)
    return df






def col_nulls(df):
    """
    Function creates a dataframe that shows null count, mean, and percent of each column
    """
    # create dataframe that has column name as first column
    col_nulls = pd.DataFrame()
    col_nulls['columns_name'] = df.isna().sum().index

    # create new column that hold the sum of nulls from each column
    col_nulls['row_null_count'] = df.isna().sum().values


    # create new column that hold the average of nulls from each column
    col_nulls['row_null_percent'] = df.isna().mean().values


    # sort values by percent
    col_nulls = col_nulls.sort_values(by=['row_null_percent'], ascending=False)
    
    return col_nulls








def row_nulls(df):
    """
    Function creates a dataframe that shows row null count and percnet"
    """
    # Create df with number of rows with a specific number of null columns
    row_nulls = pd.DataFrame(df.isna().sum(axis=1).value_counts(), columns=['num_rows_with_n_null_cols'])

    # make first columnb the number of nulls
    row_nulls = row_nulls.reset_index()

    # rename index to match values
    row_nulls = row_nulls.rename(columns={'index':'n_null_cols'})

    # create columsn for percent of null cols
    row_nulls['percent_null_cols'] = row_nulls.n_null_cols / df.shape[1]

    # sort df by percentn of null cols
    row_nulls = row_nulls.sort_values(by=['percent_null_cols'], ascending=False)
    
    return row_nulls
    
    
    
    