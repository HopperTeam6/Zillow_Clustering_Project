# import pandas and numpy to start coding
import pandas as pd
import numpy as np




def cluster_1(train_scaled, train):
    # cluster method
    from sklearn.cluster import KMeans
    
    # Filter columns from train scaled
    cols = ['house_dollar_sqft', 'age']
    train_scaled_X = train_scaled[cols]

    # Move forward with k=3
    kmeans = KMeans(n_clusters = 3, random_state=123)
    kmeans.fit(train_scaled_X)

    # And assign the cluster number to a column on the dataframe
    train["cluster"] = kmeans.predict(train_scaled_X)

    # run one sample T-Test
    alpha = 0.05
    cluster_logerror = train[train.cluster==2].logerror_abs
    overall_logerror = train.logerror_abs.mean()

    t, p = stats.ttest_1samp(cluster_logerror, overall_logerror)
    
    return t, p, train