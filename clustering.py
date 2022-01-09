# import pandas and numpy to start coding
import pandas as pd
import numpy as np



def elbow_method(train_scaled):
    
    # Use elbow method to see if inertia values support visual exploration
    # plot inertia vs k

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k, random_state=123).fit(train_scaled).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')
        
    return plt.show()


def tt_test_1samp(train, n_clusters=5):
    # run one sample T-Test
    alpha = 0.05
    cluster_logerror = train[train.cluster==n_clusters].logerror_abs
    overall_logerror = train.logerror_abs.mean()
    t, p = stats.ttest_1samp(cluster_logerror, overall_logerror)

    return t, p


def cluster_combo_1(train_scaled):
    
    # Filter columns from train scaled
    cols = ['house_dollar_sqft', 'age']
    train_scaled_X = train_scaled[cols]


    # Move forward with k=6
    kmeans = KMeans(n_clusters = 6, random_state=123)
    kmeans.fit(train_scaled_X)

    # And assign the cluster number to a column on the dataframe
    train["cluster"] = kmeans.predict(train_scaled_X)
    
    return train



def cluster_combo_2(train_scaled):
    # Filter columsn from train scaled
    cols = ['house_dollar_sqft', 'land_dollar_sqft']
    train_scaled_X = train_scaled[cols]

    # Move forward with k=6
    kmeans = KMeans(n_clusters = 6, random_state=123)
    kmeans.fit(train_scaled_X)

    # And assign the cluster number to a column on the dataframe
    train["cluster"] = kmeans.predict(train_scaled_X)
    
    return train


def cluster_combo_3(train_scaled):
    
    # Filter columsn from train scaled
    cols = ['land_dollar_sqft', 'age']
    train_scaled_X = train_scaled[cols]

    # Move forward with k=3
    kmeans = KMeans(n_clusters = 6, random_state=123)
    kmeans.fit(train_scaled_X)

    # And assign the cluster number to a column on the dataframe
    train["cluster"] = kmeans.predict(train_scaled_X)
    
    return train


def cluster_combo_4(train_scaled):
    # Filter columsn from train scaled
    cols = ['longitude', 'latitude']
    train_scaled_X = train_scaled[cols]
    train_scaled_X.head()

    # Move forward with k=5
    kmeans = KMeans(n_clusters = 5, random_state=123)
    kmeans.fit(train_scaled_X)

    # And assign the cluster number to a column on the dataframe
    train["cluster"] = kmeans.predict(train_scaled_X)
    
    return train





