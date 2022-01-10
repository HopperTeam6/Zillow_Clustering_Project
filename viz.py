# import pandas and numpy to start coding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_predictions(y_validate):
    plt.figure(figsize=(16,8))

    #Basline
    plt.plot(y_validate.logerror_abs, y_validate.mean_pred, alpha=.5, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (0,.034))

    # Ideal Line
    plt.plot(y_validate.logerror_abs, y_validate.logerror_abs, alpha=.5, color="blue", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (0.005, 0), rotation=27)

    # Model 1: OLS without Clusters
    plt.scatter(y_validate.logerror_abs, y_validate.ols_pred_wo_cluster, 
                alpha=.5, color="red", s=10, label="Model 1: OLS wo Clusters")

    # Model 2: OlS with Clusters
    plt.scatter(y_validate.logerror_abs, y_validate.ols_pred_w_cluster, 
                alpha=.5, color="yellow", s=10, label="Model 2: OLS w Clusters")

    # Model 3: OLS with Clusters and More Features
    plt.scatter(y_validate.logerror_abs, y_validate.ols_pred_w_cluster_and_features, 
                alpha=.5, color="green", s=10, label="Model 3: OLS w Clusters and More Features")

    # Model 4: Polynomial Regresion with Clusters and More Features
    plt.scatter(y_validate.logerror_abs, y_validate.pr_pred, 
                alpha=.5, color="brown", s=10, label="Model 4: Poly Regression with Clusters and More Features")

    plt.legend()
    plt.xlabel("Actual ABS LogError Values")
    plt.ylabel("Predicted ABS LogError Values")
    # plt.title("Where are predictions more extreme? More modest?")
    # plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    # plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()
    
    
    
    
def plot_pred_actual_hist(y_validate):   
    # plot to visualize actual vs predicted. 
    plt.figure(figsize=(16,8))
    plt.hist(y_validate.logerror_abs, color='blue', alpha=.5, label="Absolute Value of Log Errors")
    plt.hist(y_validate.ols_pred_wo_cluster, color='red', alpha=.5, label="Model 1: OLS without Clusters")
    plt.hist(y_validate.ols_pred_w_cluster, color='yellow', alpha=.5, label="Model2: OLS with Clusters")
    plt.hist(y_validate.ols_pred_w_cluster_and_features, color='green', alpha=.5, label="Model 3: OLS with Cluster and More Features")
    plt.hist(y_validate.pr_pred, color='green', alpha=.5, label="Model 4: Poly Regressor with Clusters and More Features")
    plt.xlabel("ABS LogError Values")
    plt.ylabel("Number of Homes ")
    plt.title("Comparing the Distribution of Actual ABS Log Error to Distributions of Predicted ABS Log Error for Models")
    plt.legend()
    plt.show()

    
def plot_residuals(y_validate):    
    # y_validate.head()
    plt.figure(figsize=(16,8))
    plt.axhline(label="No Error")
    plt.annotate("Line of No Error", (0, -.01))
    plt.scatter(y_validate.logerror_abs, y_validate.ols_pred_wo_cluster - y_validate.logerror_abs, 
                alpha=.5, color="red", s=10, label="Model: OLS without Cluster")
    plt.scatter(y_validate.logerror_abs, y_validate.ols_pred_w_cluster - y_validate.logerror_abs, 
                alpha=.5, color="yellow", s=10, label="Mode2: OLS with Cluster")
    plt.scatter(y_validate.logerror_abs, y_validate.ols_pred_w_cluster_and_features - y_validate.logerror_abs, 
                alpha=.5, color="green", s=10, label="Model 3: OLS with Cluster and More Features")
    plt.scatter(y_validate.logerror_abs, y_validate.pr_pred - y_validate.logerror_abs, 
                alpha=.5, color="green", s=10, label="Model 4: Poly Regressor with Cluster and More Features")
    plt.legend()
    plt.xlabel("Actual Tax Value")
    plt.ylabel("Residual/Error: Predicted Tax Value - ABS LogError Values")
    # plt.title("Do the size of errors change as the actual value changes?")
    # plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    # plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()