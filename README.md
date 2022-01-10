# Hopper Clustering Project by Kan and Mason

## About the Project 

### Project Goals

The goal of this project is to discovering drivers for Zillow Zestimate Error by using clustering. It is the hope of the project that clustering features will decrease the RMSE for regression modeling the logerror.

### Project Description

By understanding the drivers for the Zestimate error, Zillow models can be adjusted to increase accuracy of house values. This project will explore clusters of data in order to determine features that drive Zestimate error in order to improved the accuracy of current regression models. 

### Initial Questions

Which county has the most log error?

Which tax rate has the most log error?

What combinatin of bedroom and bathroom has the most log error?

What tax value has the most log error?

Is log error associated to house size?

Is log error associated to lot size?

### Data Dictionary

Provided in Repo

### Steps to Reproduce

1. Clone the repo. ensure to have all modules. Link: https://github.com/mason-kan/Zillow_Clustering_Project

2. Confirm .gitignore is hiding your env.py file so it won't get pushed to GitHub 

3. You will need an env.py file that contains the hostname, username and password of the mySQL server that contains the zillow database and tables. Store that env file locally in the repository.

4. Run the Report Notebook

### The Plan

1. Aquire
- Pull Zillow Data
- Data to include transaction from 2017 for single family homes

2. Clean
- Revove duplicates
- Remove nulls buy remove columns, rows, or imputing values
- Remove outliers

3. Prepare
- Scale Data
- Create new features
- Split Data

4. Explore scaled and unscaled data
- Visualize log error
- visulisize multivariant interaction with log error
- Use stats test
- Create new features


5. Cluster
- creat 3 different combination of clusters based on explore findings

6. Explore Clusters
- visualize multivariant interaction by clusters
- identify clusters that differentiate from others

7. Modeling
- use regression modeling with cluster to improve the RMSE value

8. Evaluate
- visilize the difference in RMSE  from different models
- use the test dataset to confirm best model

9. Present
- create video for 5 minute presentation
- Key Findings of only the most important info of Zestimate error drivers

#### Wrangle

##### Modules (acquire.py + prepare.py + clustering.py + modeling.py + viz.py )
 
1. write code to pull and filter day via SQL
2. test acquire function and add to acquire.py module
3. write code to clean data by removing nulls, outliers, and useless columns. Add/rename columsn, split and scale data.
4. test functions, merge into wrangle function, add to prepare.py
5. write code code to create cluster combinations, elbow test, t test.
6. test code and odd to cluster.
7. write code to create dummies from cluster column, create X_y versions, scale X_y
8. merge functions, test new function, add to modeling.py
9. write code for feature selction using select k best and rfe. write code to create models for baseline, models 1 - 4
10. test functions and add to modeling.py
11. write code to visualize predictions and actual values.
12. test function and add to viz.py
13. import all moduels into report and test

##### Missing Values (report.ipynb)

- *Decisions made and reasons are communicated and documented for handling missing values.*

- *(later projects) If you imputed based on computing a value (such as mean, median, etc), that was done after splitting the data, and the value was derived from the training dataset only and then imputed into all 3 datasets. If you filled missing values with 0 or a constant not derived from existing values, that can be done prior to splitting the data.*

For example: 

1. Removed columsn with 50% null values
2. Remove rows with 30% null values
3. Removed outliers that were 1.5 times Interquartile Range above third quartile or below first quartile
4. Removed all other nulls because counts were relatively small

##### Data Split (prepare.py (def function), report.ipynb (run function))

1. Split the data into train, test, validate using 56%, 24%, 20% ratio for each set

##### Using your modules (report.ipynb)

1. Created acquire and prepare modules to call function to pull, clean, split, and scale data.

#### Explore

##### Ask a clear question, [discover], provide a clear answer (report.ipynb)

* Do any of the features correlate with the target variable?
* Is log error concentrated in any one area?
* Is log error higher in any county?


##### Exploring through visualizations (report.ipynb)

* Do any of the features correlate with the target variable?
plot correations using feature shouse area and logerro 

* Is log error concentrated in any one area?
plot log error using scatter and hue by logerror

* Is log error higher in any county?
plot log error using strip plot and seperate by county

##### Statistical tests (report.ipynb)

- use leven's test to check variance
- use t test to check if log error average is greatest in LA county

##### Summary (report.ipynb)

- log erro correlation exist with house area, age, dollar per sqft of land, and bed to bath ratio

#### Modeling

##### Select Evaluation Metric (Report.ipynb)

- compared models by using RMSE

##### Evaluate Baseline (Report.ipynb)

- baseline was set for mean of logerror abs

##### Develop 3 Models (Report.ipynb)

- the models were created.
- model 1: used ols with only four features, no cluster. features: age, dollar per sqft house, dollar per sqft land, bed to bath ratio
- model 2: used ols with same previous features plus cluster dummies
- model 3: used ols with same previous feature plus longitude, latitude, and tax rate
- model 3: used poly reg with same previous features

#### Evaluate on Train (Report.ipynb)

- each model did slightly better than the previous. 

##### Evaluate on Validate (Report.ipynb)
 
- confirmed non under/over fit with vaidate

##### Evaluate Top Model on Test (Report.ipynb)

- model 4: poly reg peformed consistent with new data

## Report (Final Notebook) 

#### code commenting (Report.ipynb)

- comments made

#### markdown (Report.ipynb)

- markdown made

#### Written Conclusion Summary (Report.ipynb)

- summary included logerror drivers and models needs improvement

#### conclusion recommendations (Report.ipynb)

- recommendation is to not use models due to needed improvement

#### conclusion next steps (Report.ipynb)

- investigate drivers of logerror in LA. create more columsn to reduce dimensions.

#### no errors (Report.ipynb)

- no errors in final

## Video Presentation
