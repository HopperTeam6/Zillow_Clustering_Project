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

##### Modules (acquire.py + clean.py + wrangle.py + prepare.py + modeling.py )
 
1. test acquire function
2. add to acquire.py module
3. write code to clean data in notebook
4. merge code into a function & test
5. write code to split data in notebook
6. merge code into a single function & test
7. merge functions in a single function & test
8. Add all 3 functions (or more) to prepare.py file
9. import into notebook and test functions

##### Missing Values (report.ipynb)

- *Decisions made and reasons are communicated and documented for handling missing values.*

- *(later projects) If you imputed based on computing a value (such as mean, median, etc), that was done after splitting the data, and the value was derived from the training dataset only and then imputed into all 3 datasets. If you filled missing values with 0 or a constant not derived from existing values, that can be done prior to splitting the data.*

For example: 

1. handle missing values for age in a way to be able to keep what is there
2. if there are columns > 70% missing then I will drop those columns. 

##### Data Split (prepare.py (def function), report.ipynb (run function))

- *When splitting data into samples, there should be 3 adequately sized samples - train, validate, and test. As a good starting point, 50%, 30%, and 20%, (or 50%, 26%, 24% for simplicity when doing the splitting) are reasonable split proportions. But that can vary depending on the number of observations you have. Test can go as low as 10% if needed.*

- *Someone should be able to run your code and get the same observations in the same samples, i.e. reproduce your split, because you set the random state to a seed.*  

- *Data should always be split prior to exploration of variable relationships.*

- *(later projects) In addition, imputers, scalers, feature elimination or selection algorithms should all be run after the split so that they are fit on train and transformed on validate and test.*

For example: 

1. Use function we have used in class, as that one seems to meet all the requirements. 

##### Using your modules (report.ipynb)

- *After creating the wrangle module(s), you want to import those into your final report so that you can use those functions you wrote to acquire and prepare your data with ease, with little clutter, and with reduced risk of running into issues when reproducing the report.*

- *The functions should be called to prepare your data (as opposed to re-writing the code of the functions in your notebook), and you should include in a markdown cell, the steps you took to prepare the data and why you made the decisions you did.*

For example:

1. once acquire.py and prepare.py are created and tested, import into final report notebook to be ready for use. 


#### Explore

##### Ask a clear question, [discover], provide a clear answer (report.ipynb)

- *At least 4 of the questions asked and answered of the data are shared in the final report notebook.*

- *You should call out questions of the data using natural language that speaks to the business stakeholders in markdown cells, ideally a header prior to the visualization or statistical test, that you then explore. This does not take the place of stating your null hypothesis/alternative hypothesis when doing a statistical test. But those hypotheses are generally for you. By writing questions that you intend to answer with visualizations and statistical tests in natural language, like ""Are office supplies leading to differences in profit in Texas?"", you are able to guide both yourself and your reader through the highlights of your analysis. You ask a question, create a visual, run a statistical test (if appropriate), and wrap it nicely with a markdown cell that contains a clear answer in layman's terms. You do all that before moving to the next question.*

For example: 

1. Was the phrase "women and children first" just a myth or did they really try to get out women and children first? If so, they would show as more likely to survive. 

2. Do those who travel first class get quicker access to life boats? 

3. Do families with small children get priority access? Even if they at the bottom of the boat in 3rd class? 

4. Did traveling alone make a difference? Did it depend on sex? What was the survival rate for women traveling alone vs. men traveling along? 

5. If families were more likely to be saved, is there a max family size where that benefit is lost? 

4. Contextual questions: Did most people die or survive? How many men/women were on the boat? How many across different classes? 


##### Exploring through visualizations (report.ipynb)

- *At least 5 visualations are included in your final report.*

- *The ones included answer a question (remember, NO is an answer) or provide necessary context (such as the distribution of the target variable). All statistical tests included in the final report should be supported with an visualization of the interaction of the variables being tested. Charts in the final report should have titles and labels that are descriptive and useful for the end user/audience/consumer of the report.*

- *All visualizations in the final report are mentioned or discussed if a verbal presentation is given.*

For example: 

1. Was the phrase "women and children first" just a myth or did they really try to get out women and children first? If so, they would show as more likely to survive. 

  - are women more likely to survive? plot barplot x-axis is sex and y-axis is survival rate
  - are children more likely to survive? bin age into 0-16, 17+, plot barplot on x-axis where y is survival rate (new variable = is_child)
  - run chi-square test sex + survival
  - run a chi-square test is_child + survival
  - run a t-test on age and survived

2. Do those who travel first class get quicker access to life boats? 

  - 

3. Do families with small children get priority access? Even if they at the bottom of the boat in 3rd class? 

4. Did traveling alone make a difference? Did it depend on sex? What was the survival rate for women traveling alone vs. men traveling along? 

5. If families were more likely to be saved, is there a max family size where that benefit is lost? 

4. Contextual questions: Did most people die or survive? How many men/women were on the boat? How many across different classes? 
  
  - plot 3 subplots of proportions - pie/donut - survived, sex, class. 

##### Statistical tests (report.ipynb)

- *At least 2 statistical tests are included in your final report.*

- *The correct tests are run, given the data type and distribution, and the correct conclusions are drawn. For example (other tests may be used):*

  - *correlation: 2 continuous variables, normally distributed, testing for LINEAR correlation only (H_0: Not linearly dependent)*

  - *independent t-test: 1 continuous, somewhat normally distributed variable, one boolean variable, equal variance, independent (H_0: population mean of each group is equal)*
  
  - *chi-square test: 2 discrete variables. (H_0: the 2 variables are independent of each other).*

##### Summary (report.ipynb)

- *Following your exploration section, you summarize your analysis (in a **markdown** cell using natural language): what you found and how you will use it moving forward.*

- *This includes **key takeaways** from all the questions answered in explore, a **list of which features** will be used in modeling and **why**, and which features will not move forward and why. You may only call out a few of these features in the presentation, but having that there for reference is important in a report. A group of features may have the same reason why, and those can be mentioned together.*

#### Modeling

##### Select Evaluation Metric (Report.ipynb)

- *Clear communication as to how you evaluated and compared models.*

- *What metric(s) did you use and why? For example, in one case, you may decide to use precision over accuracy. If so, why? If you use multiple metrics, how will you decide which to select if metric is better for model A but another is better for model B? Will you rank them? Find a way to aggregate them into a single metric you can use to rank?*

##### Evaluate Baseline (Report.ipynb)

- *Having a baseline tells you whether a model you build using the features you selected is any better than predicting by using only the target variable. One way a baseline is created in classification is by making predictions purely based on the most common outcome class, like predicting that all titanic passengers will die, becuase the majroity did die. By doing that, you end up with the highest accuracy without using extra information from features. The baseline is based on the training dataset. For a continuous target variable, the baseline  could be predicting that all salaries will be the median salary of our labeled train data. The predictions should be made on the training data using this information (like the predicted value, y_hat, for all passengers "survived" == 0) and then performance evaluated to measure your models against. If any model you build does not perform as well as a baseline that uses no features, then your features are not significant drivers of the outcome.*

##### Develop 3 Models (Report.ipynb)

- *The 3 models can differ based on the features used, the hyperparameters selected, and/or the algorithm used to fit the data.*

#### Evaluate on Train (Report.ipynb)

- *All models should be evaluated on train: the training sample is our largest sample, and it is a sample of data we have to both fit the model AND see how the model performs. We should never skip straight to validate. We would be missing out on valuable observations.*

##### Evaluate on Validate (Report.ipynb)
 
- *The top models should be evaluated with the validation sample dataset. It is important to use the validate sample for checking for any overfitting that may have occurred when fitting the model on train. If you are creating 10's of models, it is also important to only validate a handful of your top models with the Validate dataset. Otherwise, your data will have seen validate as much as train and you could accidentally introduce some implicit bias based on data and results you see while validating on so many models. *

##### Evaluate Top Model on Test (Report.ipynb)

- *Your top performing model, and only your top performing model should be evaluated on your test dataset. The purpose of having a test dataset to evaluate only the final model on is to have an estimate of how the model will perform in the future on data it has never seen.*

## Report (Final Notebook) 

#### code commenting (Report.ipynb)

- *Your code contains code comments that are helpful to the reader in understanding what each blocks/lines of code are doing.*

#### markdown (Report.ipynb)

- *Notebook contains adequate markdown that documents your thought process, decision making, and navigation through the pipeline. This should be present throughout the notebook consistently, wtih not just headers, but plenty of content that guides the reader and leaves no questions or doubt as to why you did something, e.g.*

#### Written Conclusion Summary (Report.ipynb)

- *Your conclusion summary should addresses the questions you raised in the opening of the project, which we would want to see at the end of every final notebook. Ideally, when the deliverable is a report, the summary should tie together your analysis, the drivers of the outcome, and how you would expect your ML model to perform in the future on unseen data, in layman's terms.*

#### conclusion recommendations (Report.ipynb)

- *Your notebook should ends with a conclusion that contains  actionable recommendations based on your insights and analysis to the business stakeholder(s), your simulated audience, or someone who would find this information valuable (if there is no stakeholder). Your recommendations should not be not about what to do differently with the data, but instead should be based on the business or domain you are studying.*

#### conclusion next steps (Report.ipynb)

- *Your conclusion should include next steps from a data science perspective that will assist in improving your research. Ideally, if you talk about trying more algorithms to improve performance, think about why you need to improve performance. And if the business calls for it, remember the best way to improve performance is to have better predictors/features. If you talk about gathering more data, being specific about what data you think will help you understand the problem better and why is the way to go!*

#### no errors (Report.ipynb)

- *Your final notebook should run without error. One error in a notebook can lead to the rest of it erroring out. If you have a reader who doesn't know python, they will then not be able to consume your report.*

## Video Presentation

### intro (live)

- *Speaker kicks of the presentation by introducing themselves and their project through a one-liner of what it's about.*

### audience & setting  (live)

- *Always be aware of the audience and setting for your presentation.  What is the appropriate level of technicality? What is the appropriate depth given audience, setting and medium in which its delivered. The way you communicate should be appropriate for the audience: volume, speed of talk, flow, professionalism. (Codeup Data Science Instructor Team, virtually delivered via jupyter notebook).*

### content (live)

- *Notebook talked through step-by-step, in an understandable and meaningful way. Extraneous content in the notebook is not present.*

### Verbal Conclusion (findings, next steps, recommendations)  (live) 

- *Presentation is concluded with a summary of what was found, recommendations, and next steps. The presentation does not just drop off after modeling, but the entire project is nicely tied up and summarized.*

### time (live) 

- *Time limit of 5 minutes is adhered to. The time is managed well, in that there is appropriate time spent on each section. The time of 5 minutes should not be met by talking quickly but by reducing the amount or depth of information conveyed, and by finding easier and more simplified methods to convey the more complex information. The speech should be natural, and take the time needed for the audience to consume the information. So the time is well spent when you have practiced and you have taken the extra time it takes to reduce the content in your notebook and presentation. Time should not be spent scrolling through 10's of visualizations or hundreds of lines of code.*

## Deliver Predictions

### Deliver predictions (.csv) 

*A csv with predictions made from the top model developed should be submitted, as per instructions in the project spec.*

