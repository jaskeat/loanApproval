# Loan Approval Prediction

This project is an analysis of a loan approval dataset to analyse and create a prediction model for loans. I have used Python with SciKit, Streamlit, Pandas and other packages to do this analysis.

Along with the project, I also made a website using Streamlit where you can input various metrics to see the predicted result form the machine learning model.

<!-- TODO: PUT GIF HERE -->

![Streamlit GIF](/Pictures/WebApp.gif)

### Goal

The goal of this project is to create a prediction model that can classify loans with high accuracy. This model aims to assist financial institutions in automating the loan approval process, thereby reducing manual effort and minimizing human error. Additionally, this project could provide insights into the key factors that influence loan approval decisions, helping institutions to better understand the criteria that should be considered when evaluating loan applications.

By leveraging machine learning techniques, the project aspires to enhance the efficiency and reliability of the loan approval process, ultimately contributing to more informed and data-driven decision-making.

### Technologies used:

**Python**: Used with Jupyter Notebook to make the machine learning model  
**SciKit-Learn**: For building and evaluating the prediction model  
**Pandas**: For data manipulation and analysis  
**Streamlit**: For creating the web application to input metrics and see predictions  
**Matplotlib**: For data visualization  
**NumPy**: For numerical operations  
**Joblib**: Used to save the model as a file for later use

### Dataset:

The dataset used for this is a synthetic dataset inspired by real datasets. There are few datasets for loan approval so a synthetic dataset was the next best option. It has 45000 rows and 14 variables seen below.

|             Column             |                    Description                    |    Type     |
| :----------------------------: | :-----------------------------------------------: | :---------: |
|           person_age           |                 Age of the person                 |    Float    |
|         person_gender          |               Gender of the person                | Categorical |
|        person_education        |              Highest education level              | Categorical |
|         person_income          |                   Annual income                   |    Float    |
|         person_emp_exp         |          Years of employment experience           |   Integer   |
|     person_home_ownership      | Home ownership status (e.g., rent, own, mortgage) | Categorical |
|           loan_amnt            |               Loan amount requested               |    Float    |
|          loan_intent           |                Purpose of the loan                | Categorical |
|         loan_int_rate          |                Loan interest rate                 |    Float    |
|      loan_percent_income       |   Loan amount as a percentage of annual income    |    Float    |
|   cb_person_cred_hist_length   |         Length of credit history in years         |    Float    |
|          credit_score          |            Credit score of the person             |   Integer   |
| previous_loan_defaults_on_file |        Indicator of previous loan defaults        | Categorical |
| loan_status (target variable)  | Loan approval status: 1 = approved; 0 = rejected  |   Integer   |

Dataset preview:

| person_age | person_gender | person_education | person_income | person_emp_exp | person_home_ownership | loan_amnt | loan_intent | loan_int_rate | loan_percent_income | cb_person_cred_hist_length | credit_score | previous_loan_defaults_on_file | loan_status |     |
| ---------: | ------------: | ---------------: | ------------: | -------------: | --------------------: | --------: | ----------: | ------------: | ------------------: | -------------------------: | -----------: | -----------------------------: | ----------: | --- |
|       22.0 |        female |           Master |       71948.0 |              0 |                  RENT |   35000.0 |    PERSONAL |         16.02 |                0.49 |                        3.0 |          561 |                             No |           1 |     |
|       21.0 |        female |      High School |       12282.0 |              0 |                   OWN |    1000.0 |   EDUCATION |         11.14 |                0.08 |                        2.0 |          504 |                            Yes |           0 |     |
|       25.0 |        female |      High School |       12438.0 |              3 |              MORTGAGE |    5500.0 |     MEDICAL |         12.87 |                0.44 |                        3.0 |          635 |                             No |           1 |     |

Loan_Status is the target variable we will try to predict and we will analyze the rest to choose the right set of features for the prediciton.

##### Distribution of Loan_status:

![Distribution of Loan_status](/Pictures/loanstatus_distribution.png)

From the distribution we can see that most of the loans are Rejected which is expected as Banks and institutions have to selectively pick which loans to approve. When testing our model for accuracy we should pay attention to this distribution.

### Methodology

For the prediction model, I decided to go with the decision Tree model. Before going to the model, I had to pick the correct features. For this I used backward selection where I started with all features and removed them one by one and used Accuracy to decide which ones to remove. In the end, I ended with these features:

| person_income | loan_amnt | person_emp_exp | loan_int_rate | loan_percent_income | cb_person_cred_hist_length | credit_score |
| ------------- | --------- | -------------- | ------------- | ------------------- | -------------------------- | ------------ |

Initially, the accuracy of the decision Tree model was quite poor but after increasing the maximum samples for node, the accuracy was greatly approved. Part of the decision tree diagram can be seen below.

![Decision Tree](/Pictures/decisionTree.png)

### Results

The data was split into training and test data, where the entire model was trained on the training data. Below are the results from the test data.

##### Confusion matrix:

|                 | Predicted Positive | Predicted Negative |
| --------------- | ------------------ | ------------------ |
| Actual Positive | 1386               | 1101               |
| Actual Negative | 534                | 8229               |
| **Accuracy**    |                    | **85.47%**         |

From the accuracy we can see that the model does well to predict the Loan status. However, since the distribution of the loan status is extremely skewed, we cannot solely rely on the accuracy. To get a better understanding we can look at the True postive rate.

| True Positive Rate (TPR) |
| ------------------------ |
| 72%                      |

The TPR shows that when the model predicts an approved Loan status, it is correct 72% of the time which can be improved but it shows that the model works well.

This can further be seen with the ROC Curve.

![ROC Curve](/Pictures/rocCurve.png)

### Snapshot of predicted results

| person_income | loan_amnt | person_emp_exp | loan_int_rate | loan_percent_income | cb_person_cred_hist_length | credit_score | Actual | Predicted |
| ------------: | --------: | -------------: | ------------: | ------------------: | -------------------------: | -----------: | -----: | --------: |
|       72707.0 |   24000.0 |              4 |         10.84 |                0.33 |                        4.0 |          614 |      1 |         1 |
|       85497.0 |   14075.0 |              0 |         14.74 |                0.16 |                        4.0 |          623 |      0 |         0 |

When doing a deeper dive on the predicted results, we can see that even if someone is applying for a loan insignificant to their yearly income, it can get rejected as a bank may not want to give smaller loans. We can have a further look on the most important features.

##### Feature importance

| Feature                     | Ranking |
| --------------------------- | ------- |
| loan_percent_income:        | 1       |
| loan_int_rate:              | 2       |
| person_income:              | 3       |
| loan_amnt:                  | 4       |
| credit_score:               | 5       |
| cb_person_cred_hist_length: | 6       |
| person_emp_exp:             | 7       |

Here we can see that features that show the Credit bearing ability of the borrower are lower on the ranking suggesting that they are not as important where has features related to the loan like the amount or income of the borrower are more important.

### Improvements and considerations

There are a lot of improvements to be made in this model, firstly it would be better to use real data instead of a synthetic data like I used for this prediction model as then the conclusion would be more definitive.

Things like feature selection could also have been automated instead of doing it one by one as it takes longer and it is also more susceptible to errors. Other machine learning techniques also could've been used to compare to the decision tree to choose the best model possible. Random forests or other logistic regression techniques can be considered.

### Conclusion

In this project, we developed a loan approval prediction model using a synthetic dataset. By using a decision tree model, we achieved an accuracy of 85.47% and a True Positive Rate (TPR) of 72% which could be improved but shows that the model can be relied upon.

Key insights showed that loan-related factors, such as loan amount and interest rate, are more significant in approval decisions than the borrower's credit history. This was seen in cases where the borrower had great credit history but if the loan amount was too small, they would still get rejected.

This project highlights the potential of machine learning to automate and improve the loan approval process, aiding financial institutions in making data-driven decisions.
