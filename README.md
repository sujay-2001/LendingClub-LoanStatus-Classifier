# LendingClub-LoanStatus-Classifier
**Dataset Link: https://www.kaggle.com/datasets/wordsforthewise/lending-club** 
[![Demo Link]](https://github.com/sujay-2001/LendingClub-LoanStatus-Classifier/blob/main/Rep%20Image_YT.png)(https://youtu.be/mTZgdKDqtJ4)

## Ask:
### Problem:
For banks, which deal with large number of loan applications on a single day, it becomes difficult for them
to analyse and process the loan, based on whether the loan applicant will  be able to pay back the loan or not.
### Goal: 
(i) Develop a model that would predict whether a loan applicant would pay back the loan or not, based on the features
of the applicant. 
(ii) Implement the model in an API.

## Prepare:
Dataset from Kaggle as mentioned (LendingClub Loan Data) was collected,
checked to ensure that data used in unbiased and secure. This dataset contained information on features of loan 
applicants, and the information whether they paid the loan back or charged off.

## Clean:
The imported data had a lot of missing entries, and unstructured attributes. Missing data was dealt by filling them
with average values, and removing few observations. Highly correlated features (>0.95) were removed. And irrelevant
attributes were dropped off. These were done with the help of pandas module.

## Analyse:
This included studying relationships between different attributes, by plots and manipulating data (aggregation,
grouping, pivot tables, etc). Feature engineering was also done for features like address to extract valuable
information with respect to derived variables (for instance state, zipcode in address). Dummies were created for
categorical variables. The entire exploratory data analysis was done using pandas, seaborn, matplotlib, etc.

## Model Training:
This included 
8
