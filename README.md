# LendingClub-LoanStatus-Classifier
**Dataset Link: https://www.kaggle.com/datasets/wordsforthewise/lending-club** 
[![Demo Link](https://github.com/sujay-2001/LendingClub-LoanStatus-Classifier/blob/main/Rep%20Image(YT).png)](https://youtu.be/mTZgdKDqtJ4)

## Ask: 
### Problem:
For banks, which deal with large number of loan applications on a single day, it becomes difficult for them
to analyse and process the loan, based on whether the loan applicant will  be able to pay back the loan or not.
This would also ensure smooth functioning, and improve business processes in banks.
### Goal: 
(i) Develop a **model** that would predict whether a loan applicant would pay back the loan or not, based on the features
of the applicant. <br />
(ii) Implement the model in an **API**.

## Prepare:
Dataset from Kaggle as mentioned (LendingClub Loan Data) was collected,
checked to ensure that data used in unbiased and secure. This dataset contained information on features of loan 
applicants, and the information whether they paid the loan back or charged off.

## Clean:
The imported data had a lot of missing entries, and unstructured attributes. Missing data was dealt by filling them
with average values, and removing few observations. Highly correlated features (>0.95) were removed. And irrelevant
attributes were dropped off. These were done with the help of **pandas** module.

## Analyse:
This included studying relationships between different attributes, by plots and manipulating data (aggregation,
grouping, pivot tables, etc). Feature engineering was also done for features like address to extract valuable
information with respect to derived variables (for instance state, zipcode in address). Dummies were created for
categorical variables. The entire exploratory data analysis was done using **pandas, seaborn, matplotlib,** etc.

## Model Training:
This included: <br />
(i)   Train-test split: Dataset was split into train and test data with the help of train_test_split from **sklearn**. <br />
(ii)  Feature Scaling: This is the final step of data preprocessing. It is a technique to standardize the independent
variables of the dataset in a specific range. In feature scaling, we put our variables in the same range and in the
same scale so that no any variable dominate the other variable. For feature scaling, we will import StandardScaler
class of **sklearn.preprocessing library**.  <br />
(iii) Model: A sequential neural network model was created using **Tensorflow, Keras**. Dropouts and earlystopping method was
added to avoid overfitting to training data while training. The trained model is then saved using pickle and performance is evaluated. <br />
![Demo Link](https://github.com/sujay-2001/LendingClub-LoanStatus-Classifier/blob/main/Loss_Plt.png)

## Implement model in API:
An API was developed using **Python, Flask** (for Backend) and **HTML, CSS** (for Frontend). This accepts all the important features of a loan applicant
and then saves the features, to send it to the model (saved model) which would predict the outcome (loan status).

## Conclusion:
All the important features that are crucial in determining the loan status of an applicant were obtained, neural network model was built to predict the 
outcome (learning from data), and the model was implemented in an API. The API could be improved by interacting with databases (like SQL, cloud data) and predict the outcome.

