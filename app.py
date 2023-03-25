import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['POST'])
def home():
    return render_template('index.html')


@app.route('/Predict', methods=['POST'])
def Predict():
    '''
    For rendering results on HTML GUI
    '''
    df = pd.read_csv('lending_club_loan_data.csv')
    df = df.drop('emp_title', axis=1)
    df = df.drop('title', axis=1)
    df = df.drop('emp_length', axis=1)
    df = df.drop('installment', axis=1)
    df = df.drop('earliest_cr_line', axis=1)
    df = df.drop('grade', axis=1)
    df = df.drop('issue_d', axis=1)
    total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
    def fill_mort_acc(total_acc, mort_acc):
        '''
        Accepts the total_acc and mort_acc values for the row.
        Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
        for the corresponding total_acc value for that row.

        total_acc_avg here is a Series containing the mapping of the
        groupby averages of mort_acc per total_acc values.
        '''
        if np.isnan(mort_acc):
            return total_acc_avg[total_acc]
        else:
            return mort_acc

    df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
    df = df.dropna()
    df['zip_code'] = df['address'].apply(lambda x: x.split(' ')[-1])
    df = df.append(pd.Series(), ignore_index=True)
    df['loan_amnt'] = df['loan_amnt'].fillna(float(request.form.get('loan_amnt')))
    df['term'] = df['term'].fillna(str(request.form.get('term')))
    df['int_rate'] = df['int_rate'].fillna(float(request.form.get('int_rate')))
    df['annual_inc'] = df['annual_inc'].fillna(float(request.form.get('ann_inc')))
    df['dti'] = df['dti'].fillna(float(request.form.get('dti')))
    df['open_acc'] = df['open_acc'].fillna(float(request.form.get('open_acc')))
    df['pub_rec'] = df['pub_rec'].fillna(float(request.form.get('pub_rec')))
    df['revol_bal'] = df['revol_bal'].fillna(float(request.form.get('rev_bal')))
    df['revol_util'] = df['revol_util'].fillna(float(request.form.get('rev_util')))
    df['total_acc'] = df['total_acc'].fillna(float(request.form.get('total_acc')))
    df['mort_acc'] = df['mort_acc'].fillna(float(request.form.get('mort_acc')))
    df['pub_rec_bankruptcies'] = df['pub_rec_bankruptcies'].fillna(float(request.form.get('pub_rec_bankruptcies')))
    df['sub_grade'] = df['sub_grade'].fillna(str(str(request.form.get('grade'))+str(request.form.get('sub_grade'))))
    df['verification_status'] = df['verification_status'].fillna(str(request.form.get('verification_status')))
    df['application_type'] = df['application_type'].fillna(str(request.form.get('application_type')))
    df['initial_list_status'] = df['initial_list_status'].fillna(str(request.form.get('initial_list_status')))
    df['purpose'] = df['purpose'].fillna(str(request.form.get('purpose')))
    df['home_ownership'] = df['home_ownership'].fillna(str(request.form.get('home_ownership')))
    df['zip_code'] = df['zip_code'].fillna(str(request.form.get('zip_code')))
    df['term'] = df['term'].apply(lambda x: int(x[:3]))
    df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

    #Creating dummies
    subgrade_dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
    df = pd.concat([df.drop(['sub_grade'], axis=1), subgrade_dummies], axis=1)
    dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']],
                             drop_first=True)
    df = pd.concat(
        [df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis=1), dummies],
        axis=1)
    homeownership_dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
    df = pd.concat([df.drop('home_ownership', axis=1), homeownership_dummies], axis=1)
    zip_dummies = pd.get_dummies(df['zip_code'], drop_first=True)
    df = pd.concat([df.drop(['zip_code', 'address'], axis=1), zip_dummies], axis=1)
    df = df.drop('loan_status', axis=1)
    x = df.values
    # Make prediction
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    def classify(p):
        # Assigns 0/1 based on sigmoid probability
        if p > 0.5:
            return "The loan applicant is more likely to pay back the loan"
        else:
            return "The loan applicant is more likely to charge off the loan"

    y = model.predict(x)
    prediction = classify(y[-1])

    return render_template('result.html', prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)
