"""
This is the Python Test for the churn_library.py module.

This module will be used to create
    1. import_data - To import dataset
    2. peform_eda - perform exploratory data analysis
    3. encode_helper - encode category variables
    4. perform_feature_engineering - perform feature engineering
    5. train_test_model - train and test models and create image and store .pkl models

Author: Sandeep Pandey
Date: 3-Mar-2023
"""


# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df_churn = pd.read_csv(pth)
    return df_churn


def perform_eda(df_churn):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    df_churn['Churn'] = df_churn['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df_churn['Churn'].hist()
    plt.savefig('images/eda/Churn_hist.png')
    plt.figure(figsize=(20, 10))
    df_churn['Customer_Age'].hist()
    plt.savefig('images/eda/Cust_age_hist.png')
    plt.figure(figsize=(20, 10))
    df_churn.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_bar.png')
    plt.figure(figsize=(20, 10))
    sns.histplot(df_churn['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/Total_Trans_Ct.png')
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_churn.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/corr_heatmap.png')


def encoder_helper(df_churn, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: optional argument could be used for naming variables or index y column

    output:
            df: pandas dataframe with new columns for
    '''
    for i in category_lst:
        category_ls = []
        category_groups = df_churn.groupby(i).mean()['Churn']
#         print(category_groups)

        for val in df_churn[i]:
            category_ls.append(category_groups.loc[val])

        df_churn[f'{i}_Churn'] = category_ls

    return df_churn


def perform_feature_engineering(df_churn):
    '''
    input:
              df: pandas dataframe
              response: optional argument that could be used for naming variables or index y column

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_label = df_churn['Churn']
    x_features = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x_features[keep_cols] = df_churn[keep_cols]
    return train_test_split(
        x_features,
        y_label,
        test_size=0.3,
        random_state=42)


def classification_report_image(y_test,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_test:  test response values
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Generate the classification report
    report_test_rf = classification_report(y_test, y_test_preds_rf)
    report_test_lr = classification_report(y_test, y_test_preds_lr)

    # Create a plot of the report
    fig, ax_s = plt.subplots(figsize=(8, 6))
    print(fig)
    ax_s.axis('off')
    plt.figtext(
        0.5,
        0.01,
        report_test_rf,
        wrap=True,
        horizontalalignment='center',
        fontsize=12)
    # Save the plot as an image file
    plt.savefig('images/results/rf_classification_report.png')
    plt.figtext(
        0.5,
        0.01,
        report_test_lr,
        wrap=True,
        horizontalalignment='center',
        fontsize=12)
    # Save the plot as an image file
    plt.savefig('images/results/lr_classification_report.png')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    # store in outputpath
    plt.savefig(f'{output_pth}_feature_importance.png')


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # rfc_model = joblib.load('./models/rfc_model.pkl')
    # lr_model = joblib.load('./models/logistic_model.pkl')
#     y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

#     y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # plots create roc_curve image
    plt.figure(figsize=(15, 8))
    ax_s = plt.gca()
    plot_roc_curve(lrc, x_test, y_test, ax=ax_s, alpha=0.8)
    plot_roc_curve(cv_rfc, x_test, y_test, ax=ax_s, alpha=0.8)
    plt.savefig('images/results/roc_curve.png')

    classification_report_image(y_test,
                                y_test_preds_lr,
                                y_test_preds_rf)
    feature_importance_plot(cv_rfc, x_train, 'images/results/')


if __name__ == '__main__':
    df = import_data('data/bank_data.csv')
    perform_eda(df)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_new = encoder_helper(df, cat_columns)
    X_train_o, X_test_o, y_train_o, y_test_o = perform_feature_engineering(
        df_new)
    train_models(X_train_o, X_test_o, y_train_o, y_test_o)
