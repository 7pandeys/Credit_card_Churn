"""
This is the Python Test for the churn_library.py module.

This module will be used to test
    1. import_data
    2. peform_eda
    3. encode_data
    4. perform_feature_engineering
    5. train_test_model

Author: Sandeep Pandey
Date: 3-Mar-2023
"""


import os
import logging
# import churn_library_solution as cls
from churn_library import import_data, encoder_helper

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(i_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_data = i_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_data.shape[0] > 0
        assert df_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    try:
        assert os.path.isfile('images/eda/Churn_hist.png')
        assert os.path.isfile('images/eda/Cust_age_hist.png')
        assert os.path.isfile('images/eda/marital_bar.png')
        assert os.path.isfile('images/eda/Total_Trans_Ct.png')
        assert os.path.isfile('images/eda/corr_heatmap.png')
    except AssertionError as err:
        logging.error("Testing test_eda: all required images are not present.")
        raise err


def test_encoder_helper(e_helper):
    '''
    test encoder helper
    '''
    df_data = import_data("data/bank_data.csv")
    cat_columns_exist = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    cat_columns = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_news = e_helper(df_data, cat_columns_exist)
# 	print(df.columns)
    for i in cat_columns:
        try:
            assert i in df_news
        except AssertionError as err:
            logging.error(
                "Testing test_encoder_helper: all required columns are not presents")
            raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df_data = import_data("data/bank_data.csv")
    cat_columns_exist = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df_data['Churn'] = df_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df_news = encoder_helper(df_data, cat_columns_exist)
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
    for i in keep_cols:
        try:
            assert i in df_news
        except AssertionError as err:
            logging.error(
                "Testing test_perform_feature_engineering: all required columns are not present")
            raise err


def test_train_models():
    '''
    test train_models
    '''
    try:
        assert os.path.isfile('models/logistic_model.pkl')
        assert os.path.isfile('models/rfc_model.pkl')
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: all required models are not present.")
        raise err


if __name__ == "__main__":
    test_import(import_data)
    test_eda()
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering()
    test_train_models()
# 	pass
