"""

Module contain function of testing the functions inside churn_library.py
Author : Peter Sun
Date : 23rd Feburary 2022


"""

from cmath import log
from csv import excel
import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = cl.import_data("./data/bank_data.csv")
        logging.info("SUCESS: Testing import_data")
    except FileNotFoundError as err:
        logging.error("ERROR: Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df = cl.import_data('./data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    # if any column is missing
    try:
        cl.perform_eda(df)
        logging.info('SUCESS: performing EDA...')
    except KeyError as err:
        logging.error(f'ERROR: fail to find {err.args[0]} columns')
    # check if plots are created
    list_of_plots = [
        'Churn_histgram',
        'Customer_Age_histgram',
        'Martital_status',
        'Corr_plot',
        'Total_Trans_Ct']
    for plot in list_of_plots:
        try:
            assert os.path.isfile(f'./images/eda/{plot}.png') is True
            logging.info(f'SUCESS: {plot}.png is created.')
        except AssertionError:
            logging.error(f'ERROR: {plot} was not found.')


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = cl.import_data('./data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    X, y = cl.encoder_helper(df, category_lst)
    # number of rows of X, and length of y is same as df
    try:
        assert X.shape[0] == df.shape[0]
        logging.info(
            'SUCESS: number of rows of X is same as number of rows of raw dataframe')
    except AssertionError:
        logging.error(
            'ERROR: number of rows of X is DIFFERENT than number of rows of raw dataframe')
    try:
        assert len(y) == df.shape[0]
        logging.info(
            'SUCESS: number of rows of y is same as number of rows of raw dataframe')
    except AssertionError:
        logging.error(
            'ERROR: number of rows of y is DIFFERENT than number of rows of raw dataframe')

    # check missing values
    ct_x_missing = X.isnull().sum().sum()
    ct_y_missing = y.isnull().sum()
    try:
        assert ct_x_missing == 0
        logging.info('SUCESS: no missing value in X')
    except AssertionError:
        logging.error(f'ERROR: there are {ct_x_missing} missing values in X')
    try:
        assert ct_y_missing == 0
        logging.info('SUCESS: no missing value in y')
    except AssertionError:
        logging.error(f'ERROR: there are {ct_y_missing} missing values in y')


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df = cl.import_data('./data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    X, y = cl.encoder_helper(df, category_lst)
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(X, y)
    try:
        assert X_train.shape[0] + X_test.shape[0] == df.shape[0]
        assert X_train.shape[1] == X_test.shape[1] == X.shape[1]
        logging.info(
            'SUCESS: dimension of X_train,X_test,y_train,y_train are correct.')
    except AssertionError:
        logging.error(
            'ERROR: dimenion of X_train,X_test,y_train,y_train are NOT correct.')

def test_train_models():
    '''
    test train_models
    '''
    df = cl.import_data('./data/bank_data.csv')
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    X, y = cl.encoder_helper(df, category_lst)
    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(X, y)
    # check if two models (randomForest and logistic regression) are outputed.
    cl.train_models(X_train, X_test, y_train, y_test)
    try:
        os.path.isfile('./models/logistic_model.pkl') is True
        os.path.isfile('./models/rfc_model.pkl') is True
        logging.info('SUCESS: there are two models')
    except AssertionError:
        logging.error('ERROR: models or one model are missing')


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
