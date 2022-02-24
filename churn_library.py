# library doc string


# import libraries
from pyrsistent import optional
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    return pd.read_csv(pth,index_col=0)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # plot histgram
    val=['Churn','Customer_Age']
    plt.figure(figsize=(20,10))
    for v in val:
        df[v].hist()
        plt.title(f'{v} distribution')
        plt.savefig(f'./images/{v}_histgram.png')
        plt.close()

    # bar plot
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Martital Status')
    plt.savefig('./images/Martital_status.png')
    plt.close()

    # distplot
    sns.displot(df['Total_Trans_Ct'],aspect=20/10)
    plt.title('Total_Trans_Ct')
    plt.savefig('./images/Total_Trans_Ct.png')
    plt.close()

    # heatmap
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.title('Correlation between Vars')
    plt.savefig('./images/Corr_plot.png')
    plt.close()



def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            X: pandas dataframe with new columns for
            y: target variables (binary)
    '''
    y = df['Churn']
    X = pd.DataFrame()
    
    # create target encoding for all categorical variables
    # category_lst=['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
    for col in category_lst:
        temp_list=[]
        groups=df.groupby(col).mean()['Churn']
        for val in df[col]:
                temp_list.append(groups.loc[val])
        df[f'{col}_Churn']=temp_list

    # create cols to keep and add to new X df.
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    return X,y

def perform_feature_engineering(X,y):
    '''
    input:
              X: pandas dataframe of all X variables (should be all numerical variables)
              y: pandas series includes target variables (should be numerical binary variable)

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def classification_report_file(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as txt file
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(8, 6))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 14}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 14}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 14}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 14}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/RandomForest_train_test_results.png')
    plt.close()

    plt.rc('figure', figsize=(8, 6))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 14}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 14}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 14}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 14}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/Logistic_regression_train_test_results.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
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
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    
    # save plot
    plt.savefig(output_pth)
    plt.close()

def roc_curve(rf_model,lr_model,X_test,y_test):
    """
    create ROC curve for two models
    input:
              rf_model: randomforest model
              lr_model: logistic regression model
              X_test: X testing data
              y_test: y testing data
    output:
              None
    """
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rf_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/roc_curve.png')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train models (logistic regression and random forest with grid search).
    store model scores to a txt file.
    store two model. 
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              y_test_preds_rf: randomforest y prediction with testing data.
              y_train_preds_rf: randomforest y prediction with training data.
              y_test_preds_lr: logistic regression y prediction with testing data.
              y_train_preds_lr: logistic regression y prediction with training data.


    '''
    # define moddel (random forest and logistic regression)
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=300)
    
    # grid search parameter
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
        }
    # perform grid search on random fores
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)   

    # fit logistic regression
    lrc.fit(X_train, y_train)
    
    # results - random forest
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    # results - logistic regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)   
    
    # store two models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # store model results
    classification_report_file(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    return y_test_preds_rf,y_train_preds_rf,y_test_preds_lr,y_train_preds_lr

if __name__=='__main__':
     # read in data
     df=import_data('./data/bank_data.csv')
     # change target variable to 0->existing customer. 1-> chruned customer.
     df['Churn']=df['Attrition_Flag'].apply(lambda val:0 if val=='Existing Customer' else 1)
     # eda analysis
     perform_eda(df)
     # list of categorical variables
     category_lst=['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
     # target encoding categorical variables
     X,y=encoder_helper(df,category_lst)
     # separate train and test data
     X_train, X_test, y_train, y_test = perform_feature_engineering(X,y)
     # train the model and outoput predictions results
     y_test_preds_rf,y_train_preds_rf,y_test_preds_lr,y_train_preds_lr=train_models( X_train, X_test, y_train, y_test)
     # load back the best model (random forest)
     rfc_model = joblib.load('./models/rfc_model.pkl')
     lr_model = joblib.load('./models/logistic_model.pkl')
     # create roc curve
     roc_curve(rfc_model,lr_model,X_test,y_test)
     # create feature importance plot
     feature_importance_plot(rfc_model,X,'./images/feature_importance_rf.png')
