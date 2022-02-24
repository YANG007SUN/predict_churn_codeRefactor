# Predict Customer Churn

Author: Peter

Date: 23rd Feburary 2022

## Project Description
In this project, I identified credit card customers that are most likely to churn. The Project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package also has the flexibility of being run interactively or from the command-line interface (CLI).

## Project files
`churn_library.py`: python file for creating models<br>
`churn_script_logging_and_tests.py` python file for testing `churn_library.py` file.

## Files in Repo
Data:
- bank_data.csv<br>

Images:
- eda
    - Churn_histgram.png (distribution)
    - Customer_Age_histgram.png (distribution)
    - Corr_plot.png (heatmap)
    - Martital_status.png (distribution)
    - Total_Trans_Ct.png (distribution)

- results:
    - Logistic_regression_train_test_results.png
    - RandomForest_train_test_results.png
    - feature_importance_rf.png
    - roc_curve.png

Logs:
- churn_library.log

Models:
- logistic_model.pkl
- rfc_model.pkl

Python Files:
- churn_library.py
- churn_notebook.ipynb
- churn_script_logging_and_tests.py

README.md

## Purpose of files:
`churn_library.py`: Including functions for creating models.

`churn_script_logging_and_tests.py`: Including functions for testing the functions inside `churn_library.py` file. It will generate logs inside `logs` folder.

