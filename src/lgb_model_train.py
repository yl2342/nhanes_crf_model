## loading packages

import os
import time
import math
import random
from random import sample
from random import seed

import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scipy.stats import randint
from scipy.stats import uniform

import joblib



## 59 variables used for the extended model
extended_variables = [
    'INDFMPIR', 'health_insurance', 'htn_history', 'age', 'ALT', 'AST',
    'BUN', 'GLU', 'LDH', 'CHOL', 'TOTPRO', 'POTAS', 'SODI', 'albuminuria',
    'CHL', 'HBA1C', 'CREATININE', 'BILIRUBIN', 'CALCIUM', 'BICARBONATE',
    'SY_mean', 'DI_mean', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'BMXWT', 'BMXHT',
    'WT_DIFF_KG', 'lwlstyr', 'exerc2lwlastyr', 'ARM_PF', 'ARM_BMD',
    'ARM_LEAN', 'LEG_PF', 'LEG_BMD', 'LEG_LEAN', 'TR_PF', 'TR_BMD',
    'TR_LEAN', 'TWMT_log', 'walk_bike', 'gender_Male',
    'race_ethn_Non-Hispanic Black', 'race_ethn_Non-Hispanic White',
    'race_ethn_Other', 'educ_High school diploma or GED',
    'educ_Less than high school', 'marital_status_other',
    'family_income_Low income', 'employment_status_Unemployed',
    'employment_status_Working', 'smoking_status_Former smoker',
    'smoking_status_Never smoker', 'alcohol_intake_Heavier drinker',
    'alcohol_intake_Light drinker', 'alcohol_intake_Moderate drinker',
    'alcohol_intake_Never drinker', 'physical_activity_Inactive',
    'physical_activity_Insufficient active']


# 30 variables used for the parsimonious model
parsimonious_variables = ['INDFMPIR', 'health_insurance', 'htn_history', 'age', 'SY_mean',
    'DI_mean', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'BMXWT', 'BMXHT',
    'WT_DIFF_KG', 'lwlstyr', 'exerc2lwlastyr', 'TWMT_log', 'walk_bike',
    'gender_Male', 'race_ethn_Non-Hispanic Black',
    'race_ethn_Non-Hispanic White', 'race_ethn_Other',
    'educ_High school diploma or GED', 'educ_Less than high school',
    'marital_status_other', 'family_income_Low income',
    'employment_status_Unemployed', 'employment_status_Working',
    'smoking_status_Former smoker', 'smoking_status_Never smoker',
    'alcohol_intake_Heavier drinker', 'alcohol_intake_Light drinker',
    'alcohol_intake_Moderate drinker', 'alcohol_intake_Never drinker',
    'physical_activity_Inactive', 'physical_activity_Insufficient active']



def load_data(type = 'parsimonious'):
    """
    Load the data and select the variables used for the model
    :param type: string, 'extended' or 'parsimonious'  
    """
    ## read the data (category variables are already dummified) and select the variables 
    if type == 'extended':
        train = pd.read_csv('../data/processed/train_all_dummies.csv')[extended_variables + ["y_train"]]
        test = pd.read_csv('../data/processed/test_all_dummies.csv')[extended_variables + ["y_test"]]
    else:
        train = pd.read_csv('../data/processed/train_all_dummies.csv')[parsimonious_variables + ["y_train"]]
        test = pd.read_csv('../data/processed/test_all_dummies.csv')[parsimonious_variables + ["y_test"]]
    ## train
    X_train = train.drop(['y_train'], axis =1)
    y_train = train['y_train']
    # test
    X_test = test.drop(['y_test'], axis =1)
    y_test = test['y_test']
    return X_train, y_train, X_test, y_test




def model_train(type = 'parsimonious'):
    """
    Train the model using lightgbm with the data specified
    """
    ## load the data
    X_train, y_train, X_test, y_test = load_data(type)

    ## random search 100 times
    search_iter = 100

    param_dist_lgb = {
        'learning_rate' : [0.1, 0.01,0.05],
        'num_leaves': [15,31,63,127,255,511,1023,2047],
        'min_child_samples':[1,5,10,15,20],
        'subsample': uniform(loc = 0.4, scale =0.5),
        'reg_lambda' : uniform(loc = 0.0, scale =0.3),
        'bagging_freq' : [0,5,10,15,30],
        'n_estimators' :[30,50,100,200,500,1000],
        'feature_fraction' : uniform(loc = 0.4, scale =0.5)
    }
    # lgb model
    lgb_reg = lgb.LGBMRegressor(
        objective = 'regression',
        random_state= 42,
        min_split_gain = 0.00001,
        verbose = -1)
    
    ## using rmse score for random search
    randomsearch_lgb_rmse  = RandomizedSearchCV(
        estimator= lgb_reg,
        param_distributions=param_dist_lgb,
        n_iter=search_iter,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs = -1,
        refit=True,
        random_state= 42,
        verbose=1)

    ## rmse score loss
    start = time.time()
    print('===================================================')
    print('Training {} model...'.format(type))

    randomsearch_lgb_rmse.fit(X=X_train,y=y_train)

    print('Best score rmse in validation set: {}'.format(randomsearch_lgb_rmse.best_score_))

    best_estimator = randomsearch_lgb_rmse.best_estimator_

    ## predict the test set
    pred_test = best_estimator.predict(X_test)

    print('Test RMSE: {}, test MAE:{}, test R^2: {}'.format(np.sqrt(mean_squared_error(y_test, pred_test)),
                                                            mean_absolute_error(y_test,pred_test),
                                                            r2_score(y_test, pred_test)))
    # save the model
    joblib.dump(best_estimator, '../model/lgb_'+type+'.pkl')
    print('Model saved in ../model/lgb_{}.pkl'.format(type))

    end = time.time()
    print('Execution time : {} minutes'.format((end - start)/60))
    print('===================================================')


def main():
    """
    Main function to train the model
    """
    # train the parsimonious model
    model_train(type = 'parsimonious')
    # train the extended model
    model_train(type = 'extended')



if __name__ == '__main__':
    main()
    