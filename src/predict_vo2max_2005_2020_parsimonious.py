

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib



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

def pred_crf_2005_2020():
    """
    Predict Vo2max for for eligible participants from 2005-2020 (16 years)
    """
    X_pred_parsimonious = pd.read_csv('../data/processed/X_pred_all_dummies.csv')[parsimonious_variables+["seqn"]]
    
    # seqn: ID of the participant
    seqn_16 = X_pred_parsimonious['seqn'].values
    
    # load the model
    lgb_parsimonious =  joblib.load('../model/lgb_parsimonious.pkl')
    
    # predict
    pred16y_lgb_parsimonious = lgb_parsimonious.predict(X_pred_parsimonious.drop(['seqn'],
                                                                                 axis =1))

    pred16_df = pd.DataFrame({
        'seqn': seqn_16,
        'vo2max_parsimonious_lgb_pred': pred16y_lgb_parsimonious,

    })

    # save the prediction
    pred16_df.to_csv('../prediction/predicted_vo2max_2005_2020_parsimonious.csv')
    print("Prediction saved to ../prediction/predicted_vo2max_2005_2020_parsimonious.csv")



if __name__ == '__main__':
    pred_crf_2005_2020()