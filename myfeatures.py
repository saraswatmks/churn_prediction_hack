#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 19:56:20 2017

@author: manish
"""

## create new features

import pandas as pd

def engineer(train, test):
    
    train['total_credit_amt'] = train['C_prev1'] + train['C_prev2'] + train['C_prev3'] + train['C_prev3'] + train['C_prev5'] + train['C_prev6']
    test['total_credit_amt'] = test['C_prev1'] + test['C_prev2'] + test['C_prev3'] + test['C_prev3'] + test['C_prev5'] + test['C_prev6']
    
    train['total_debit_amt'] = train['D_prev1'] + train['D_prev2'] + train['D_prev3'] + train['D_prev3'] + train['D_prev5'] + train['D_prev6']
    test['total_debit_amt'] = test['D_prev1'] + test['D_prev2'] + test['D_prev3'] + test['D_prev3'] + test['D_prev5'] + test['D_prev6']
    
    train['final_worth_eng'] = train.apply(lambda row: str(row['FINAL_WORTH_prev1']) + str(row['ENGAGEMENT_TAG_prev1']), axis=1)
    test['final_worth_eng'] = test.apply(lambda row: str(row['FINAL_WORTH_prev1']) + str(row['ENGAGEMENT_TAG_prev1']), axis=1)
    
    ## vintage mean
    train['vintage_by_age'] = train.groupby('age')['vintage'].transform('mean')
    test['vintage_by_age'] = train.groupby('age')['vintage'].transform('mean')
        
    ## Average montly balance - ratio
    train['amb_ratio1'] = train['I_AQB_PrevQ2'] / train['I_AQB_PrevQ1']
    test['amb_ratio1'] = test['I_AQB_PrevQ2'] / test['I_AQB_PrevQ1']

    ## bin age variable
    bins = [0,18,40,60,80,90]
    labels = ['teens','adults','oldies','more-oldies','deads']
    
    # categories = pd.cut(df['age'], bins, labels=labels)
    train['age_bin'] = pd.cut(train['age'], bins, labels=labels)
    test['age_bin'] = pd.cut(test['age'], bins, labels=labels)

    #### Ratio of end of month balance #######
    train['eop_ratio1'] = train['EOP_prev2'] / train['EOP_prev1']
    test['eop_ratio1'] = test['EOP_prev2'] / test['EOP_prev1']
    
    train['eop_ratio2'] = train['EOP_prev3'] / train['EOP_prev1']
    test['eop_ratio2'] = test['EOP_prev3'] / test['EOP_prev1']
    
    train['eop_ratio3'] = train['EOP_prev4'] / train['EOP_prev1']
    test['eop_ratio3'] = test['EOP_prev4'] / test['EOP_prev1']
    
    train['eop_ratio4'] = train['EOP_prev5'] / train['EOP_prev1']
    test['eop_ratio4'] = test['EOP_prev5'] / test['EOP_prev1']
    
    train['eop_ratio5'] = train['EOP_prev5'] / train['EOP_prev1']
    test['eop_ratio5'] = test['EOP_prev5'] / test['EOP_prev1']

    
    #### NET FEATURES ##########
    
    train['net_prev1'] = train['C_prev1'] - train['D_prev1']
    test['net_prev1'] = test['C_prev1'] - test['D_prev1']
    
    train['net_prev2'] = train['C_prev2'] - train['D_prev2']
    test['net_prev2'] = test['C_prev2'] - test['D_prev2']
    
    train['net_prev3'] = train['C_prev3'] - train['D_prev3']
    test['net_prev3'] = test['C_prev3'] - test['D_prev3']
    
    train['net_prev4'] = train['C_prev4'] - train['D_prev4']
    test['net_prev4'] = test['C_prev4'] - test['D_prev4']
    
    train['net_prev5'] = train['C_prev5'] - train['D_prev5']
    test['net_prev5'] = test['C_prev5'] - test['D_prev5']

    train['net_prev6'] = train['C_prev6'] - train['D_prev6']
    test['net_prev6'] = test['C_prev6'] - test['D_prev6']
    
    
    ## percent change vinary
    train['percent_chng_bin'] = train['Percent_Change_in_Credits'].map(lambda x: 1 if x < 0 else 0)
    test['percent_chng_bin'] = test['Percent_Change_in_Credits'].map(lambda x: 1 if x < 0 else 0)
    
    
    
    ##### RATIO TRAIN FEATURES ############################
    
    ### train prev_1
    train['ratio_1c'] = train['ATM_C_prev1'] / train['C_prev1']
    train['ratio_1d'] = train['ATM_D_prev1'] / train['D_prev1']
    
    train['ratio_2c'] = train['BRANCH_C_prev1'] / train['C_prev1']
    train['ratio_2d'] = train['BRANCH_C_prev1'] / train['D_prev1']
    
    train['ratio_3c'] = train['IB_C_prev1'] / train['C_prev1']
    train['ratio_3d'] = train['IB_D_prev1'] / train['D_prev1']
    
    train['ratio_4c'] = train['MB_C_prev1'] / train['C_prev1']
    train['ratio_4d'] = train['MB_D_prev1'] / train['D_prev1']
    
    train['ratio_5c'] = train['POS_C_prev1'] / train['C_prev1']
    train['ratio_5d'] = train['POS_D_prev1'] / train['D_prev1']
 

     ### train prev_2
    train['ratio_6c'] = train['ATM_C_prev2'] / train['C_prev2']
    train['ratio_6d'] = train['ATM_D_prev2'] / train['D_prev2']
    
    train['ratio_7c'] = train['BRANCH_C_prev2'] / train['C_prev2']
    train['ratio_7d'] = train['BRANCH_C_prev2'] / train['D_prev2']
    
    train['ratio_8c'] = train['IB_C_prev2'] / train['C_prev2']
    train['ratio_8d'] = train['IB_D_prev2'] / train['D_prev2']
    
    train['ratio_9c'] = train['MB_C_prev2'] / train['C_prev2']
    train['ratio_9d'] = train['MB_D_prev2'] / train['D_prev2']
    
    train['ratio_10c'] = train['POS_C_prev2'] / train['C_prev2']
    train['ratio_10d'] = train['POS_D_prev2'] / train['D_prev2']
    
    
    ### train prev_3
    train['ratio_11c'] = train['ATM_C_prev3'] / train['C_prev3']
    train['ratio_11d'] = train['ATM_D_prev3'] / train['D_prev3']
    
    train['ratio_12c'] = train['BRANCH_C_prev3'] / train['C_prev3']
    train['ratio_12d'] = train['BRANCH_C_prev3'] / train['D_prev3']
    
    train['ratio_13c'] = train['IB_C_prev3'] / train['C_prev3']
    train['ratio_13d'] = train['IB_D_prev3'] / train['D_prev3']
    
    train['ratio_14c'] = train['MB_C_prev3'] / train['C_prev3']
    train['ratio_14d'] = train['MB_D_prev3'] / train['D_prev3']

    train['ratio_15c'] = train['POS_C_prev3'] / train['C_prev3']
    train['ratio_15d'] = train['POS_D_prev3'] / train['D_prev3']
    
    ### train prev_4
    train['ratio_11c'] = train['ATM_C_prev4'] / train['C_prev4']
    train['ratio_11d'] = train['ATM_D_prev4'] / train['D_prev4']
    
    train['ratio_12c'] = train['BRANCH_C_prev4'] / train['C_prev4']
    train['ratio_12d'] = train['BRANCH_C_prev4'] / train['D_prev4']
    
    train['ratio_13c'] = train['IB_C_prev4'] / train['C_prev4']
    train['ratio_13d'] = train['IB_D_prev4'] / train['D_prev4']
    
    train['ratio_14c'] = train['MB_C_prev4'] / train['C_prev4']
    train['ratio_14d'] = train['MB_D_prev4'] / train['D_prev4']
    
    train['ratio_15c'] = train['POS_C_prev4'] / train['C_prev4']
    train['ratio_15d'] = train['POS_D_prev4'] / train['D_prev4']
    

    ### train prev5
    train['ratio_16c'] = train['ATM_C_prev5'] / train['C_prev5']
    train['ratio_16d'] = train['ATM_D_prev5'] / train['D_prev5']
    
    train['ratio_17c'] = train['BRANCH_C_prev5'] / train['C_prev5']
    train['ratio_17d'] = train['BRANCH_C_prev5'] / train['D_prev5']
    
    train['ratio_18c'] = train['IB_C_prev5'] / train['C_prev5']
    train['ratio_18d'] = train['IB_D_prev5'] / train['D_prev5']
    
    train['ratio_19c'] = train['MB_C_prev5'] / train['C_prev5']
    train['ratio_19d'] = train['MB_D_prev5'] / train['D_prev5']
    
    train['ratio_20c'] = train['POS_C_prev5'] / train['C_prev5']
    train['ratio_20d'] = train['POS_D_prev5'] / train['D_prev5']
    
       
    ### train prev6
    train['ratio_21c'] = train['ATM_C_prev6'] / train['C_prev6']
    train['ratio_21d'] = train['ATM_D_prev6'] / train['D_prev6']
    
    train['ratio_22c'] = train['BRANCH_C_prev6'] / train['C_prev6']
    train['ratio_22d'] = train['BRANCH_C_prev6'] / train['D_prev6']
    
    train['ratio_23c'] = train['IB_C_prev6'] / train['C_prev6']
    train['ratio_23d'] = train['IB_D_prev6'] / train['D_prev6']
    
    train['ratio_24c'] = train['MB_C_prev6'] / train['C_prev6']
    train['ratio_24d'] = train['MB_D_prev6'] / train['D_prev6']
    
    train['ratio_25c'] = train['POS_C_prev6'] / train['C_prev6']
    train['ratio_25d'] = train['POS_D_prev6'] / train['D_prev6']
    
    
    ###### RATIO TEST FEATURES #####################################3
    
    
       ### test prev_1
    test['ratio_1c'] = test['ATM_C_prev1'] / test['C_prev1']
    test['ratio_1d'] = test['ATM_D_prev1'] / test['D_prev1']
    
    test['ratio_2c'] = test['BRANCH_C_prev1'] / test['C_prev1']
    test['ratio_2d'] = test['BRANCH_C_prev1'] / test['D_prev1']
    
    test['ratio_3c'] = test['IB_C_prev1'] / test['C_prev1']
    test['ratio_3d'] = test['IB_D_prev1'] / test['D_prev1']
    
    test['ratio_4c'] = test['MB_C_prev1'] / test['C_prev1']
    test['ratio_4d'] = test['MB_D_prev1'] / test['D_prev1']
    
    test['ratio_5c'] = test['POS_C_prev1'] / test['C_prev1']
    test['ratio_5d'] = test['POS_D_prev1'] / test['D_prev1']
 

     ### test prev_2
    test['ratio_6c'] = test['ATM_C_prev2'] / test['C_prev2']
    test['ratio_6d'] = test['ATM_D_prev2'] / test['D_prev2']
    
    test['ratio_7c'] = test['BRANCH_C_prev2'] / test['C_prev2']
    test['ratio_7d'] = test['BRANCH_C_prev2'] / test['D_prev2']
    
    test['ratio_8c'] = test['IB_C_prev2'] / test['C_prev2']
    test['ratio_8d'] = test['IB_D_prev2'] / test['D_prev2']
    
    test['ratio_9c'] = test['MB_C_prev2'] / test['C_prev2']
    test['ratio_9d'] = test['MB_D_prev2'] / test['D_prev2']
    
    test['ratio_10c'] = test['POS_C_prev2'] / test['C_prev2']
    test['ratio_10d'] = test['POS_D_prev2'] / test['D_prev2']
    
    
    ### test prev_3
    test['ratio_11c'] = test['ATM_C_prev3'] / test['C_prev3']
    test['ratio_11d'] = test['ATM_D_prev3'] / test['D_prev3']
    
    test['ratio_12c'] = test['BRANCH_C_prev3'] / test['C_prev3']
    test['ratio_12d'] = test['BRANCH_C_prev3'] / test['D_prev3']
    
    test['ratio_13c'] = test['IB_C_prev3'] / test['C_prev3']
    test['ratio_13d'] = test['IB_D_prev3'] / test['D_prev3']
    
    test['ratio_14c'] = test['MB_C_prev3'] / test['C_prev3']
    test['ratio_14d'] = test['MB_D_prev3'] / test['D_prev3']
    
    test['ratio_15c'] = test['POS_C_prev3'] / test['C_prev3']
    test['ratio_15d'] = test['POS_D_prev3'] / test['D_prev3']
    
    ### test prev_4
    test['ratio_11c'] = test['ATM_C_prev4'] / test['C_prev4']
    test['ratio_11d'] = test['ATM_D_prev4'] / test['D_prev4']
    
    test['ratio_12c'] = test['BRANCH_C_prev4'] / test['C_prev4']
    test['ratio_12d'] = test['BRANCH_C_prev4'] / test['D_prev4']
    
    test['ratio_13c'] = test['IB_C_prev4'] / test['C_prev4']
    test['ratio_13d'] = test['IB_D_prev4'] / test['D_prev4']
    
    test['ratio_14c'] = test['MB_C_prev4'] / test['C_prev4']
    test['ratio_14d'] = test['MB_D_prev4'] / test['D_prev4']
    
    test['ratio_15c'] = test['POS_C_prev4'] / test['C_prev4']
    test['ratio_15d'] = test['POS_D_prev4'] / test['D_prev4']
    
    
    ### test prev5
    test['ratio_16c'] = test['ATM_C_prev5'] / test['C_prev5']
    test['ratio_16d'] = test['ATM_D_prev5'] / test['D_prev5']
    
    test['ratio_17c'] = test['BRANCH_C_prev5'] / test['C_prev5']
    test['ratio_17d'] = test['BRANCH_C_prev5'] / test['D_prev5']
    
    test['ratio_18c'] = test['IB_C_prev5'] / test['C_prev5']
    test['ratio_18d'] = test['IB_D_prev5'] / test['D_prev5']
    
    test['ratio_19c'] = test['MB_C_prev5'] / test['C_prev5']
    test['ratio_19d'] = test['MB_D_prev5'] / test['D_prev5']
    
    test['ratio_20c'] = test['POS_C_prev5'] / test['C_prev5']
    test['ratio_20d'] = test['POS_D_prev5'] / test['D_prev5']
    
       
    ### test prev6
    test['ratio_21c'] = test['ATM_C_prev6'] / test['C_prev6']
    test['ratio_21d'] = test['ATM_D_prev6'] / test['D_prev6']
    
    test['ratio_22c'] = test['BRANCH_C_prev6'] / test['C_prev6']
    test['ratio_22d'] = test['BRANCH_C_prev6'] / test['D_prev6']
    
    test['ratio_23c'] = test['IB_C_prev6'] / test['C_prev6']
    test['ratio_23d'] = test['IB_D_prev6'] / test['D_prev6']
    
    test['ratio_24c'] = test['MB_C_prev6'] / test['C_prev6']
    test['ratio_24d'] = test['MB_D_prev6'] / test['D_prev6']
    
    test['ratio_25c'] = test['POS_C_prev6'] / test['C_prev6']
    test['ratio_25d'] = test['POS_D_prev6'] / test['D_prev6']
    
   
    return train, test