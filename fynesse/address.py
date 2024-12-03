from .config import *

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score



def fit_and_predict(x, y, x_pred, design_matrix = lambda x: x.reshape(-1, 1), regularised = False, alpha = 0.05, L1_wt = 1.0):
    
    # Example of design_matrix: design_matrix = lambda x: sm.add_constant(np.concatenate(((x).reshape(-1, 1), (x**2).reshape(-1, 1), (x**3).reshape(-1, 1)), axis=1))
    
    np.random.seed(42)
    
    if regularised:
        model = sm.OLS(y, design_matrix(x))
        results = model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
        y_pred = results.predict(design_matrix(x_pred))
    else:
        model = sm.OLS(y, design_matrix(x))
        results = model.fit()
        y_pred = results.get_prediction(design_matrix(x_pred)).summary_frame(alpha=0.00)['mean']
    
    return y_pred

def cross_validation(x, y, k):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    r2_list = []
    rmse_list = []
    corr_list = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        y_pred_test = fit_and_predict(x_train, y_train, x_test)
        
        r2_list.append(r2_score(y_test, y_pred_test))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        corr_list.append(np.corrcoef(y_test, y_pred_test)[0, 1])
    
    return (np.mean(r2_list), np.mean(rmse_list), np.mean(corr_list))

def predict_age_profile(nssec_df, age_df, nssec_row_to_predict):
    
    norm_age_df = age_df.div(age_df.sum(axis=1), axis=0)

    preds = []

    for year in range(100):
        x = np.array(nssec_df)
        y = np.array(norm_age_df[year])
        
        model = sm.OLS(y, x)
        result = model.fit()
        y_pred = result.get_prediction(nssec_row_to_predict).summary_frame(alpha=0.00)["mean"]

        preds.append(y_pred[0])
    
    return preds
