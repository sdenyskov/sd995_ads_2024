from .config import *

from . import access, assess
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from pyproj import Proj, transform


def oa_by_coordinates(latitude: float, longitude: float) -> str:

    wgs84 = Proj(init='epsg:4326')  # WGS 84 (latitude, longitude)
    osgb = Proj(init='epsg:27700')  # National Grid (easting, northing)

    x, y = transform(wgs84, osgb, longitude, latitude)

    conn = access.create_connection()
    
    query = f"""
    SELECT oa.OA21CD
    FROM oa_data oa
    WHERE ST_Contains(
        oa.geometry, 
        ST_GeomFromText('POINT({x} {y})', 27700)
    );
    """

    oa = access.execute_query(conn, query)

    return oa[0][0] if oa else None

def get_true_from_census(latitude: float, longitude: float, column: str) -> float:
    """
    Args:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.

    Returns:
    float: True value in selected column.
    """

    oa = oa_by_coordinates(latitude, longitude)

    if oa is None:
        return None
    else:
        conn = access.create_connection()
        
        query = f"""
        SELECT {column}
        FROM census_data_1
        WHERE geography_code = '{oa}';
        """
        true_value = access.execute_query(conn, query)[0][0]

        return float(true_value)

def fit_and_predict(x, y, x_pred, model_name):
    
    if model_name == 'ols':
        model = sm.OLS(y, x)
        results = model.fit()
        y_pred = results.get_prediction(x_pred).summary_frame(alpha=0.00)['mean']
    elif model_name == 'rfr':
        model = RandomForestRegressor(
            n_estimators=100,       # number of trees
            max_depth=10,           # max depth of trees
            random_state=42,        # ensure reproducibility
            n_jobs=-1               # use all processors
        )
        model.fit(x, y)
        y_pred = model.predict(x_pred)
    else:
        raise Exception("Unknown model")
    
    return y_pred

def cross_validation(x, y, k, model_name, plot_pt=False):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmses = []
    r2s = []
    correlations = []

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        y_val_pred = fit_and_predict(x_train, y_train, x_val, model_name=model_name)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred)
        correlation = np.corrcoef(y_val, y_val_pred)[0, 1]

        print(f"RMSE = {rmse:.6f}, R2 = {r2:.4f}, CORR = {100 * correlation:.2f}%")

        rmses.append(rmse)
        r2s.append(r2)
        correlations.append(correlation)

    print()
    print(f"Average rmse across folds: {np.mean(rmses):.6f}")
    print(f"Average r2 across folds: {np.mean(r2s):.4f}")
    print(f"Average correlation across folds: {100 * np.mean(correlations):.2f}%")

    if plot_pt:
        print()
        print(f"Plotting predictions against true values for the last fold:")
        assess.plot_predictions_against_trues(y_val, y_val_pred)

    return (rmses, r2s, correlations)

def estimate_NSSEC_L15_Norm(latitude: float, longitude: float) -> float:
    """
    Args:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.

    Returns:
    float: Estimated share of students in that area (value between 0 and 1).
    """

    conn = access.create_connection()
    access.db_to_csv(conn, 'census_data_1', 'backup/census_data_1.csv')
    census_data_1_df = pd.read_csv('backup/census_data_1.csv')

    oa = oa_by_coordinates(latitude, longitude)
    if oa is None:
        return None
    
    selected_features = ['amenity_nearby', 'tourism_nearby', 'cuisine_nearby', 'sport_nearby', 'religion_nearby', 'military_nearby', 'building_nearby']

    x = census_data_1_df[selected_features].to_numpy()
    y = census_data_1_df['NSSEC_L15_Norm'].to_numpy()
    x_pred = census_data_1_df[census_data_1_df['geography_code'] == oa][selected_features].to_numpy()

    y_pred = fit_and_predict(x, y, x_pred, model_name='rfr')

    return float(y_pred)

def estimate_MI_AddrYearAgoOutUK_Norm(latitude: float, longitude: float) -> float:
    """
    Args:
    latitude (float): The latitude coordinate.
    longitude (float): The longitude coordinate.

    Returns:
    float: Estimated share of migrants that used to live outside of the UK one year ago in that area (value between 0 and 1).
    """
    
    conn = access.create_connection()
    access.db_to_csv(conn, 'census_data_1', 'backup/census_data_1.csv')
    census_data_1_df = pd.read_csv('backup/census_data_1.csv')

    oa = oa_by_coordinates(latitude, longitude)
    if oa is None:
        return None
    
    selected_features = ['amenity_nearby', 'tourism_nearby', 'cuisine_nearby', 'emergency_nearby', 'sport_nearby', 'religion_nearby', 'military_nearby', 'amenity', 'cuisine']
    
    x = census_data_1_df[selected_features].to_numpy()
    y = census_data_1_df['MI_AddrYearAgoOutUK_Norm'].to_numpy()
    x_pred = census_data_1_df[census_data_1_df['geography_code'] == oa][selected_features].to_numpy()

    y_pred = fit_and_predict(x, y, x_pred, model_name='rfr')

    return float(y_pred)

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

"""

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

"""
