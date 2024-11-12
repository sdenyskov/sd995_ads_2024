# This file contains code for suporting addressing questions in the data

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

"""
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats
"""

"""Address a particular question that arises from the data"""

def kmeans_clusters(df, n_clusters, normalise):

    data = df.copy().drop(columns=['location'])

    if normalise:
        scaler = MinMaxScaler()
        scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    else:
        scaled_data = data
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(scaled_data)
    clusters = kmeans.labels_

    return clusters

def extend_feature_df(df, locations_dict, clusters):
    
    latitudes = [locations_dict[location][0] for location in locations_dict]
    longitudes = [locations_dict[location][1] for location in locations_dict]

    to_concat = pd.DataFrame({"latitude": latitudes, "longitude": longitudes, "cluster": clusters})
    result = pd.concat([df, to_concat], axis=1)

    return result

def distance(lat1, lon1, lat2, lon2):
    """
    Calculates distance between two points with given coordinates using 111 km approximation
    """
    return np.sqrt((lat1 - lat2) * 111 * (lat1 - lat2) * 111 + (lon1 - lon2) * 111 * (lon1 - lon2) * 111)

def get_distance_matrix_df(osm_feature_counts_df_coor):
    
    # Create distance matrix filled with 0s
    dim = len(osm_feature_counts_df_coor)
    distance_matrix = np.zeros((dim, dim))

    # Calculate distance matrix
    for i in range(dim):
        for j in range(dim):
            distance_matrix[i, j] = distance(
                osm_feature_counts_df_coor.loc[i, 'latitude'], 
                osm_feature_counts_df_coor.loc[i, 'longitude'],
                osm_feature_counts_df_coor.loc[j, 'latitude'], 
                osm_feature_counts_df_coor.loc[j, 'longitude']
            )

    # Convert distance matrix to DataFrame
    distance_matrix_df = pd.DataFrame(distance_matrix,
                                    columns=osm_feature_counts_df_coor['location'],
                                    index=osm_feature_counts_df_coor['location'])
    
    return distance_matrix_df

def capitalize(x):
    return x.upper() if isinstance(x, str) else str(x)

def match_buildings_from_osm(valid_buildings_from_osm, all_houses_from_db):

    # Create a copy of valid_buildings_from_osm
    matched_buildings_from_osm = valid_buildings_from_osm.copy()

    # Create empty lists to store matched data
    matched_price = []
    matched_date_of_transfer = []
    matched_latitude = []
    matched_longitude = []

    # Iterate through the rows of matched_buildings_from_osm
    for _, osm_row in matched_buildings_from_osm.iterrows():
        name = capitalize(osm_row['name'])
        housenumber = capitalize(osm_row['addr:housenumber'])
        street = capitalize(osm_row['addr:street'])
        city = capitalize(osm_row['addr:city'])
        postcode = capitalize(osm_row['addr:postcode'])

        # Iterate through the rows of all_houses_from_db
        match_found = False
        for _, db_row in all_houses_from_db.iterrows():
            primary_addressable_object_name = capitalize(db_row['primary_addressable_object_name'])
            secondary_addressable_object_name = capitalize(db_row['secondary_addressable_object_name'])
            db_street = capitalize(db_row['street'])
            db_city = capitalize(db_row['town_city'])
            db_postcode = capitalize(db_row['postcode'])
            
            # Check if street, postcode, and city match
            if street == db_street and postcode == db_postcode and city == db_city:
                if (housenumber == primary_addressable_object_name) or (housenumber == secondary_addressable_object_name) or (name in primary_addressable_object_name) or (name in secondary_addressable_object_name):
                    matched_price.append(db_row['price'])
                    matched_date_of_transfer.append(db_row['date_of_transfer'])
                    matched_latitude.append(db_row['latitude'])
                    matched_longitude.append(db_row['longitude'])
                    match_found = True
                    break

        # If no match was found, append None to the lists
        if not match_found:
            matched_price.append(None)
            matched_date_of_transfer.append(None)
            matched_latitude.append(None)
            matched_longitude.append(None)

    # Add the matched data as new columns in matched_buildings_from_osm
    matched_buildings_from_osm['price'] = matched_price
    matched_buildings_from_osm['date_of_transfer'] = matched_date_of_transfer
    matched_buildings_from_osm['latitude'] = matched_latitude
    matched_buildings_from_osm['longitude'] = matched_longitude

    # Remove rows where no match was found
    matched_buildings_from_osm.dropna(subset=['price', 'date_of_transfer', 'latitude', 'longitude'], inplace=True)

    return matched_buildings_from_osm
