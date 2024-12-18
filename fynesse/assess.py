from .config import *

#### Imports ####

from . import access
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import osmnx as ox
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


#### Database queries ####

def print_counts_per_region(conn, table, column):
    
    n_rows = access.execute_query(conn, f"SELECT COUNT(*) AS row_count FROM {table};")[0][0]
    print(f"Total: {n_rows}")

    for letter in ['E', 'W', 'S', 'N']:
        
        n_rows = access.execute_query(conn, f"SELECT COUNT(*) AS row_count FROM {table} WHERE {column} LIKE '{letter}%';")[0][0]
        print(f"{letter}: {n_rows}")

def add_geography_and_oa_to_pp_data(conn, year, month):
    
    x = f"{year}-{month:02}-01"
    y = f"{year}-{month:02}-31"

    query = f"""
    UPDATE pp_data_with_geography 
    JOIN postcode_data
    ON pp_data_with_geography.postcode = postcode_data.postcode
    SET 
        pp_data_with_geography.easting = postcode_data.easting,
        pp_data_with_geography.northing = postcode_data.northing,
        pp_data_with_geography.geography = ST_GeomFromText(CONCAT('POINT(', postcode_data.easting, ' ', postcode_data.northing, ')'))
    WHERE pp_data_with_geography.date_of_transfer >= '{x}' AND pp_data_with_geography.date_of_transfer <= '{y}';
    """
    
    start = time.time()
    access.execute_query(conn, query)

    query = f"""
    UPDATE pp_data_with_geography
    JOIN oa_data
    ON ST_Within(pp_data_with_geography.geography, oa_data.geometry)
    SET pp_data_with_geography.OA21CD = oa_data.OA21CD
    WHERE pp_data_with_geography.date_of_transfer >= '{x}' 
    AND pp_data_with_geography.date_of_transfer <= '{y}';
    """
    
    access.execute_query(conn, query)
    end = time.time()
    
    print(f"Transactions from {x} to {y} have been processed in {end - start} seconds.")

def add_feature_from_oa_to_pc_data(conn, original_table, feature):
    
    query = f"""
    ALTER TABLE pc_data
    ADD COLUMN {feature} bigint DEFAULT 0 NOT NULL;
    """
    access.execute_query(conn, query)

    query = f"""
    UPDATE pc_data
    JOIN (
        SELECT
            oa_to_pc.PCON25CD,
            SUM({original_table}.{feature}) AS {feature}
        FROM
            oa_to_pc
        JOIN
            {original_table} ON oa_to_pc.OA21CD = {original_table}.OA21CD
        GROUP BY
            oa_to_pc.PCON25CD
    ) AS temp
    ON pc_data.ONS_ID = temp.PCON25CD
    SET pc_data.{feature} = temp.{feature};
    """
    access.execute_query(conn, query)

    print(f"Feature {feature} has been added to pc_data successfully.")

def housing_upload_join_data_csv(conn, start_date, end_date, csv_file_name):
    # start_date = str(year) + "-01-01"
    # end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print(f'Selecting data')
  cur.execute(f'''
    SELECT 
      pp.price, pp.date_of_transfer, pp.postcode, pp.property_type, 
      pp.new_build_flag, pp.tenure_type, pp.primary_addressable_object_name, 
      pp.secondary_addressable_object_name, pp.street, pp.locality, 
      pp.town_city, pp.district, pp.county, po.country, po.latitude, 
      po.longitude, po.easting, po.northing
    FROM (
      SELECT 
        price, date_of_transfer, postcode, property_type, new_build_flag, 
        tenure_type, primary_addressable_object_name, secondary_addressable_object_name, 
        street, locality, town_city, district, county
      FROM pp_data
      WHERE date_of_transfer BETWEEN "{start_date}" AND "{end_date}"
    ) AS pp
    INNER JOIN postcode_data AS po 
    ON pp.postcode = po.postcode;
  ''')
  rows = cur.fetchall()

  # Write the rows to the CSV file
  with open(csv_file_name, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print(f'CSV has been formed. Uploading data to the database')
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_name + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  print(f'Data has been uploaded to the databse successfully')
  conn.commit()

def housing_upload_join_data(conn, start_date, end_date):
    # start_date = str(year) + "-01-01"
    # end_date = str(year) + "-12-31"

    cur = conn.cursor()
    print(f'Selecting and inserting data from {start_date} to {end_date}')
    query = f"""
        INSERT INTO prices_coordinates_data (
            price, date_of_transfer, postcode, property_type, 
            new_build_flag, tenure_type, primary_addressable_object_name, 
            secondary_addressable_object_name, street, locality, town_city, 
            district, county, country, latitude, longitude, easting, northing
        )
        SELECT 
            pp.price, pp.date_of_transfer, pp.postcode, pp.property_type, 
            pp.new_build_flag, pp.tenure_type, pp.primary_addressable_object_name, 
            pp.secondary_addressable_object_name, pp.street, pp.locality, 
            pp.town_city, pp.district, pp.county, po.country, po.latitude, 
            po.longitude, po.easting, po.northing
        FROM pp_data AS pp
        INNER JOIN postcode_data AS po 
        ON pp.postcode = po.postcode
        WHERE pp.date_of_transfer BETWEEN \"{start_date}\" AND \"{end_date}\";
    """
    cur.execute(query)
    conn.commit()
    print(f'Data from {start_date} to {end_date} has been inserted successfully')

def find_correlation_in_table(conn, table, column1, column2):

    query = f"""
    SELECT
        (AVG({column1} * {column2}) - AVG({column1}) * AVG({column2})) /
        (STD({column1}) * STD({column2})) AS correlation
    FROM
        {table}
    """
    
    corr = access.execute_query(conn, query)[0][0]

    return corr

def correlation_matrix_from_table(conn, table, features_list):
    
    correlation_dict = {}

    for i in range(len(features_list)):
        for j in range(i, len(features_list)):
            
            feature1 = features_list[i]
            feature2 = features_list[j]

            correlation = find_correlation_in_table(conn, table, feature1, feature2)

            if feature1 not in correlation_dict:
                correlation_dict[feature1] = {}
            correlation_dict[feature1][feature2] = correlation

            if feature1 != feature2:
                if feature2 not in correlation_dict:
                    correlation_dict[feature2] = {}
                correlation_dict[feature2][feature1] = correlation

    correlation_df = pd.DataFrame(correlation_dict)

    return correlation_df

#### Dataframe operations ####

def normalise_df(data, columns_not_to_normalize = ['location']):

    # separating columns that should not be normalised
    excluded_data = data[columns_not_to_normalize]
    data_to_normalize = data.drop(columns=columns_not_to_normalize)
    
    # normalising remaining columns
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data_to_normalize), columns=data_to_normalize.columns) # add index=data.index
    
    # concatenating excluded and normalised data
    normalised_df = pd.concat([excluded_data, scaled_data], axis=1)

    return normalised_df

def kmeans_clusters(data, n_clusters):
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    clusters = kmeans.labels_

    return clusters


#### Working with OSM data ####

def count_pois_near_coordinates_by_bounds(bounds, tags):
    
    (north, south, west, east) = bounds

    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    pois_df = pd.DataFrame(pois)
    
    poi_counts = {}
    for tag, value in tags.items():
        if tag in pois_df.columns:
            if value is True:
                poi_counts[tag] = pois_df[tag].notnull().sum() # counting all POIs for this tag
            elif isinstance(value, list):
                poi_counts[tag] = pois_df[tag].isin(value).sum() # counting POIs that match one of the list values
            else:
                raise ValueError(f"Unexpected value: tags[{tag}] = {value}")
        else:
            poi_counts[tag] = 0
    
    return poi_counts

def get_houses_from_price_data_by_bounds(conn, bounds):

    (north, south, east, west) = bounds
    min_date = '1900-01-01'

    cur = conn.cursor()
    cur.execute(f"SELECT pcd.*, pp.primary_addressable_object_name, pp.secondary_addressable_object_name, pp.street FROM prices_coordinates_data AS pcd JOIN pp_data AS pp ON pcd.postcode = pp.postcode AND pcd.date_of_transfer = pp.date_of_transfer AND pcd.price = pp.price WHERE pcd.latitude BETWEEN {south} AND {north} AND pcd.longitude BETWEEN {west} AND {east} AND date_of_transfer >= '{min_date}'")
    results = cur.fetchall()

    columns = [
        "price", 
        "date_of_transfer", 
        "postcode", 
        "property_type", 
        "new_build_flag", 
        "tenure_type", 
        "locality", 
        "town_city", 
        "district", 
        "county", 
        "country", 
        "latitude", 
        "longitude", 
        "db_id", 
        "primary_addressable_object_name", 
        "secondary_addressable_object_name", 
        "street"]
    
    df = pd.DataFrame(results, columns = columns)

    conn.commit()

    return df

def filter_buildings_from_osm(all_buildings_from_osm):

    all_buildings_from_osm = all_buildings_from_osm[all_buildings_from_osm['geometry'].geom_type == 'Polygon'] # filtering polygons only

    all_buildings_from_osm['area_m2'] = all_buildings_from_osm.copy().to_crs(epsg=3395).geometry.area
    # transforming from geographical coordinate reference system to projected reference system, then computing area in square meters
    
    valid_buildings_from_osm = all_buildings_from_osm[all_buildings_from_osm['addr:street'].notnull() 
                                                    & all_buildings_from_osm['addr:housenumber'].notnull() 
                                                    & all_buildings_from_osm['addr:postcode'].notnull()
                                                    & all_buildings_from_osm['addr:city'].notnull()] # filtering buildings based on the presence of address
    
    return valid_buildings_from_osm

def capitalize(x):
    return x.upper() if isinstance(x, str) else str(x)

def match_buildings_from_osm(valid_buildings_from_osm, all_houses_from_db):

    matched_buildings_from_osm = valid_buildings_from_osm.copy()

    matched_price = []
    matched_date_of_transfer = []
    matched_latitude = []
    matched_longitude = []

    # iterating through rows of matched_buildings_from_osm
    for _, osm_row in matched_buildings_from_osm.iterrows():
        name = capitalize(osm_row['name'])
        housenumber = capitalize(osm_row['addr:housenumber'])
        street = capitalize(osm_row['addr:street'])
        city = capitalize(osm_row['addr:city'])
        postcode = capitalize(osm_row['addr:postcode'])

        # iterating through rows of all_houses_from_db
        match_found = False
        for _, db_row in all_houses_from_db.iterrows():
            primary_addressable_object_name = capitalize(db_row['primary_addressable_object_name'])
            secondary_addressable_object_name = capitalize(db_row['secondary_addressable_object_name'])
            db_street = capitalize(db_row['street'])
            db_city = capitalize(db_row['town_city'])
            db_postcode = capitalize(db_row['postcode'])
            
            # checking if street, postcode, and city match
            if street == db_street and postcode == db_postcode and city == db_city:
                if (housenumber == primary_addressable_object_name) or (housenumber == secondary_addressable_object_name) or (name in primary_addressable_object_name) or (name in secondary_addressable_object_name):
                    matched_price.append(db_row['price'])
                    matched_date_of_transfer.append(db_row['date_of_transfer'])
                    matched_latitude.append(db_row['latitude'])
                    matched_longitude.append(db_row['longitude'])
                    match_found = True
                    break

        if not match_found:
            matched_price.append(None)
            matched_date_of_transfer.append(None)
            matched_latitude.append(None)
            matched_longitude.append(None)

    # adding matched data as new columns to matched_buildings_from_osm
    matched_buildings_from_osm['price'] = matched_price
    matched_buildings_from_osm['date_of_transfer'] = matched_date_of_transfer
    matched_buildings_from_osm['latitude'] = matched_latitude
    matched_buildings_from_osm['longitude'] = matched_longitude

    # removing rows where no match was found
    matched_buildings_from_osm.dropna(subset=['price', 'date_of_transfer', 'latitude', 'longitude'], inplace=True)

    return matched_buildings_from_osm


#### Plotting ####

def plot_clusters(clustered_data, n_clusters):
    
    colors = sns.color_palette("hsv", n_clusters)  # other options: "Spectral", "cubehelix", etc.

    plt.figure(figsize = (10, 10))

    for cluster, color in enumerate(colors):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster]
        plt.scatter(
            cluster_data['longitude'], 
            cluster_data['latitude'], 
            label=f'Cluster {cluster}', 
            color=color
        )
        for _, row in cluster_data.iterrows():
            plt.text(row['longitude'] + 0.02, row['latitude'] + 0.02, row['location'])

    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Clusters')
    plt.show()

def plot_cluster_matrix(matrix, xlabel, ylabel):

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='coolwarm', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Count')
    plt.title("Cluster Grid")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(range(len(matrix.columns)), matrix.columns)
    plt.yticks(range(len(matrix.index)), matrix.index)
    
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            plt.text(j, i, f"{matrix.iloc[i, j]}", ha='center', va='center', color="black")

    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(data, columns_to_drop):

    correlation_matrix = data.drop(columns_to_drop).corr()

    dim = len(correlation_matrix.columns)

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation Coefficient')
    plt.title("Correlation Matrix of Features")
    plt.xticks(range(dim), correlation_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(dim), correlation_matrix.columns)

    for i in range(dim):
        for j in range(dim):
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()

def plot_correlation_between_dataframes(df1, df2):
    
    correlation_matrix_full = np.zeros((df1.shape[1], df2.shape[1]))
    
    for i, col1 in enumerate(df1.columns):
        for j, col2 in enumerate(df2.columns):
            correlation_matrix_full[i, j] = df1[col1].corr(df2[col2])
    
    correlation_matrix_df = pd.DataFrame(correlation_matrix_full, index=df1.columns, columns=df2.columns)
    # correlation_matrix_df = correlation_matrix_df.applymap(lambda x: np.nan if -0.10 < x < 0.10 else x)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(correlation_matrix_df, annot=True, fmt='.2f', cmap='coolwarm', 
                cbar_kws={'label': 'Correlation'}, ax=ax, linewidths=0.5, linecolor='black', 
                annot_kws={'size': 10, 'weight': 'bold'})

    ax.set_xticklabels(correlation_matrix_df.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(np.arange(len(correlation_matrix_df.index)))
    ax.set_yticklabels(correlation_matrix_df.index, rotation=0, fontsize=10)
        
    # ax.set_xticklabels(correlation_matrix_df.columns, rotation=45, ha='right', fontsize=10)
    # ax.set_yticklabels(correlation_matrix_df.index, rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_predictions_against_trues(y, y_pred, limits = None):
    
    plt.figure(figsize=(8, 6))

    if limits is not None:
        plt.xlim(limits)
        plt.ylim(limits)

    plt.scatter(y, y_pred, color='blue', label='Predictions')
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label="Perfect Prediction")

    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.show()

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    corr = np.corrcoef(y, y_pred)[0, 1]

    print(f"RMSE: {rmse}\nR2: {r2}\nCORRELATION: {corr}")

def plot_predictions_and_trues(x_values, true_values, pred_values, limits = None):
    
    plt.figure(figsize=(8, 6))

    if limits is not None:
        plt.xlim(limits)
        plt.ylim(limits)
    
    plt.plot(x_values, true_values, label=f"True values", color = 'orange')
    plt.plot(x_values, pred_values, label=f"Predicted values", color = 'blue')

    plt.legend()
    plt.show()

def get_ew_boundary_polygon():
    
    uk_boundary_coords = [
        (-5.75, 50.00),
        (-4.45, 51.10),
        (-3.10, 51.25),
        (-2.75, 51.55),
        (-3.35, 51.35),
        (-5.35, 51.75),
        (-4.15, 52.60),
        (-4.60, 53.40),
        (-3.15, 53.50),
        (-3.75, 54.60),
        (-2.00, 55.80),
        (-1.60, 55.60),
        (-1.15, 54.65),
        (-0.50, 54.50),
        (0.20, 53.60),
        (0.45, 53.00),
        (1.30, 53.00),
        (1.80, 52.60),
        (1.60, 52.00),
        (0.80, 51.50),
        (1.70, 51.40),
        (0.35, 50.50),
        (-3.25, 50.60),
        (-3.70, 50.20),
        (-4.70, 50.30),
        (-5.15, 50.00),
        (-5.75, 50.00)
    ]

    return uk_boundary_coords

def plot_buildings(bounds, place_name, layer1, layer2 = None, layer3 = None):

    north, south, east, west = bounds
    
    graph = ox.graph_from_bbox(north, south, east, west) # retrieving main graph
    nodes, edges = ox.graph_to_gdfs(graph) # retrieving nodes and edges
    area = ox.geocode_to_gdf(place_name) # getting place boundary related to the place name as a geodataframe

    fig, ax = plt.subplots()
    area.plot(ax=ax, facecolor="white") # plotting the footprint
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray") # plotting street edges

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # plotting buildings
    layer1.plot(ax=ax, color="grey", alpha=0.7, markersize=10)
    if layer2 is not None:
        layer2.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    if layer3 is not None:
        layer3.plot(ax=ax, color="red", alpha=0.7, markersize=10)
    
    plt.tight_layout()
