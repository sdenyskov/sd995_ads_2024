from .config import *

# from . import access

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import osmnx as ox
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# import bokeh
# import seaborn
# import sklearn.decomposition as decomposition
# import sklearn.feature_extraction


def csv_preview(file_name):

    try:
        with open("./" + file_name, 'r') as file:
            column_titles = file.readline().strip()
            print("Column Titles:")
            print(column_titles)

            print("\nFirst 10 Rows:")
            for i in range(10):
                row = file.readline().strip()
                if row:
                    print(row)
                else:
                    break
    except Exception as e:
        print(f"Error: {e}")

def execute_query(conn, query):
    """
    Examples are:
    SHOW TABLES;
    SHOW TABLE STATUS LIKE '{table_name}';
    SHOW INDEX FROM `{table_name}`;
    SELECT * FROM `{table_name}` LIMIT {sample_size};
    SELECT count(*) FROM `{table_name}`;
    SELECT MIN({column}) AS min_value, MAX({column}) AS max_value FROM `{table_name}`
    """

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    conn.commit()

    return rows

def create_table_by_query(conn, query, table_name):

    execute_query(conn, f'CREATE TABLE {table_name} AS ({query})')

def create_single_index(conn, index_name, table_name, field_name):

    rows = execute_query(conn, f"CREATE INDEX {index_name} ON {table_name}({field_name});")

    return rows

def create_multiple_index(conn, index_name, table_name, field_names):

    rows = execute_query(conn, f"CREATE INDEX {index_name} ON `{table_name}` ({', '.join(field_names)})")

    return rows

def get_summary_on_db(conn):

    tables = execute_query(conn, "SHOW TABLES;")

    for row in tables:
        table_name = row[0]
        print(f"\nTable: {table_name}")

        table_status = execute_query(conn, f"SHOW TABLE STATUS LIKE '{table_name}';")
        approx_row_count = table_status[0][4] if table_status else 'Unable to fetch row count'
        print("\nApprox Row Count:", approx_row_count//100000/10, "M")

        limit = 5
        first_rows = execute_query(conn, f"SELECT * FROM `{table_name}` LIMIT {limit};")
        print(first_rows)

        indices = execute_query(conn, f"SHOW INDEX FROM `{table_name}`;")
        if indices:
            print("\nIndices:")
            for index in indices:
                print(f" - {index[2]} ({index[10]}): Column {index[4]}")
        else:
            print("\nNo indices set on this table.")



################################



def normalise_df(data, columns_not_to_normalize = ['location']):

    # Separate columns that should not be normalized
    excluded_data = data[columns_not_to_normalize]
    data_to_normalize = data.drop(columns=columns_not_to_normalize)
    
    # Normalize the remaining columns
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data_to_normalize), columns=data_to_normalize.columns) # add index=data.index
    
    # Concatenate excluded and normalized data
    normalised_df = pd.concat([excluded_data, scaled_data], axis=1)

    return normalised_df

def kmeans_clusters(data, n_clusters):
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    clusters = kmeans.labels_

    return clusters



################################



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

def plot_correlation_matrix(data, columns_to_drop):

    correlation_matrix = data.drop(columns_to_drop).corr()

    dim = len(correlation_matrix.columns)

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation Coefficient')
    plt.title("Correlation Matrix of Features")
    plt.xticks(range(dim), correlation_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(dim), correlation_matrix.columns)

    # Add correlation values to cells
    for i in range(dim):
        for j in range(dim):
            plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()

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



################################



def count_pois_near_coordinates_by_bounds(bounds, tags):
    
    (north, south, west, east) = bounds

    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    pois_df = pd.DataFrame(pois)
    
    poi_counts = {}
    for tag, value in tags.items():
        if tag in pois_df.columns:
            if value is True:
                # count all POIs for this tag
                poi_counts[tag] = pois_df[tag].notnull().sum()
            elif isinstance(value, list):
                # count POIs that match one of the list values
                poi_counts[tag] = pois_df[tag].isin(value).sum()
            else:
                # raise an error
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

    # Filter polygons only
    all_buildings_from_osm = all_buildings_from_osm[all_buildings_from_osm['geometry'].geom_type == 'Polygon']

    # From geographical coordinate reference system to projected reference system, then compute area in square meters
    all_buildings_from_osm['area_m2'] = all_buildings_from_osm.copy().to_crs(epsg=3395).geometry.area
    
    # Filter buildings based on the presence of address
    valid_buildings_from_osm = all_buildings_from_osm[all_buildings_from_osm['addr:street'].notnull() 
                                                    & all_buildings_from_osm['addr:housenumber'].notnull() 
                                                    & all_buildings_from_osm['addr:postcode'].notnull()
                                                    & all_buildings_from_osm['addr:city'].notnull()]
    
    return valid_buildings_from_osm

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



################################



def plot_buildings(bounds, place_name, layer1, layer2 = None, layer3 = None):

    north, south, east, west = bounds
    
    # Retrieve main graph
    graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    # Get place boundary related to the place name as a geodataframe
    area = ox.geocode_to_gdf(place_name)

    # Initialise plot
    fig, ax = plt.subplots()

    # Plot the footprint
    area.plot(ax=ax, facecolor="white")

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot buildings
    layer1.plot(ax=ax, color="grey", alpha=0.7, markersize=10)
    if layer2 is not None:
        layer2.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    if layer3 is not None:
        layer3.plot(ax=ax, color="red", alpha=0.7, markersize=10)
    
    plt.tight_layout()
