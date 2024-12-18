from .config import *

#### Imports ####

import requests
import pymysql
import csv
import pandas as pd
import osmnx as ox
import yaml
import zipfile
import io
import math
import numpy as np
import os
import subprocess


#### Downloading data ####

def download_data_csv(file_url_list, file_name_list):
    """
    Downloads list of files from the web,
    saves them into specified locations.
    """

    if len(file_url_list) != len(file_name_list):
        raise Exception("file_url_list and file_name_list must be of equal length")

    for i in range(len(file_url_list)):
        file_url = file_url_list[i]
        file_name = file_name_list[i]
        print (f"Downloading data from {file_url}")
        response = requests.get(file_url)
        if response.status_code == 200:
            with open("./" + file_name, "wb") as file:
                file.write(response.content)

def download_data_zip(url, file_name):
    """
    Downloads data from the web in the format of zip, 
    saves specififed file from that zip.
    """
    
    print("Downloading...")
    response = requests.get(url)
    if response.status_code == 200:
        print("ZIP file has been downloaded successfully!")

        zip = zipfile.ZipFile(io.BytesIO(response.content))
        temp = zip.open(file_name)

        file = open("./" + file_name, "wb")
        file.write(temp.read())
        print("File has been saved successfully.")

def download_and_extract_zip(url, zip_path, extraction_folder):
    """
    Downloads data from the web in the format of zip, 
    extracts data into specified folder.
    """
    
    print("Downloading...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded zip file to {zip_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    print("Extracting...")
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            os.makedirs(extraction_folder, exist_ok=True)
            zip_ref.extractall(extraction_folder)
        print(f"Extracted files to {extraction_folder}")
    else:
        print(f"{zip_path} is not a valid zip file.")

def download_census_data(code, base_dir=''):

    url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
    extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"Files already exist at: {extract_dir}.")
        return

    os.makedirs(extract_dir, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Files extracted to: {extract_dir}")


#### Moving data between Dataframes, csv files and datadase ####

def csv_to_df(file_name):

    try:
        df = pd.read_csv("./" + file_name)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def csv_to_db(conn, file_name, table_name, ignore_first_row):

    cur = conn.cursor()

    csv_file_path = './' + file_name
    print(f'Loading data into the table')
    if ignore_first_row:
        cur.execute(f"LOAD DATA LOCAL INFILE '{csv_file_path}' INTO TABLE `{table_name}` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n' IGNORE 1 LINES;")
    else:
        cur.execute(f"LOAD DATA LOCAL INFILE '{csv_file_path}' INTO TABLE `{table_name}` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
    print(f'Data loaded into the table {table_name}')

    conn.commit()

def db_to_csv(conn, table_name, file_name):

    try:
        cur = conn.cursor()

        cur.execute(f"SELECT * FROM `{table_name}`;")
        column_titles = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        
        with open(file_name, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(column_titles)
            writer.writerows(rows)
        print(f"Data successfully exported to {file_name}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.commit()

def df_to_csv(df, file_name):
    
    try:
        df.to_csv("./" + file_name, index=False)
        print(f"DataFrame successfully saved to {file_name}")
    except Exception as e:
        print(f"Error: {e}")

def db_to_df(conn, table_name, columns, sort_by_column):
    
    query = f"SELECT {', '.join(columns)} FROM {table_name} ORDER BY {sort_by_column} ASC"
    r = execute_query(conn, query)
    df = pd.DataFrame(r, columns=columns)

    return df


#### Getting information about files ####

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

def count_rows_in_file(file_name):
    
    result = subprocess.run(['wc', '-l', file_name], capture_output=True, text=True)
    row_count = int(result.stdout.split()[0])
    
    return row_count

def print_columns(file_path):

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        first_row = next(reader)
    
    for i, column_title in enumerate(first_row):
        print(f"Column {i}: {column_title}")

def count_columns(file_path):

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        first_row = next(reader)
        column_count = len(first_row)

    return column_count

#### Computing distances, bounds, getting OSM data ####

def get_bounds(lat, lon, km_box_dimension):
    """
    lat: Latitude of the center point in decimal degrees.
    lon: Longitude of the center point in decimal degrees.
    km_distance: Distance between sides in kilometers.
    Returns (up, down, left, right): border coordinates.
    """

    earth_radius = 6371.0 # km
    km_distance = km_box_dimension / 2

    angular_distance = km_distance / earth_radius
    
    up_lat = lat + math.degrees(angular_distance)
    down_lat = lat - math.degrees(angular_distance)
    
    left_lon = lon - math.degrees(angular_distance / math.cos(math.radians(lat)))
    right_lon = lon + math.degrees(angular_distance / math.cos(math.radians(lat)))
    
    return (up_lat, down_lat, left_lon, right_lon)

def get_bounds_by_name(place_name):
    """
    Fetches the boundary polygon for the given place name.
    """

    gdf = ox.geocode_to_gdf(place_name)
    
    bounds = gdf.total_bounds

    return bounds

def km_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    R = 6371.0
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    km_distance = R * c
    
    return km_distance

def get_all_buildings_from_osm_by_bounds(bounds, tags = {'building': True}):

    (north, south, east, west) = bounds

    all_buildings_from_osm = ox.geometries_from_bbox(north, south, east, west, tags)
    
    return all_buildings_from_osm

def get_osm_data(bounds, tags, features, file_name):
    
    print(f'Getting data within bounds {bounds} to file {file_name}')
    try:
        df = get_all_buildings_from_osm_by_bounds(bounds, tags)
        for feature in features:
            if feature not in df.columns:
                df[feature] = None
        df = df[features]
        df.to_csv("./" + file_name, index=True, header=True)
        print(f'{file_name} has been formed successfully')
    except Exception as e:
        print(f'{file_name} has not been created due to error "{e}"')


#### Pipeline for adding new feature from census dataset ####

def add_census_feature(conn, table, code, column, name, dir = '.'):
    
    download_census_data(code, dir)
    file_path = f'{dir}/census2021-{code.lower()}/census2021-{code.lower()}-oa.csv'
    census_data_df = pd.read_csv(file_path)
    title = census_data_df.columns[column]
    print(f"Original title: {title}")

    # user_input = input("Do you want to proceed with adding this feature? (Press Enter to continue or 'q' to quit): ")
    # if user_input.lower() == 'q':
    #     print("Operation cancelled by the user.")
    #     return

    query = """
    CREATE TABLE IF NOT EXISTS `temp` (
      `date` int unsigned NOT NULL,
      `geography` varchar(9) COLLATE utf8_bin NOT NULL,
      `geography_code` varchar(9) COLLATE utf8_bin NOT NULL,
    """

    for i in range(3, count_columns(file_path)):
        query += f'  `{i}` int unsigned NOT NULL,'

    query += """
      `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
      PRIMARY KEY (`db_id`)
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1;"""

    execute_query(conn, query)

    csv_to_db(conn, file_path, 'temp', ignore_first_row=True)

    create_single_index(conn, 'idx_geography_code', 'temp', 'geography_code')

    column_name = f'{code}_{column}_{name}'

    query = f"ALTER TABLE {table} ADD COLUMN {column_name} int unsigned NOT NULL DEFAULT 0;"

    execute_query(conn, query)
        
    query = f"""
    UPDATE {table}
    SET {column_name} = (
        SELECT temp.{column}
        FROM temp
        WHERE {table}.OA21CD = temp.geography_code
    );
    """

    execute_query(conn, query)

    execute_query(conn, "DROP TABLE temp;")

    print(f"Column {column_name} has been added to the table {table} successfully.")


#### Operations with database

def create_connection(database='ads_2024'):
    """ 
    Create a database connection to the MariaDB database 
    specified by the host url and database name.
    Returns Connection object or None.
    """
    
    with open("credentials.yaml") as file:
        credentials = yaml.safe_load(file)

    conn = None
    try:
        conn = pymysql.connect(user=credentials["username"],
                               passwd=credentials["password"],
                               host=credentials["url"],
                               port=(int)(credentials["port"]),
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    
    return conn

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

def create_table_by_query(conn, table_name, columns, column_names, data_types, constraints):

    if columns != len(column_names):
        raise Exception("columns != len(column_names)")
    elif columns != len(data_types):
        raise Exception("columns != len(data_types)")
    elif columns != len(constraints):
        raise Exception("columns != len(constraints)")
    else:
        query = f'CREATE TABLE {table_name} ({", ".join([f"`{column_names[i]}` {data_types[i]} {constraints[i]}" for i in range(columns)])})'
        print(query)
        execute_query(conn, query)

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
        table_status = execute_query(conn, f"SHOW TABLE STATUS LIKE '{table_name}';")
        approx_row_count = table_status[0][4] if table_status else 'Unable to fetch row count'
        print(f"\nTable {table_name} - Approx Row Count {approx_row_count//100000/10}M")

        column_names = execute_query(conn, f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}';")
        print(tuple(item[0] for item in column_names))

        limit = 3
        first_rows = execute_query(conn, f"SELECT * FROM `{table_name}` LIMIT {limit};")
        for row in first_rows:
            print(row)

        indices = execute_query(conn, f"SHOW INDEX FROM `{table_name}`;")
        if indices:
            print("Indices:")
            for index in indices:
                print(f" - {index[2]} ({index[10]}): Column {index[4]}")
        else:
            print("No indices set on this table.")

def get_summary_on_table(conn, table_name):

    table_status = execute_query(conn, f"SHOW TABLE STATUS LIKE '{table_name}';")
    approx_row_count = table_status[0][4] if table_status else 'Unable to fetch row count'
    print(f"\nTable {table_name} - Approx Row Count {approx_row_count//100000/10}M")

    column_names = execute_query(conn, f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}';")
    print(tuple(item[0] for item in column_names))

    limit = 3
    first_rows = execute_query(conn, f"SELECT * FROM `{table_name}` LIMIT {limit};")
    for row in first_rows:
        print(row)

    indices = execute_query(conn, f"SHOW INDEX FROM `{table_name}`;")
    if indices:
        print("Indices:")
        for index in indices:
            print(f"- Column {index[4]} - Index {index[2]} ({index[10]})")
    else:
        print("No indices set on this table.")
