from .config import *

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

# import httplib2
# import oauth2
# import tables
# import mongodb
# import sqlite


################ These functions are used for downloading data and transforming it by the scheme csv <-> db <-> df ################

def download_data_csv(file_url_list, file_name_list):
    """
    Download data from the web in the format of csv.
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

def download_data_zip(file_url, file_name):
    """
    Download data from the web in the format of zip.
    """
    
    # Getting zip
    response = requests.get(file_url)

    if response.status_code == 200:
        print("ZIP file has been downloaded successfully!")

        # Getting csv
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        csv_file = zip_file.open(file_name)

        # Saving csv
        file = open("./" + file_name, "wb")
        file.write(csv_file.read())
        print("File has been saved successfully.")

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

def load_census_data(code, level='msoa'):
    
    return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')

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

def download_and_extract_zip(url, zip_file_path, extraction_folder):
    print("Downloading file")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded zip file to {zip_file_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

    print("Extracting file")
    if zipfile.is_zipfile(zip_file_path):
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            os.makedirs(extraction_folder, exist_ok=True)
            zip_ref.extractall(extraction_folder)
        print(f"Extracted files to {extraction_folder}")
    else:
        print(f"{zip_file_path} is not a valid zip file.")



################ These functions are used for operating with OSM-style data ################



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
    
    # Fetch the boundary polygon for the given place name
    gdf = ox.geocode_to_gdf(place_name)
    
    bounds = gdf.total_bounds

    return bounds

def km_distance(lat1, lon1, lat2, lon2):

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Radius of the Earth in kilometers
    R = 6371.0

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    km_distance = R * c
    return km_distance

def get_all_buildings_from_osm_by_bounds(bounds, tags = {'building': True}):

    (north, south, east, west) = bounds

    # Get information about all buildings in the area
    all_buildings_from_osm = ox.geometries_from_bbox(north, south, east, west, tags)
    
    return all_buildings_from_osm
