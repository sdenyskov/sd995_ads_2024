from .config import *

import requests
import pymysql
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import osmnx as ox

"""
import httplib2
import oauth2
import tables
import mongodb
import sqlite
"""

"""
This file is for accessing the data.

Place commands in this file to access the data electronically. 
Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, 
both intellectual property and personal data privacy rights. 
Beyond the legal side also think about the ethical issues around this data.
"""

def hello_world():

    print("Hello from the data science library!")

def download_price_paid_data(year_from, year_to):
    """
    Download UK house price data for given year range
    """

    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"

    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"

    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def create_connection(user, password, host, database, port=3306):
    """ 
    Create a database connection to the MariaDB database 
    specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):

    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"

    # Select the data
    cur = conn.cursor()
    print('Selecting data for year: ' + str(year))
    cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
    rows = cur.fetchall()

    csv_file_path = 'output_file.csv'

    # Write the rows to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(rows)
    
    print('Storing data for year: ' + str(year))
    cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
    print('Data stored for year: ' + str(year))

    conn.commit()

def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    
    distance = distance_km / 111
    north = latitude + distance
    south = latitude - distance
    west = longitude - distance
    east = longitude + distance

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

def normalise_df(data):

    locations = data['location']
    data = data.drop(columns=['location'])
    
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    locations_df = pd.DataFrame({"location": locations})
    normalised_df = pd.concat([locations_df, scaled_data], axis=1)

    return normalised_df

def get_bounds(latitude, longitude, box_dim_km):

    box_dim = box_dim_km / 111
    north = latitude + box_dim / 2
    south = latitude - box_dim / 2
    east = longitude + box_dim / 2
    west = longitude - box_dim / 2

    return (north, south, east, west)

def get_houses_in_the_area(cursor, bounds):

    north, south, east, west = bounds
    columns = ["price", 
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

    # cursor.execute(f"SELECT * FROM `prices_coordinates_data` WHERE latitude BETWEEN {south} AND {north} AND longitude BETWEEN {west} AND {east} AND date_of_transfer >= '2020-01-01'")
    cursor.execute(f"SELECT pcd.*, pp.primary_addressable_object_name, pp.secondary_addressable_object_name, pp.street FROM prices_coordinates_data AS pcd JOIN pp_data AS pp ON pcd.postcode = pp.postcode AND pcd.date_of_transfer = pp.date_of_transfer AND pcd.price = pp.price WHERE pcd.latitude BETWEEN {south} AND {north} AND pcd.longitude BETWEEN {west} AND {east} AND pcd.date_of_transfer >= '2020-01-01'")
    results = cursor.fetchall()

    df = pd.DataFrame(results, columns = columns)

    return df

def get_all_buildings_from_osm(bounds):

    north, south, east, west = bounds

    # Get information about all buildings in the area
    all_buildings_from_osm = ox.geometries_from_bbox(north, south, east, west, tags = {'building': True})

    # Filter polygons only
    all_buildings_from_osm = all_buildings_from_osm[all_buildings_from_osm['geometry'].geom_type == 'Polygon']

    # From geographical coordinate reference system to projected reference system, then compute area in square meters
    all_buildings_from_osm['area_m2'] = all_buildings_from_osm.copy().to_crs(epsg=3395).geometry.area

    return all_buildings_from_osm

def filter_buildings_from_osm(all_buildings_from_osm):
    
    # Filter buildings based on the presence of address
    valid_buildings_from_osm = all_buildings_from_osm[all_buildings_from_osm['addr:street'].notnull() 
                                                    & all_buildings_from_osm['addr:housenumber'].notnull() 
                                                    & all_buildings_from_osm['addr:postcode'].notnull()
                                                    & all_buildings_from_osm['addr:city'].notnull()]
    
    return valid_buildings_from_osm

def data():
    """
    Read the data from the web or local file, 
    returning structured format such as a data frame
    """

    raise NotImplementedError
