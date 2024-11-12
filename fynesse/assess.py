from .config import *

from . import access

import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox

"""
import bokeh
import seaborn
import sklearn.decomposition as decomposition
import sklearn.feature_extraction
"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def plot_clusters(clustered_locations):
    cluster_colors = {0: 'red', 1: 'green', 2: 'blue'}

    plt.figure(figsize = (10, 10))
    for cluster, color in cluster_colors.items():
        cluster_data = clustered_locations[clustered_locations['cluster'] == cluster]
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
    plt.title('Map of cities by cluster')
    plt.show()

def plot_distance_matrix_df(distance_matrix_df, locations):

    dim = len(distance_matrix_df)

    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix_df, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Distance in km')

    plt.title("Distance matrix")
    plt.xticks(np.arange(dim), locations, rotation=45, ha='right')
    plt.yticks(np.arange(dim), locations)
    plt.tight_layout()

def plot_correlation_matrix(correlation_matrix):

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

def plot_prices_per_sqm(matched_buildings_from_osm):

    dates = matched_buildings_from_osm['date_of_transfer']
    price_per_sqm = matched_buildings_from_osm["price"] / matched_buildings_from_osm["area_m2"]

    plt.figure(figsize=(10, 6))
    plt.plot(dates, price_per_sqm, marker='o', linestyle='', color='b')

    plt.xlabel('Date')
    plt.ylabel('Price per sqm')
    plt.title('Price per sqm over time')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
