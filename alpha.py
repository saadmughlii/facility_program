import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from haversine import haversine, Unit
import matplotlib.pyplot as plt
import simplekml
from simplekml import Kml, Style, IconStyle, LabelStyle, Icon
from warnings import simplefilter, filterwarnings
import csv

# Ignoring warnings
simplefilter(action='ignore', category=FutureWarning)
filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")


def calculateDistance(cityCoords, startCoords):
    return haversine(cityCoords, startCoords, unit=Unit.MILES)


def plotKvalues(maxKValue, coordList, inertias):
    print("Please wait while we generate the elbow graph. It may take a moment based on inputs and processing power.")
    k_values = range(1, maxKValue)  # You can adjust this range as needed

    # Calculate inertia for each k
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(coordList)
        inertias.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertias, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()

    # Prompt the user for the number of facilities (K)
    optimal_k = int(input("Enter the number of facilities (K): "))
    return optimal_k


def calculateClusters(optimal_k, coordList, filteredCities):
    # Perform K-Means clustering with the optimal k
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

    clusterLabels = kmeans_optimal.fit_predict(coordList)

    # Add the cluster labels to the filtered dataset
    filteredCities['cluster'] = clusterLabels

    # Define styles for the Placemarks
    servedCity = simplekml.Style()
    servedCity.iconstyle.icon.href = 'datasources/city_icon.png'

    facilityCity = simplekml.Style()
    facilityCity.labelstyle.color = 'ff0b86b8'  # Gold text
    facilityCity.labelstyle.scale = 1.5  # Bigger size
    facilityCity.iconstyle.icon.href = 'datasources/facility_icon.png'
    facilityCity.iconstyle.scale = 2

    # Facility locations
    facilityLocations = kmeans_optimal.cluster_centers_
    facility_names = [f"Facility {i + 1}" for i in range(optimal_k)]

    kml = simplekml.Kml()

    for i, (lat, lon) in enumerate(facilityLocations):
        facilityPoint = kml.newpoint(
            name=facility_names[i],
            coords=[(lon, lat)]
        )
        facilityPoint.style = facilityCity

    for i in range(optimal_k):
        cluster_cities = filteredCities[filteredCities['cluster'] == i]

        # Check if there are cities in this cluster
        if not cluster_cities.empty:
            for _, row in cluster_cities.iterrows():
                lat, lon = row['lat'], row['lng']
                cityPoint = kml.newpoint(
                    name=row['city'],
                    coords=[(lon, lat)]
                )
                cityPoint.style = servedCity

    kml.save("clusters.kml")
    print("KML file saved as clusters.kml")

    # Combine coordinates of cities and facilities
    combined_coordinates = []

    for i in range(optimal_k):
        cluster_center = kmeans_optimal.cluster_centers_[i]
        lat, lon = cluster_center[0], cluster_center[1]
        facility_data = {
            'city name/facility': f'Facility {i + 1}',
            'lat': lat,
            'lng': lon
        }
        combined_coordinates.append(facility_data)

    for i in range(optimal_k):
        cluster_cities = filteredCities[filteredCities['cluster'] == i]

        # Check if there are cities in this cluster
        if not cluster_cities.empty:
            for _, row in cluster_cities.iterrows():
                city_data = {
                    'city name/facility': row['city'],
                    'lat': row['lat'],
                    'lng': row['lng']
                }
                combined_coordinates.append(city_data)

    # Save combined coordinates to a CSV file
    combined_csv_path = "combined_coordinates.csv"
    with open(combined_csv_path, 'w', newline='') as csv_file:
        fieldnames = ['city name/facility', 'lat', 'lng']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_coordinates)

    print(f"Combined coordinates saved to {combined_csv_path}")


def main():
    rawData = pd.read_csv('dataSources/uscities.csv')
    ###########################################################################

    # Taking inputs from the user
    # Getting user input for start city and extracting the coordinates and saving it into a tuple.
    startName = input("Enter the name of your starting city: ")
    if startName not in rawData['city'].values:
        print(f"Error: City name '{startName}' is not found in the dataset.")
        return  # Exit the program
    
    # Taking input for radius in miles around the facilities.
    radiusMiles = input("Enter the radius in miles around the facilities: ")
    radiusMiles = int(radiusMiles)

    ###########################################################################

    # Creation of Data structures

    # This creates a data structure named startcity based on the city name entered by the user. It parses the starting city's
    # latitude and longitude and saves it into a tuple so it stays immutable
    startCity = rawData.loc[rawData['city'] == startName, ['lat', 'lng']].values.tolist()[0]
    startCity = tuple(startCity)

    # Filter cities within the specified radius from the starting city. It
    # calculates the distance between each city in the dataset and the specified city
    # by using python's haversine module
    filteredCities = rawData[rawData.apply(lambda row: calculateDistance((row['lat'], row['lng']), startCity), axis=1) <= radiusMiles].copy()

    # Extract the coordinates of the featured cities
    coordList = filteredCities[['lat', 'lng']]

    # Initialize a list to store inertias
    inertias = []

    ###########################################################################

    # Calculation of K-means cluster using Python's sklearn module.
    # 1 Elbow method:
    # K will determine the number of facilities. It will first apply the elbow method
    # to find the optimal K value. It calculates the sum of distances^2
    # for different values of K and plots it. As of now, the user will look at the optimal
    # point and will decide on the value of K.

    # Define a range of K values to test
    maxKValue = 11  # You can adjust this range as needed
    optimal_k = plotKvalues(maxKValue, coordList, inertias)

    ###########################################################################

    calculateClusters(optimal_k, coordList, filteredCities)


if __name__ == "__main__":
    main()
    
    
#CSV file provided by https://simplemaps.com/data/us-cities
#Free city icon provided by https://www.iconfinder.com/icons/1571981/abstract_basic_circle_dot_geometric_point_shape_icon
#Free facility icon provided by https://www.iconfinder.com/icons/8156581/industry_factory_industrial_pollution_emission_icon