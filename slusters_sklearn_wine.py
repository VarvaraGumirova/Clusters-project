import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from copy import deepcopy

wine = datasets.load_wine()
samples = wine.data

x = samples[:,0]
y = samples[:,12]

alcohol_and_proline = np.array(list(zip(x, y)))

# Set random centroids
k = 3

np.random.seed(42)  # Set a fixed seed
centroids_x = np.random.uniform(min(x), max(x), size=k)
centroids_y = np.random.uniform(min(y), max(y), size=k)

centroids = np.array(list(zip(centroids_x, centroids_y)))

# Calculating Eucledean Distance
def distance(a, b):
    one = (a[0] - b[0]) ** 2
    two = (a[1] - b[1]) ** 2
    distance = (one + two) ** 0.5
    return distance


# Cluster labels for each point, initially all zeroes
labels = np.zeros(len(samples))

# Distances to each centroid, initially all zeroes
distances = np.zeros(k)

centroids_old = np.zeros(centroids.shape)

error = np.zeros(3)

# Calculating error
error[0] = distance(centroids[0], centroids_old[0])
error[1] = distance(centroids[1], centroids_old[1])
error[2] = distance(centroids[2], centroids_old[2])

while error.all() != 0:
    for i in range(len(alcohol_and_proline)):
        distances[0] = distance(alcohol_and_proline[i], centroids[0])
        distances[1] = distance(alcohol_and_proline[i], centroids[1])
        distances[2] = distance(alcohol_and_proline[i], centroids[2])
        cluster = np.argmin(distances)
        labels[i] = cluster

    centroids_old = deepcopy(centroids)

    for i in range(k):
        points = []
        for j in range(len(alcohol_and_proline)):
            if labels[j] == i:
                points.append(alcohol_and_proline[j])
        centroids[i] = np.mean(points, axis=0)
        error[0] = distance(centroids[0], centroids_old[0])
        error[1] = distance(centroids[1], centroids_old[1])
        error[2] = distance(centroids[2], centroids_old[2]) 


colors = ['r', 'g', 'b']

# Creating the plot:
for i in range(k):
        points = []
        for j in range(len(alcohol_and_proline)):
            if labels[j] == i:
                points.append(alcohol_and_proline[j])
        points = np.array(points) # convert to np.array for slicing and plotting in the next line, no 2d slicing for lists
        plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)



plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)

plt.xlabel('Alcohol')
plt.ylabel('Proline')
plt.show()



# Calculate Davies-Bouldin Index (DBI)
def calculate_dbi(data, labels, centroids, k):
    # Step 1: Calculate intra-cluster distances (compactness)
    compactness = np.zeros(k)  # Store compactness for each cluster
    for i in range(k):
        points = []  # Collect points in cluster i
        for j in range(len(data)):
            if labels[j] == i:
                points.append(data[j])
        points = np.array(points)
        
        # Calculate the mean distance of points to the centroid
        distances = []
        for point in points:
            dist = distance(point, centroids[i])  # Distance from point to centroid
            distances.append(dist)
        compactness[i] = np.mean(distances)  # Average distance to centroid
        print(f"Compactness for Cluster {i}: {compactness[i]}")  # Print compactness for each cluster

    # Step 2: Calculate inter-cluster distances (separation)
    separation = np.zeros((k, k))  # Distance between centroids
    for i in range(k):
        for j in range(k):
            if i != j:  # No need to calculate separation for the same cluster
                separation[i, j] = distance(centroids[i], centroids[j])
                print(f"Separation between Cluster {i} and Cluster {j}: {separation[i, j]}")

    # Step 3: Calculate DBI
    dbi = 0
    for i in range(k):  # For each cluster
        max_ratio = 0
        for j in range(k):  # Compare with every other cluster
            if i != j:  # Avoid self-comparison
                ratio = (compactness[i] + compactness[j]) / separation[i, j]
                max_ratio = max(max_ratio, ratio)  # Track the maximum ratio for cluster i
        dbi += max_ratio  # Add the maximum ratio to DBI

    dbi = dbi / k  # Average over all clusters
    return dbi

# Call the function to calculate DBI
dbi = calculate_dbi(alcohol_and_proline, labels, centroids, k)
print(f"Davies-Bouldin Index (DBI): {dbi}")
