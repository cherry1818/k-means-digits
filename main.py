import numpy as np
from sklearn.datasets import load_digits

# Funkcja assign_clusters przypisuje każdy punkt danych do najbliższego centroidu
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = np.linalg.norm(point - centroids, axis=1)  # Odległość punktu od każdego centroidu
        cluster = np.argmin(distances)  # Znalezenie najbliższego centroidu
        clusters.append(cluster)
    return np.array(clusters)


#Funkcja update_centroids oblicza nowe centroidy na podstawie aktualnych przypisań punktów do klastrów
def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        points_in_cluster = data[clusters == i]  # Znalezienie punktów przypisanych do klastra i
        new_centroid = points_in_cluster.mean(axis=0) if len(points_in_cluster) > 0 else np.zeros(data.shape[1])
        #Obliczanie nowego centroida jako srednią z punktów w klastrze i. Jeśli jest pusty, to centroidem staje
        # się wektor zerowy.
        new_centroids.append(new_centroid)
    return np.array(new_centroids)


# Funkcja K-means
def kmeans(data, k, max_iter=100):    # Losowe wybieranie k centroidów z danych
    centroids = data[np.random.choice(data.shape[0], k, replace=False)] #losowo wybierane k centroidów

    for i in range(max_iter):
        clusters = assign_clusters(data, centroids)  # Przypisanie każdego punktu do najbliższego centroidu
        new_centroids = update_centroids(data, clusters, k)  # Wyznaczenie nowych centroidów

        if np.all(new_centroids == centroids):
            break  # Jeśli centroidy się nie zmieniły, przerywamy pętlę

        centroids = new_centroids  # Ustawienie nowych centroidów

    return clusters, centroids
