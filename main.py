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


# Funkcja obliczająca dokładność, porównując klastery z prawdziwymi etykietami
#Oblicza dokładność klastrowania, porównując przypisania klastrów do rzeczywistych etykiet. 
#Dla każdego klastra znajduje najczęściej występującą rzeczywistą etykietę i zlicza, 
#ile punktów w tym klastrze ma tę samą etykietę. Całkowita liczba poprawnych przypisań jest
#dzielona przez całkowitą liczbę punktów danych, aby uzyskać dokładność.
def calculate_accuracy(clusters, true_labels, k):
    correct = 0
    for i in range(k):
        cluster_labels = true_labels[clusters == i]
        if len(cluster_labels) == 0:
            continue
        most_common_label = np.bincount(cluster_labels).argmax()
        correct += (cluster_labels == most_common_label).sum()
    return correct / len(true_labels)
