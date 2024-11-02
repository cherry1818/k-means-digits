import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans as SKLearnKMeans
from main import kmeans, calculate_accuracy


def test():
    digits = load_digits()
    data = digits.data
    true_labels = digits.target
    k = 10
    clusters, centroids = kmeans(data, k)
    accuracy = calculate_accuracy(clusters, true_labels, k)

    print("Centroidy klastrów:")
    print(centroids)

    print("Przypisania punktów do klastrów:")
    print(clusters)

    print("Dokładność własnej implementacji K-means:", accuracy)

# Testowanie K-means z biblioteki sklearn w celu porównania
def test_sklearn_kmeans():
    digits = load_digits()
    data = digits.data
    true_labels = digits.target
    k = 10

    sklearn_kmeans = SKLearnKMeans(n_clusters=k, n_init=10, random_state=42)
    sklearn_clusters = sklearn_kmeans.fit_predict(data)
    accuracy = calculate_accuracy(sklearn_clusters, true_labels, k)

    print("Dokładność K-means ze sklearn:", accuracy)


test()
test_sklearn_kmeans()
