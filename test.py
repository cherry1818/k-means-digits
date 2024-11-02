import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans as SKLearnKMeans
from main import kmeans

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

# Uruchomienie testów
if __name__ == "__main__":
    print("Uruchamianie testów:")
    test()
    test_sklearn_kmeans()
