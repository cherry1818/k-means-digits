import unittest
import numpy as np
from main import assign_clusters, update_centroids, kmeans

class TestKMeans(unittest.TestCase):
    
    def setUp(self):
        # przykładowe dane do testu
        self.data = np.array([
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0]
        ])
        self.centroids = np.array([
            [1.0, 1.0],
            [5.0, 8.0]
        ])
    
    def test_assign_clusters(self):
        # test, sprawdzający prawidłowość przypisania punktów do centroidów
        clusters = assign_clusters(self.data, self.centroids)
        expected_clusters = np.array([0, 0, 1, 1, 0, 1])
        np.testing.assert_array_equal(clusters, expected_clusters)
    
    def test_update_centroids(self):
        # test, sprawdzający czy centroidy zostały prawidłowo zaktualizowane
        clusters = np.array([0, 0, 1, 1, 0, 1])
        new_centroids = update_centroids(self.data, clusters, 2)
        expected_centroids = np.array([
            [1.16666667, 1.46666667],
            [7.33333333, 9.0]
        ])
        np.testing.assert_almost_equal(new_centroids, expected_centroids, decimal=6)
    
    def test_kmeans_final(self):
        # test sprawdza, czy liczba centroidów jest prawidłowa i czy prawidłowe są ostateczne przypisania punktów do klastrów
        clusters, centroids = kmeans(self.data, k=2, max_iter=10)
        self.assertEqual(len(centroids), 2)
        final_clusters = assign_clusters(self.data, centroids)
        self.assertTrue(np.all(clusters == final_clusters))


unittest.main()
