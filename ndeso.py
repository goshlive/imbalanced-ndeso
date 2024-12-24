#########################################
### Developed by @I.M. Putrama
### Initial Version: v1.0.
### Budapest, Hungary.
### Date: Apr 28, 2024 02:14 PM
#########################################

import numpy as np

from scipy.spatial import distance

from sklearn.utils import check_array

from imblearn.base import BaseSampler
from imblearn.over_sampling import RandomOverSampler

class NDE(BaseSampler):
    def __init__(self, n_neighbors=5, dist_metric='euclidean', random_state=None):
        self.n_neighbors = n_neighbors
        self.sampling_strategy = 'auto'
        self._sampling_type = 'clean-sampling'
        self.sampling_type = self._sampling_type
        self.dist_metric = dist_metric
        self.random_state = random_state        

    def _fit_resample(self, X_ori, y_ori):
        # Validate input
        X = check_array(X_ori, ensure_2d=True)
        y = check_array(y_ori, ensure_2d=False)
        
        # Calculate pairwise distances between all points
        d = distance.cdist(X, X, metric=self.dist_metric)

        # Compute centroids of each class
        class_labels = np.unique(y)
        centroids = {label: X[y == label].mean(axis=0) for label in class_labels}

        # Get k nearest neighbors' indices
        indexes = np.argsort(d)[:, 1:self.n_neighbors + 1]
        
        # Initialize a set to store the indices for displacement
        idx_disp = set()

        for i in range(len(X)):
            neighbors = indexes[i]

            # Check how many neighbors belong to the same class
            same_class = np.sum(y[neighbors] == y[i])
            diff_class = self.n_neighbors - same_class

            # If the number of different class neighbors is more than half, take this point for displacement
            if diff_class > same_class:
                idx_disp.add(i)
                idx_disp.update(neighbors[y[neighbors] == y[i]])

        for i in idx_disp:
            class_centroid = centroids[y[i]]
            original_distance = distance.cdist([X[i]], [class_centroid], metric=self.dist_metric)[0][0]
            direction_vector = (class_centroid - X[i]) / original_distance
            displacement_distance = np.mean(d[i, indexes[i]])
            X[i] = class_centroid - direction_vector * displacement_distance

        return X, y
        

class NDESO(NDE):
    def __init__(self, n_neighbors=5, sampler=None, dist_metric='euclidean', random_state=None):
        super().__init__(n_neighbors, dist_metric, random_state)
        self.sampler = sampler
        
    def _fit_resample(self, X_ori, y_ori):
        # Resample using NDE
        X, y = super()._fit_resample(X_ori, y_ori)
        
        # Further refinement using a specified oversampler (defult to random oversampler)
        os = RandomOverSampler(random_state=self.random_state)
        if self.sampler:
            os = self.sampler
        return os.fit_resample(X, y)
    
    
