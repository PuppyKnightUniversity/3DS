import numpy as np
from sklearn.metrics import pairwise_distances


class KCenterGreedy:
    def __init__(self, features,metric='euclidean'):
        self.features=features
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.features.shape[0]
        self.already_selected = []
        print('Shape of features:', self.features.shape)


    def _update_min_distances(self, cluster_centers, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        new_centers = [c for c in cluster_centers if c not in self.already_selected]
        if new_centers:
            dist_to_new_centers = pairwise_distances(self.features, self.features[new_centers], metric=self.metric)
            if self.min_distances is None:
                self.min_distances = np.min(dist_to_new_centers, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist_to_new_centers)

    def select_batch(self, already_selected, N):
        self.already_selected = already_selected if already_selected is not None else []

        try:
            print('Calculating distances...')
            self._update_min_distances(self.already_selected, reset_dist=True)
        except Exception as e:
            print('Error:', e)
            self._update_min_distances(self.already_selected, reset_dist=False)

        new_batch = []
        for _ in range(N):
            if not self.already_selected:
                ind = np.random.choice(self.n_obs)
            else:
                ind = np.argmax(self.min_distances)
            assert ind not in self.already_selected, "Selected index should not be in already selected list"
            self._update_min_distances([ind], reset_dist=False)
            new_batch.append(ind)
            self.already_selected.append(ind)

        print('Max distance from centers:', max(self.min_distances).item())
        return self.already_selected
