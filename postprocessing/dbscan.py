import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def create_dbscan_dataset(anomaly_map):
    mask = anomaly_map.ravel() > 0
    x = np.tile(np.arange(1024), 1024)[mask]
    y = np.repeat(np.arange(1024), 1024)[mask]
    return np.c_[x, y]


def cluster(X):
    dbscan = DBSCAN(eps=1, min_samples=5)
    labels = dbscan.fit_predict(X)
    clusters = [X[labels==i] for i in np.unique(labels)]
    stats = pd.DataFrame({
        'cluster_id': np.unique(labels),
        'mean_x': [c.mean(axis=0)[0] for c in clusters],
        'mean_y': [c.mean(axis=0)[1] for c in clusters],
        'min_x': [c.min(axis=0)[0] for c in clusters],
        'min_y': [c.min(axis=0)[1] for c in clusters],
        'range_x': [(c.max(axis=0) - c.min(axis=0))[0] for c in clusters],
        'range_y': [(c.max(axis=0) - c.min(axis=0))[1] for c in clusters]
    })
    return labels, clusters, stats