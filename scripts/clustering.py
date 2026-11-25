import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class ClusterLabels(dict):
    """The label of each K-means cluster: FCC for the lowest Svm centroid, a grain boundary for every other."""


def run_kmeans(features, n_clusters, n_init, log, seed=36, max_iter=200, scale_range=(0, 10)):
    scaler = MinMaxScaler(feature_range=scale_range)
    scaled = scaler.fit_transform(features)
    log.dump(scaler)

    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=seed,
        max_iter=max_iter,
        n_init=n_init,
        algorithm='elkan',
    )
    labels = model.fit_predict(scaled)
    log.dump(model)

    return labels.astype(np.int32), model.cluster_centers_


def assign_labels(cluster_centers, n_clusters):
    svm_means = [center[0] for center in cluster_centers]
    fcc_id = int(np.argmin(svm_means))

    if n_clusters == 2:
        return ClusterLabels({fcc_id: 'FCC', 1 - fcc_id: 'Grain Boundary'})

    labels = ClusterLabels()
    boundary = 1
    for cluster in range(n_clusters):
        if cluster == fcc_id:
            labels[cluster] = 'FCC'
        else:
            labels[cluster] = f'Grain Boundary {boundary}'
            boundary += 1

    return labels
