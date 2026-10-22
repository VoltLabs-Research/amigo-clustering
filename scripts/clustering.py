import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from log import Stopwatch, log, warn


def run_kmeans(features, n_clusters, n_init, seed=36, max_iter=200, scale_range=(0, 10)):
    names = [str(name) for name in features.columns]
    log(f'Feature matrix: {features.shape[0]} atoms x {features.shape[1]} features ({", ".join(names)})')

    scaler = MinMaxScaler(feature_range=scale_range)
    scaled = scaler.fit_transform(features)
    ranges = ', '.join(
        f'{name} {low:.6g}..{high:.6g}' for name, low, high in zip(names, scaler.data_min_, scaler.data_max_)
    )
    log(f'Min-max scaled every feature to [{scale_range[0]}, {scale_range[1]}]: {ranges}')

    model = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=seed,
        max_iter=max_iter,
        n_init=n_init,
        algorithm='elkan',
    )
    log(f'Running K-means: k={n_clusters}, n_init={n_init}, max_iter={max_iter}, init=k-means++, '
        f'algorithm=elkan, seed={seed}...')
    timer = Stopwatch()
    labels = model.fit_predict(scaled)
    log(f'K-means done in {timer.elapsed()}: best of {n_init} initialization(s) took {model.n_iter_} '
        f'iteration(s), inertia {model.inertia_:.6g}')
    if model.n_iter_ >= max_iter:
        warn(f'the best K-means run used all max_iter={max_iter} iterations and may not have converged')

    sizes = np.bincount(labels, minlength=n_clusters)
    for cluster, center in enumerate(scaler.inverse_transform(model.cluster_centers_)):
        centroid = ', '.join(f'{name} {value:.6g}' for name, value in zip(names, center))
        log(f'  Cluster {cluster}: {sizes[cluster]} atoms ({100.0 * sizes[cluster] / len(labels):.1f}%), '
            f'centroid {centroid}')

    return labels.astype(np.int32), model.cluster_centers_


def assign_labels(cluster_centers, n_clusters):
    svm_means = [center[0] for center in cluster_centers]
    fcc_id = int(np.argmin(svm_means))

    if n_clusters == 2:
        return {fcc_id: 'FCC', 1 - fcc_id: 'Grain Boundary'}

    labels = {}
    boundary = 1
    for cluster in range(n_clusters):
        if cluster == fcc_id:
            labels[cluster] = 'FCC'
        else:
            labels[cluster] = f'Grain Boundary {boundary}'
            boundary += 1

    return labels
