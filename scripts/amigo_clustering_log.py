from functools import singledispatchmethod

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from clustering import ClusterLabels
from run_log import RunLog, info, warn


class AmigoClusteringLog(RunLog):
    """The execution log of one K-means run.

    Each phase hands it what it built, and the log writes what that holds and how long the phase took.
    """

    def __init__(self, config, options):
        super().__init__('K-means clustering', [
            (option['flag'][2:], getattr(config, option['flag'][2:])) for option in options
        ])
        self._config = config
        self._scaler = None

    @singledispatchmethod
    def dump(self, result):
        super().dump(result)

    @dump.register
    def _(self, atoms: pd.DataFrame):
        info(f'Successfully loaded {len(atoms)} atoms in {self.phase_elapsed()}.')
        info(f'Atom columns ({len(atoms.columns)}): {" ".join(str(name) for name in atoms.columns)}')

    @dump.register
    def _(self, scaler: MinMaxScaler):
        # The K-means line maps the centroids back to the features' own units with it.
        self._scaler = scaler
        low, high = scaler.feature_range
        ranges = ', '.join(
            f'{name} {minimum:.6g}..{maximum:.6g}'
            for name, minimum, maximum in zip(self._feature_names(), scaler.data_min_, scaler.data_max_)
        )
        info(f'Min-max scaled every feature to [{low}, {high}]: {ranges}')

    @dump.register
    def _(self, model: KMeans):
        info(f'K-means done in {self.phase_elapsed()}: best of {model.n_init} initialization(s) took {model.n_iter_} '
             f'iteration(s), inertia {model.inertia_:.6g}')
        if model.n_iter_ >= model.max_iter:
            warn(f'the best K-means run used all max_iter={model.max_iter} iterations and may not have converged')

        labels = model.labels_
        sizes = np.bincount(labels, minlength=model.n_clusters)
        for cluster, center in enumerate(self._scaler.inverse_transform(model.cluster_centers_)):
            centroid = ', '.join(f'{name} {value:.6g}' for name, value in zip(self._feature_names(), center))
            info(f'  Cluster {cluster}: {sizes[cluster]} atoms ({100.0 * sizes[cluster] / len(labels):.1f}%), '
                 f'centroid {centroid}')

    @dump.register
    def _(self, labels: ClusterLabels):
        fcc_id = next((cluster for cluster, label in labels.items() if label == 'FCC'), None)
        info(f'Labeled cluster {fcc_id} as FCC (lowest {self._config.col_svm} centroid); ' + ', '.join(
            f'cluster {cluster} as {label}' for cluster, label in sorted(labels.items()) if cluster != fcc_id
        ))

    def _feature_names(self):
        return [str(name) for name in self._scaler.feature_names_in_]
