from __future__ import annotations

from dataclasses import dataclass

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import clustering
import dump
import outputs


@dataclass(frozen=True)
class RunConfig:
    input_path: str
    output_base: str
    n_clusters: int
    n_init: int
    seed: int
    col_svm: str
    col_epot: str
    col_vol: str

    @property
    def feature_columns(self):
        return [self.col_svm, self.col_epot, self.col_vol]


def parse_args(args):
    parser = argparse.ArgumentParser(prog='kmeans', add_help=False)
    parser.add_argument('input_path')
    parser.add_argument('output_base')
    parser.add_argument('--n_clusters', type=int, default=2)
    parser.add_argument('--n_init', type=int, default=50)
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--col_svm', default='Svm')
    parser.add_argument('--col_epot', default='Epot')
    parser.add_argument('--col_vol', default='Vol')

    namespace, _ = parser.parse_known_args(args)

    return RunConfig(**vars(namespace))


def run(args):
    config = parse_args(args)

    if config.n_clusters < 2:
        raise ValueError('n_clusters must be >= 2')

    atoms = dump.load_atoms(config.input_path, config.feature_columns)

    cluster_id, cluster_centers = clustering.run_kmeans(
        atoms[config.feature_columns], config.n_clusters, config.n_init, seed=config.seed
    )
    label_map = clustering.assign_labels(cluster_centers, config.n_clusters)

    return {
        'ok': True,
        'outputBase': config.output_base,
        'outputs': [
            outputs.write_atoms_parquet(config.output_base, atoms, cluster_id, label_map, config.feature_columns),
            outputs.write_summary_parquet(config.output_base, atoms, cluster_id, label_map, config.col_svm),
        ],
        'num_clusters': int(np.unique(cluster_id).size),
    }


def process(frame, config):
    del frame
    return run(config['args'])


if __name__ == '__main__':
    try:
        run(sys.argv[1:])
    except ValueError as error:
        sys.exit(f'kmeans: {error}')
