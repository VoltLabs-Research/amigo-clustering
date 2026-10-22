from __future__ import annotations

from dataclasses import dataclass

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OPTIONS = (
    {'flag': '--n_clusters', 'type': 'int', 'help': 'Number of k-means clusters.', 'default': '2'},
    {'flag': '--n_init', 'type': 'int', 'help': 'Number of centroid seedings.', 'default': '50'},
    {'flag': '--seed', 'type': 'int', 'help': 'Random seed.', 'default': '36'},
    {'flag': '--col_svm', 'type': 'string', 'help': 'Shear-strain column name.', 'default': 'Svm'},
    {'flag': '--col_epot', 'type': 'string', 'help': 'Potential-energy column name.', 'default': 'Epot'},
    {'flag': '--col_vol', 'type': 'string', 'help': 'Atomic-volume column name.', 'default': 'Vol'},
)

DESCRIPTOR = {
    'descriptor': 1,
    'name': 'kmeans',
    'description': 'AMIGO k-means clustering',
    'frameMode': 'single',
    'needsReferenceFrame': False,
    'positional': ['input', 'output_base'],
    'options': list(OPTIONS),
}

if __name__ == '__main__' and '--describe' in sys.argv[1:]:
    print(json.dumps(DESCRIPTOR))
    raise SystemExit(0)

import numpy as np

import clustering
import dump
import outputs
from log import Stopwatch, log

_PARSER_TYPES = {'int': int, 'float': float, 'string': str}


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


def build_parser():
    parser = argparse.ArgumentParser(prog='kmeans', add_help=False)
    parser.add_argument('input_path')
    parser.add_argument('output_base')
    for option in OPTIONS:
        caster = _PARSER_TYPES[option['type']]
        parser.add_argument(option['flag'], type=caster, default=caster(option['default']))
    return parser


def parse_args(args):
    namespace, _ = build_parser().parse_known_args(args)

    return RunConfig(**vars(namespace))


def run(args):
    config = parse_args(args)
    run_timer = Stopwatch()
    log('Parameters: ' + ', '.join(
        f"{option['flag'][2:]}={getattr(config, option['flag'][2:])}" for option in OPTIONS
    ))

    if config.n_clusters < 2:
        raise ValueError('n_clusters must be >= 2')

    log(f'Parsing LAMMPS file: {config.input_path}')
    timer = Stopwatch()
    atoms = dump.load_atoms(config.input_path, config.feature_columns)
    log(f'Successfully loaded {len(atoms)} atoms in {timer.elapsed()}.')
    log(f'Atom columns ({len(atoms.columns)}): {" ".join(str(name) for name in atoms.columns)}')

    cluster_id, cluster_centers = clustering.run_kmeans(
        atoms[config.feature_columns], config.n_clusters, config.n_init, seed=config.seed
    )
    label_map = clustering.assign_labels(cluster_centers, config.n_clusters)
    fcc_id = next((cluster for cluster, label in label_map.items() if label == 'FCC'), None)
    log(f'Labeled cluster {fcc_id} as FCC (lowest {config.col_svm} centroid); ' + ', '.join(
        f'cluster {cluster} as {label}' for cluster, label in sorted(label_map.items()) if cluster != fcc_id
    ))

    atoms_path = outputs.write_atoms_parquet(config.output_base, atoms, cluster_id, label_map, config.feature_columns)
    log(f'Exported atoms data to: {atoms_path}')
    summary_path = outputs.write_summary_parquet(config.output_base, atoms, cluster_id, label_map, config.col_svm)
    log(f'Summary parquet written to {summary_path}')

    result = {
        'ok': True,
        'outputBase': config.output_base,
        'outputs': [atoms_path, summary_path],
        'num_clusters': int(np.unique(cluster_id).size),
    }
    log(f'K-means clustering completed in {run_timer.elapsed()}.')
    return result


def process(frame, config):
    del frame
    return run(config['args'])


if __name__ == '__main__':
    try:
        run(sys.argv[1:])
    except ValueError as error:
        sys.exit(f'kmeans: {error}')
