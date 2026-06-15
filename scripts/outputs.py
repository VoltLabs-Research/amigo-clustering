import json

import numpy as np
import pandas as pd

IDENTITY_DTYPES = {
    'id': 'uint64',
    'x': 'float64',
    'y': 'float64',
    'z': 'float64',
}


def write_atoms_parquet(output_base, atoms, cluster_id, label_map, feature_columns):
    labels = [label_map[int(cluster)] for cluster in cluster_id]

    table = pd.DataFrame({
        'atom_index': np.arange(len(atoms), dtype='uint32'),
    }).assign(
        **{name: atoms[name].astype(dtype) for name, dtype in IDENTITY_DTYPES.items()}
    ).assign(
        bucket=labels,
        structure_id=cluster_id.astype('int32'),
        structure_name=labels,
        cluster_id=cluster_id.astype('int64'),
        **{name: atoms[name].astype('float64') for name in feature_columns},
    )

    path = f'{output_base}_atoms.parquet'
    table.to_parquet(path, index=False, compression='zstd')

    return path


def write_summary_parquet(output_base, atoms, cluster_id, label_map, col_svm):
    total = len(cluster_id)
    clusters = []

    for cluster in sorted(label_map):
        mask = cluster_id == cluster
        count = int(mask.sum())
        if count == 0:
            continue
        svm = atoms.loc[mask, col_svm].to_numpy(dtype='float64')
        clusters.append({
            'cluster_id': int(cluster),
            'label': label_map[cluster],
            'count': count,
            'percentage': round(count / total * 100, 2),
            'svm_min': round(float(svm.min()), 6),
            'svm_max': round(float(svm.max()), 6),
            'svm_mean': round(float(svm.mean()), 6),
        })

    payload = {
        'main_listing': {
            'total_atoms': total,
            'num_clusters': len(clusters),
            'feature_column': col_svm,
        },
        'sub_listings': {
            'clusters': clusters,
        },
    }

    path = f'{output_base}_kmeans.parquet'
    pd.DataFrame({'payload': [json.dumps(payload)]}).to_parquet(
        path, index=False, compression='zstd'
    )

    return path
