import pandas as pd


def load_atoms(input_path, feature_columns):
    column_names = []
    header_rows = 0

    with open(input_path, 'r') as handle:
        for index, line in enumerate(handle):
            if line.startswith('ITEM: ATOMS'):
                column_names = line.split()[2:]
                header_rows = index + 1
                break

    if not column_names:
        raise ValueError(f"'ITEM: ATOMS' header not found in dump: {input_path}")

    required = ['id', 'x', 'y', 'z', *feature_columns]
    missing = [column for column in required if column not in column_names]
    if missing:
        raise ValueError(f'Columns {missing} not found in dump. Available: {column_names}')

    atoms = pd.read_csv(
        input_path,
        sep=r'\s+',
        skiprows=header_rows,
        index_col=False,
        names=column_names,
    )

    return atoms.reset_index(drop=True)
