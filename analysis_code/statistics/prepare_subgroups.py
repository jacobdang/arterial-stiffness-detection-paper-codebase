#!/usr/bin/env python3
"""Prepare the study HbA1c groups using cohort-specific mean completion."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_hba1c_groups(frame, hba1c='hba1c_percent', cohort=None):
    result = frame.copy()
    values = pd.to_numeric(result[hba1c], errors='raise')
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError('HbA1c must contain finite measurements or empty values')
    if cohort:
        if result[cohort].isna().any():
            raise ValueError('Each participant needs a cohort assignment')
        means = values.groupby(result[cohort]).transform('mean')
    else:
        means = pd.Series(values.mean(), index=values.index)
    completed = values.fillna(means)
    if completed.isna().any():
        raise ValueError('Each cohort needs at least one observed HbA1c measurement')
    result['hba1c_completed_percent'] = completed
    result['hba1c_group'] = np.where(completed < 7, '<7%', '≥7%')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hba1c', default='hba1c_percent')
    parser.add_argument('--cohort', help='Column defining cohorts when inputs contain more than one cohort')
    args = parser.parse_args()
    result = prepare_hba1c_groups(pd.read_csv(args.input), args.hba1c, args.cohort)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print('Prepared HbA1c groups for {} participants'.format(len(result)))


if __name__ == '__main__':
    main()
