#!/usr/bin/env python3
"""Align study prediction outputs and pool probabilities for statistical analysis."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def read_predictions(path, kind):
    if kind == 'image_pickle':
        with Path(path).open('rb') as handle:
            result = pickle.load(handle)
        frame = pd.DataFrame({'participant_id': result['id_set'],
                              'outcome': result['gt_set'], 'score': result['pred_set']})
    elif kind == 'clinical_csv':
        frame = pd.read_csv(path, dtype={'patient_id': str}).rename(
            columns={'patient_id': 'participant_id', 'gt': 'outcome', 'pred': 'score'})
    else:
        raise ValueError('Format must be image_pickle or clinical_csv')
    if frame.participant_id.isna().any() or frame.participant_id.duplicated().any():
        raise ValueError('Prediction files require unique nonempty participant identifiers')
    frame['participant_id'] = frame.participant_id.astype(str)
    frame['outcome'] = pd.to_numeric(frame.outcome.replace({'True': 1, 'False': 0}))
    frame['score'] = pd.to_numeric(frame.score)
    if not frame.outcome.isin([0, 1]).all() or not frame.score.between(0, 1).all():
        raise ValueError('Expected binary outcomes and finite probabilities in [0, 1]')
    return frame.set_index('participant_id')[['outcome', 'score']]


def assemble(spec):
    output = None
    for name, settings in spec.items():
        if name in ['participant_id', 'outcome']:
            raise ValueError('Choose a model score name other than participant_id or outcome')
        if 'path_pattern' in settings:
            paths = [settings['path_pattern'].format(mi=i) for i in settings['imputations']]
        else:
            paths = [settings['path']]
        scores = []
        for path in paths:
            frame = read_predictions(path, settings['format'])
            if output is None:
                output = frame[['outcome']].copy()
            if set(output.index) != set(frame.index):
                raise ValueError('Participant sets differ across prediction files')
            frame = frame.loc[output.index]
            if not np.array_equal(frame.outcome.to_numpy(), output.outcome.to_numpy()):
                raise ValueError('Participant outcomes differ across prediction files')
            scores.append(frame.score.to_numpy())
        if not scores:
            raise ValueError('Supply at least one prediction file per model')
        output[name] = np.mean(scores, axis=0)
    if output is None:
        raise ValueError('Supply at least one model')
    return output.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True,
                        help='JSON model names, formats, and paths; image pickles must be trusted local outputs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = assemble(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
