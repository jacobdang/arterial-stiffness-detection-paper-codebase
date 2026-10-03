"""Exercise release commands with synthetic inputs; no model fitting."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]


def run(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                          env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                          capture_output=True, text=True, check=check)


def load_module(relative):
    spec = importlib.util.spec_from_file_location('workflow_module', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_launcher_selects_eye_and_mask_without_creating_run(tmp_path):
    out = tmp_path / 'run'
    result = run('scripts/run_image_model.py', 'train', '--stream', 'right',
                 '--image-root', tmp_path / 'images', '--split-root', tmp_path / 'splits',
                 '--output-dir', out, '--occlusion-root', tmp_path / 'masks',
                 '--occlusion-index', '18', '--dry-run')
    command = json.loads(result.stdout)
    assert 'model=ablation_right_eye_only_tinynet_c' in command['command']
    assert 'occlusion_seg_index=18' in command['command']
    assert command['environment']['PWV_OCCLUSION_ROOT'] == str(tmp_path / 'masks')
    assert not out.exists()
    invalid = run('scripts/run_image_model.py', 'train', '--image-root', tmp_path,
                  '--split-root', tmp_path, '--output-dir', out,
                  '--occlusion-index', '18', '--dry-run', check=False)
    assert invalid.returncode != 0


def test_clinical_dry_run_uses_twenty_imputations_and_both_modes():
    result = run('scripts/run_clinical_models.py', '--config',
                 'main_cls_code_metric_and_fusion/config.example.yaml', '--method', 'rf', '--dry-run')
    plan = json.loads(result.stdout)
    assert plan['method'] == 'rf'
    assert plan['imputations'] == list(range(1, 21))
    assert plan['modes'] == ['metric_only', 'fusion']


def test_fusion_prediction_aligns_participants_and_averages_probabilities(tmp_path):
    payload = json.loads((ROOT / 'models/public_release/fusion_lr_20mi/fusion_lr_20mi_coefficients.json').read_text())
    names = payload['input_contract']['feature_order']
    expected = []
    for mi, model in enumerate(payload['models'], 1):
        x = np.linspace(-0.2, 0.2, 3 * len(names)).reshape(3, -1) + mi * 0.001
        coef = np.array([entry['coefficient'] for entry in model['coefficients']])
        expected.append(expit(model['intercept'] + x @ coef))
        frame = pd.DataFrame(x, columns=names)
        frame.insert(0, 'participant_id', ['synthetic_a', 'synthetic_b', 'synthetic_c'])
        if mi % 2 == 0:
            frame = frame.iloc[::-1]
        frame.to_csv(tmp_path / f'imputation_{mi}.csv', index=False)
    output = tmp_path / 'scores.csv'
    args = ('scripts/predict_fusion.py', '--input-pattern', str(tmp_path / 'imputation_{mi}.csv'), '--output', output)
    run(*args)
    actual = pd.read_csv(output)
    assert actual.participant_id.tolist() == ['synthetic_a', 'synthetic_b', 'synthetic_c']
    np.testing.assert_allclose(actual.fusion_probability, np.mean(expected, axis=0), rtol=1e-12)
    frame.loc[frame.index[0], 'participant_id'] = 'different'
    frame.to_csv(tmp_path / 'imputation_20.csv', index=False)
    result = run(*args, check=False)
    assert result.returncode != 0 and 'participant sets differ' in result.stderr


def test_clinical_only_loading_does_not_require_image_predictions(tmp_path):
    module = load_module('main_cls_code_metric_and_fusion/data_loading_common_funs.py')
    path = tmp_path / 'table.csv'
    pd.DataFrame({'patient_id': ['synthetic_a', 'synthetic_b'], 'param1': [0, 1],
                  'param35': [(1399 - 1634.14) / 349.12, (1401 - 1634.14) / 349.12]}).to_csv(path, index=False)
    train, val, test, combined = module.get_trainval_test_csv(path, path, path, [1], 35, 1400, False, None, None, None)
    assert train.param35.tolist() == [False, True]
    assert 'img_score' not in train
    assert len(combined) == 4


def test_retinal_transformations_cover_all_indices_and_retain_correct_pixels():
    module = load_module('preprocessing_code/retinal_ablation/masking.py')
    image = np.repeat(np.arange(32 * 32, dtype=np.uint8).reshape(32, 32, 1), 3, axis=2)
    masks = []
    for i in range(5):
        mask = np.zeros((32, 32), dtype=bool)
        mask[3 + i:5 + i, 5 + i:8 + i] = True
        masks.append(mask)
    for index, name in module.occlusion_type.items():
        result = module.occlusion_get_processed(image, *masks, name)
        assert result.shape == image.shape and result.dtype == np.uint8
        if index == 0:
            np.testing.assert_array_equal(result, image)
        elif index == 2:
            np.testing.assert_array_equal(result, image * masks[0][:, :, None])
        elif index >= 9:
            np.testing.assert_allclose(result[~np.logical_or.reduce(masks)], image[~np.logical_or.reduce(masks)], atol=1)


def test_statistics_cli_calculates_paired_and_independent_comparisons(tmp_path):
    rng = np.random.RandomState(25)
    y = np.tile([0, 1], 40)
    frame = pd.DataFrame({'participant_id': [f'synthetic_{i}' for i in range(80)],
                          'outcome': y, 'first': expit(y + rng.normal(size=80)),
                          'second': expit(0.8 * y + rng.normal(size=80)),
                          'group': ['a'] * 40 + ['b'] * 40})
    frame.to_csv(tmp_path / 'predictions.csv', index=False)
    run('analysis_code/statistics/analyze_predictions.py', '--input', tmp_path / 'predictions.csv',
        '--scores', 'first', 'second', '--group', 'group', '--bootstrap', '50', '--output-dir', tmp_path / 'results')
    assert len(pd.read_csv(tmp_path / 'results/performance.csv')) == 4
    paired = pd.read_csv(tmp_path / 'results/paired_delong.csv')
    assert len(paired) == 2 and paired.p_value.between(0, 1).all()
    assert len(pd.read_csv(tmp_path / 'results/between_group_delong.csv')) == 2


def test_prediction_export_aligns_imputations_and_rejects_outcome_mismatch(tmp_path):
    import pickle
    module = load_module('scripts/export_prediction_table.py')
    with (tmp_path / 'image.pickle').open('wb') as handle:
        pickle.dump({'id_set': ['a', 'b'], 'gt_set': [False, True], 'pred_set': [0.1, 0.9]}, handle)
    pd.DataFrame({'patient_id': ['a', 'b'], 'gt': [False, True], 'pred': [0.2, 0.8]}).to_csv(tmp_path / 'mi_1.csv', index=False)
    pd.DataFrame({'patient_id': ['b', 'a'], 'gt': [True, False], 'pred': [0.6, 0.4]}).to_csv(tmp_path / 'mi_2.csv', index=False)
    spec = {'image': {'format': 'image_pickle', 'path': str(tmp_path / 'image.pickle')},
            'fusion': {'format': 'clinical_csv', 'path_pattern': str(tmp_path / 'mi_{mi}.csv'), 'imputations': [1, 2]}}
    result = module.assemble(spec)
    np.testing.assert_allclose(result.fusion, [0.3, 0.7])
    pd.DataFrame({'patient_id': ['b', 'a'], 'gt': [False, True], 'pred': [0.6, 0.4]}).to_csv(tmp_path / 'mi_2.csv', index=False)
    import pytest
    with pytest.raises(ValueError, match='outcomes differ'):
        module.assemble(spec)
