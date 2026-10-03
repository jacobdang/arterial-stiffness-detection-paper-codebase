# Training and testing

Run commands from the repository root. `AUTHORIZED_DATA` contains local, approved inputs; `OUTPUTS` is a local output directory. Training and regression commands below fit models when executed. Use the supplied aggregate results or inference examples when you only want to inspect the study outputs.

## Environments

Use separate Python 3.8 environments:

```bash
# Image training/evaluation and backbone comparisons; a CUDA GPU is required.
python -m pip install -e '.[training,test]'

# In a separate environment: selected-model CPU/GPU inference.
python -m pip install -e '.[inference,test]'

# In a separate environment: clinical/fusion fitting with scikit-learn 1.0.2.
python -m pip install -r requirements/clinical.txt
```

Training uses timm 0.9.2; the selected-checkpoint inference interface uses timm 0.6.7. Image training initializes the backbone from ImageNet weights through timm. Make these pretrained weights available through the normal timm cache or download mechanism. The prepared image and clinical inputs are described in [DATA_FORMATS.md](DATA_FORMATS.md).

## 1. Prepare the image splits

The split directory contains three participant-level CSVs:

- `train_table_orig_random_img_only_patient_level.csv`
- `val_table_orig_random_img_only_patient_level.csv`
- `test_table_orig_random_img_only_patient_level.csv`

Each has `patient_id`, `left_img_id`, `right_img_id`, and `param35`. **`param35` is standardized baPWV, not raw cm/s**: the loader recovers `baPWV = param35 × 349.12 + 1634.14`, then defines AS as baPWV ≥1,400 cm/s. Image fields contain names relative to the image directory. A single available eye is duplicated into the two input positions by the loader. Images are fundus-field cropped, square-padded RGB images at 384 × 384 pixels.

## 2. Train the bilateral model

Inspect the command or resolved settings first:

```bash
python scripts/run_image_model.py train --image-root AUTHORIZED_DATA/images --split-root AUTHORIZED_DATA/splits --output-dir OUTPUTS/bilateral --dry-run
python scripts/run_image_model.py train --image-root AUTHORIZED_DATA/images --split-root AUTHORIZED_DATA/splits --output-dir OUTPUTS/bilateral --show-config
```

Run training with the same command, omitting the inspection option:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_image_model.py train --image-root AUTHORIZED_DATA/images --split-root AUTHORIZED_DATA/splits --output-dir OUTPUTS/bilateral
```

The default configuration uses seed 0, 45 epochs, batch size 64, AdamW, BCE with logits, CyclicLR, mixed precision, and gradient clipping at 5. Configuration files are under [`main_cls_code_dl/conf/`](../main_cls_code_dl/conf/). The launcher exposes `--epochs`, `--workers`, and `--seed`; keep the study settings for the corresponding protocol.

The launcher calls [`train_main.py`](../main_cls_code_dl/train_main.py) with absolute input/output paths. The run directory contains `train_main.log`, `.hydra/`, and `checkpoint/epoch*.pth`. Checkpoint filenames record validation AUROC. Internal-test metrics are also recorded during training; validation AUROC determines checkpoint selection.

## 3. Test the trained checkpoints

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_image_model.py test --run-dir OUTPUTS/bilateral
```

This calls [`test_main.py`](../main_cls_code_dl/test_main.py), selects the five checkpoints with the highest validation AUROC, and evaluates training, validation, and internal-test participants with deterministic image transforms. It writes:

| Output | Meaning |
|---|---|
| `best_checkpoint_train_result.pickle`, `best_checkpoint_val_result.pickle`, `best_checkpoint_test_result.pickle` | Results from the best validation checkpoint |
| `top_checkpoint_ensemble_train_result.pickle`, `top_checkpoint_ensemble_val_result.pickle`, `top_checkpoint_ensemble_test_result.pickle` | Arithmetic mean of probabilities and features across the selected checkpoints |
| `test_main.log` | Selected checkpoint names and evaluation summaries |

Prediction files contain `config`, `gt_set` (binary outcome), `pred_set` (probability), `id_set` (participant identifier), and `feat_set` (image features). Keep these participant-level files in the authorized environment. They are also the inputs to the clinical/fusion pipeline and frozen-feature analysis.

The two distributed safetensors files use the separate `pwv-repro predict-images` interface, which returns an individual-checkpoint score. They are directly usable without a training-run log. See [model inventory](../models/public_release/README.md).

## 4. Train and test clinical/fusion models

Copy [`config.example.yaml`](../main_cls_code_metric_and_fusion/config.example.yaml) to your local configuration and set absolute paths to the 20 prepared imputation tables and aligned image predictions. `$X$` in a table path is replaced with the imputation index. For a clinical-only run, set `fusion: false`.

```bash
python scripts/run_clinical_models.py --config AUTHORIZED_DATA/fusion_config.yaml --method lr --dry-run
python scripts/run_clinical_models.py --config AUTHORIZED_DATA/fusion_config.yaml --method lr
```

The method choices are `lr`, `lda`, `lsvm`, `rbfsvm`, `rf`, and `gbc`. The functions and tuning grids are in [`main_basic.py`](../main_cls_code_metric_and_fusion/main_basic.py). Hyperparameters are selected by validation AUROC, and the selected estimator is evaluated on each split. Results are written under `method_<method>_basic/mice_imputation_<index>/metric_only/` and `/fusion/`: fitted model files, train/validation/test prediction CSVs, and a log. Use a new output directory for each run.

For the multi-factor model, use `param1`–`param12`; fusion adds `img_score`. The main Table 2 single-factor analyses use the relevant one-variable feature sets. Across imputations, average probabilities for the same participants after identifier alignment, then perform the reported statistical comparisons. The [prediction exporter](../scripts/export_prediction_table.py) does this alignment and pooling; see its [example configuration](../examples/prediction_table.example.json) and the [analysis commands](STATISTICAL_ANALYSIS.md).

## 5. Apply the released fusion coefficients

Supply 20 matching input CSVs with `participant_id`, normalized `param1`–`param12`, and the aligned study `img_score`:

```bash
python scripts/predict_fusion.py --input-pattern 'AUTHORIZED_DATA/fusion_inputs/imputation_{mi}.csv' --output OUTPUTS/fusion_predictions.csv
```

This performs inference using the released coefficients and averages probabilities across the 20 imputations. It does not fit a model or derive clinical normalization/imputation from raw measurements.

For ablations, continue to [ABLATION.md](ABLATION.md). For model comparisons and source-table lookup, continue to [STATISTICAL_ANALYSIS.md](STATISTICAL_ANALYSIS.md) and the [figure/table index](../FIGURES_AND_TABLES.md).
