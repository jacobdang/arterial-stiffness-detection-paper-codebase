# Released model files

| Asset | Files | Use |
|---|---|---|
| Bilateral TinyNet-C, epoch 14 | [safetensors](paper_main_tinynet_c/tinynet_c_epoch0014_network_state_dict.safetensors) | Single-checkpoint image inference |
| Bilateral TinyNet-C, epoch 19 | [safetensors](paper_main_tinynet_c/tinynet_c_epoch0019_network_state_dict.safetensors) | Default single-checkpoint image inference |
| Fusion logistic models, 20 imputations | [JSON](fusion_lr_20mi/fusion_lr_20mi_coefficients.json), [CSV](fusion_lr_20mi/fusion_lr_20mi_coefficients.csv) | Apply normalized clinical inputs plus the study image score; average 20 probabilities |
| File checksums and tensor metadata | [model manifest](public_model_manifest.json) | File verification |

From the repository root, with inference dependencies installed:

```bash
pwv-repro predict-images --left AUTHORIZED_DATA/left.png --right AUTHORIZED_DATA/right.png --device cpu
pwv-repro predict-images --left AUTHORIZED_DATA/left.png --right AUTHORIZED_DATA/right.png --checkpoint models/public_release/paper_main_tinynet_c/tinynet_c_epoch0014_network_state_dict.safetensors --device cpu
python scripts/predict_fusion.py --input-pattern 'AUTHORIZED_DATA/fusion_inputs/imputation_{mi}.csv' --output OUTPUTS/fusion_predictions.csv
```

Image inputs are 384 × 384 preprocessed RGB fundus images. Fusion inputs follow the exact feature order, encoding, imputation, and normalization in the JSON contract. Each image invocation returns an individual-checkpoint score. The study's image ensemble averages five validation-selected checkpoints; each training run writes the checkpoints and evaluation files for that procedure.

The [figure/table index](../../FIGURES_AND_TABLES.md) identifies where these assets relate to the paper. Structure-specific models, single-eye models, and other fitted statistical models are identified by their workflow and result tables; corresponding study inputs are accessed through the approved institutional environment.
