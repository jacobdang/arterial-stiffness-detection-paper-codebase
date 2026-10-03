# Input and output formats

All participant-level examples below describe local schemas. They are not distributed patient records. Relative paths are interpreted from the repository root unless a particular function specifies otherwise.

## Image-model input

Use three split CSVs named as listed in [TRAINING_AND_TESTING.md](TRAINING_AND_TESTING.md).

| Field | Meaning |
|---|---|
| `patient_id` | Unique participant identifier within each split; participants do not cross splits |
| `left_img_id`, `right_img_id` | Image filename relative to the image root; one eye may be blank |
| `param35` | Standardized baPWV; raw cm/s = `param35 * 349.12 + 1634.14` |

Each participant needs at least one eye. Images are 384 × 384 RGB after fundus-field cropping and square padding. The image transform uses channel mean and SD 0.5. Structure-specific image directories and mask keys are defined in [ABLATION.md](ABLATION.md).

## Clinical/fusion inputs

Training CSVs contain `patient_id`, the selected `param` columns, and `param35` in the same outcome scale. The 12 inputs are sex, age, diastolic BP, systolic BP, heart rate, BMI, diabetes duration, hypertension, dyslipidemia, cardiovascular-disease history, smoking, and drinking. Use the study's encoded, normalized, and imputed variables. [`fusion_lr_20mi_coefficients.json`](../models/public_release/fusion_lr_20mi/fusion_lr_20mi_coefficients.json) records the exact feature order and coefficient values. These coefficients do not accept raw measurements directly.

Image prediction files used by the clinical fitting code are trusted local pickle dictionaries containing aligned `id_set` and `pred_set` arrays. Image evaluation also supplies `gt_set` and `feat_set`. The 20 imputation datasets must refer to the same participant sets; fusion inference checks this alignment explicitly.

## Statistical analysis inputs

The common discrimination interface is a CSV with one row per participant, `participant_id`, binary `outcome`, and one or more probability columns such as `image`, `factor`, and `fusion`. For subgroups, add a column such as `sex_group`, `hba1c_group`, `device_group`, or `center_group` containing the original analysis assignments. Select the columns with `--scores` and `--group`. Figure 3 HbA1c assignments use cohort-specific mean completion before the 7% split; `prepare_subgroups.py` implements this rule.

Vessel correlations use one row per participant and region, after averaging available eyes: `participant_id`, `roi` (`disc`, `full`, or `macula_dd2`), the nine vessel features in `vessel_correlations.py`, and an aligned PWV or image-score target.

Retinal-structure regressions use the score, clinical, and existing completed-clinical inputs in [`analysis_code/retinal_structure/config.example.yaml`](../analysis_code/retinal_structure/config.example.yaml). Frozen-feature probes use the paths and checksums in [`analysis_code/frozen_representation/config.example.json`](../analysis_code/frozen_representation/config.example.json). Those configurations describe the specific study-table interfaces.

## Aggregate outputs

`results/publication_tables/` contains the 23 reported tables in CSV form, including multirow headings and displayed confidence intervals. `results/figure_source_data/` and the analysis-specific result directories contain numerical aggregate outputs. These are source results for reading, checking, or plotting; they are not participant-level raw data.
