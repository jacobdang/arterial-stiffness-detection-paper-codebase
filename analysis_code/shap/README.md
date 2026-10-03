# SHAP importance across imputations

Supply the 20 authorized, normalized training tables under `AUTHORIZED_DATA/shap/tables/imputation_1.csv` through `imputation_20.csv`, and the corresponding trusted fitted logistic models under `AUTHORIZED_DATA/shap/models/imputation_1.pickle` through `imputation_20.pickle`. Tables contain `patient_id` and `param1` through `param12`. Supply aligned training image scores through `--image-predictions` (a trusted local file with `id_set` and `pred_set`).

```bash
python analysis_code/shap/run_shap_20mi.py --repo-root . --image-predictions AUTHORIZED_DATA/shap/train_image_scores.pickle --output-dir OUTPUTS/shap_20mi
```

The script calculates mean absolute log-odds SHAP importance for each imputation, then averages across all 20. Error bars show between-imputation standard deviations. Use the training data and matching fitted models; participant-level inputs and manifests remain within the authorized environment.

For the participant distribution in Figure 2B, add `--beeswarm` to the same command. The script uses representative imputation 9, aligns its participants to the image predictions, and colors its signed SHAP contributions using the matching normalized feature values. Select an imputation explicitly with `--beeswarm-imputation 9`. The resulting `shap_representative_beeswarm.png` stays in the authorized environment. Neither mode fits or updates model weights.

Install the explanation dependencies with `python -m pip install -e '.[explainability]'` in a separate environment from image training.
