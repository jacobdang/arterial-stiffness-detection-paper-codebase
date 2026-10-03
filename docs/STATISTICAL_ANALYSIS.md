# Statistical analyses and figures

The [figure/table index](../FIGURES_AND_TABLES.md) is the complete lookup for all 37 article items. [Reported tables](../results/publication_tables/) preserve the final manuscript values; [numerical analysis outputs](../results/README.md) provide the included aggregate results at their stored precision.

Install the analysis dependencies in a Python 3.8 environment:

```bash
python -m pip install -e '.[analysis]'
```

## Model performance and paired comparisons

Prepare the one-row-per-participant prediction table described in [DATA_FORMATS.md](DATA_FORMATS.md). For outputs from the study training/testing commands, copy [prediction_table.example.json](../examples/prediction_table.example.json), update its file paths, and run:

```bash
python scripts/export_prediction_table.py --config AUTHORIZED_DATA/prediction_table.json --output OUTPUTS/internal_predictions.csv
```

The exporter aligns participant identifiers, verifies matching outcomes, and averages clinical/fusion probabilities across the specified imputations. It accepts trusted local image-result pickles and the `patient_id,gt,pred` CSVs produced by the clinical runner. Use `path` for a single CSV or `path_pattern` with `{mi}` and an `imputations` list. Then calculate statistics:

```bash
python analysis_code/statistics/analyze_predictions.py --input OUTPUTS/internal_predictions.csv --scores image factor fusion --output-dir OUTPUTS/internal_statistics
```

The command calculates participant-bootstrap AUROC intervals, average precision, Brier score, and paired DeLong tests for each model pair. Outputs are `performance.csv` and `paired_delong.csv`. It computes statistics from existing predictions and does not fit prediction models.

[`model_comparisons.py`](../analysis_code/statistics/model_comparisons.py) additionally supplies sensitivity/specificity, IDI, continuous NRI, calibration-regression, and paired-bootstrap routines, including `run_table2_analysis`. Its calibration functions fit statistical calibration models when called. The [Table 2 CSV](../results/publication_tables/Table_2.csv) is the reference for the manuscript's displayed estimates and confidence intervals.

## Subgroups, devices, and participating centers

Figure 3 and Table S3 use cohort-specific mean completion of missing HbA1c measurements before the 7% split. The internal-test groups contain 1,255 and 6,076 participants. Use a separate input for each cohort with `participant_id`, `outcome`, `fusion`, `age_years`, `sex_code` (1 male, 2 female), and `hba1c_percent`; optional `ethnicity_code` uses 1 for Han and 0 for non-Han, with unrecorded ethnicity assigned Han as in the study analysis.

```bash
Rscript analysis_code/statistics/subgroup_analysis.R AUTHORIZED_DATA/internal_predictions.csv OUTPUTS/internal_subgroups
Rscript analysis_code/statistics/subgroup_analysis.R AUTHORIZED_DATA/external_predictions.csv OUTPUTS/external_subgroups
```

The R command requires `pROC`. It uses 1,000 stratified bootstrap resamples, seed 123, independent-sample DeLong comparisons, Youden thresholds, and exact binomial sensitivity/specificity intervals. It analyzes existing predictions without fitting prediction models. The subgroup CSVs in `results/figure_source_data/` contain the article's displayed estimates; full participant data are used by the analysis commands.

For the general Python analysis interface, prepare the HbA1c assignments first:

```bash
python analysis_code/statistics/prepare_subgroups.py --input AUTHORIZED_DATA/internal_predictions.csv --output OUTPUTS/internal_grouped.csv
python analysis_code/statistics/analyze_predictions.py --input OUTPUTS/internal_grouped.csv --scores fusion --group hba1c_group --output-dir OUTPUTS/hba1c_statistics
```

If one input contains more than one cohort, specify its cohort column with `--cohort`. For Tables S16/S17, supply the original device or center-group labels and the image score. The center A/B comparison uses the study's existing assignment.

## Cohort characteristics and imputation

[`cohort_summaries.py`](../analysis_code/statistics/cohort_summaries.py) provides numeric/category summaries, missingness counts, group comparisons, and standardized mean differences for the cohort tables. The reported Tables 1, S5, S14, and S15 are included as CSVs. The multiple-imputation function `mice_imputation_full` is in [`ignore_test_value_multiple_imputation_refactored.R`](../preprocessing_code/image_and_table/ignore_test_value_multiple_imputation_refactored.R); its inputs follow the prepared study-table column layout. Source the R file and call the function with the training, validation, test, and output paths. It estimates the imputation models from the designated training rows.

## Structure-score regression (Figure 5)

See the [retinal-structure analysis guide](../analysis_code/retinal_structure/README.md) for input schemas and R commands. Each of the 15 scores is standardized with the sample SD and fitted separately in unadjusted, age/sex-adjusted, and fully adjusted logistic models. Clinical completion uses existing completed values followed by cohort means for remaining continuous values. The fully adjusted model includes coronary heart disease and UACR on its measurement scale. Confidence intervals use likelihood profiles, and BH adjustment covers 15 score tests within each model. For plotting alone, use the released aggregate results:

```bash
python analysis_code/figures/plot_figure5.py --input results/figure5/figure5_associations.csv --output-dir OUTPUTS/figure5
```

This plotting command does not fit a model. The R runner's `--discrimination` option estimates the separate Table S10 exploratory AUC comparisons. Figure S5 concerns metabolic-marker curves and is supplied as an aggregate visual.

## Vessel correlations (Tables S11–S12)

[`vessel_pipeline.py`](../analysis_code/statistics/vessel_pipeline.py) computes predefined vessel measurements from prepared image/mask inputs. Average the available eyes within each participant and ROI, then run:

```bash
python analysis_code/statistics/run_vessel_correlations.py --input AUTHORIZED_DATA/participant_vessel_means.csv --target pwv --output OUTPUTS/vessels_vs_pwv.csv
python analysis_code/statistics/run_vessel_correlations.py --input AUTHORIZED_DATA/participant_vessel_means.csv --target image_score --output OUTPUTS/vessels_vs_image_score.csv
```

The default is 2,000 participant bootstrap resamples, seed 20260826, available-pair Spearman correlations, and BH correction across 27 tests. These operations do not fit a prediction model.

## Frozen-feature probes (Table S13)

```bash
python analysis_code/frozen_representation/run_analysis.py --config AUTHORIZED_DATA/probe_config.json --mode probe
```

The [probe guide](../analysis_code/frozen_representation/README.md) and [protocol](../analysis_code/frozen_representation/PROTOCOL.md) describe the common 5,187-participant cohort, 27 vessel targets, three predictor sets, and nested cross-validation. This command fits statistical probe models while keeping the image representation fixed. The 81 reported probe results are already included in CSV form.

## SHAP and attribution

[SHAP instructions](../analysis_code/shap/README.md) describe the 20 training-imputation tables and matching fitted fusion models. `run_shap_20mi.py` calculates global importance (Figure 2A). Add `--beeswarm` to also draw Figure 2B from representative imputation 9. Select another specified imputation with `--beeswarm-imputation`; the signed SHAP contributions and feature colors come from that same dataset. Participant-level inputs and plots remain in the authorized environment.

[`integrated_gradients.py`](../analysis_code/attribution/integrated_gradients.py) contains the study's Figure S6 baseline and attribution functions, including `get_baseline`, `integrated_gradients_single_stream`, and `plot_dual_eye_ig_multibaseline`. Call them with the fitted bilateral image model and normalized left/right tensors. The references are the mean image, heavy blur (σ=48), and moderate blur (σ=10). Attribution computes input gradients without updating model weights. Use the article's original retinal illustrations when reading Figure S6.

## Calibration, decision analysis, and costs

[`clinical_utility.py`](../analysis_code/statistics/clinical_utility.py) provides the Hosmer–Lemeshow calculation with an explicit group count and net benefit at a specified threshold. Figure S3 uses the reported adaptive grouping (`g=431`, `df=429`, `P=0.193`) and the accompanying submitted visual. Select the same grouping when calculating that statistic from authorized predictions.

Tables S18–S20 and Figure S7 can be checked using aggregate confusion counts:

```bash
python scripts/verify_public_aggregates.py
python analysis_code/figures/render_figures.py --figures S7 --output-dir OUTPUTS/cost_figure
```

## Render aggregate plots and diagrams

```bash
python analysis_code/figures/render_figures.py --output-dir OUTPUTS/figures
```

The default renders Figure 2A, Figures 4–6, and Figures S1, S4, S7, and S8 from the included aggregate tables and diagram definitions. It creates PNG, PDF, and TIFF files without fitting models. Figure 1 additionally takes a coordinate CSV:

```bash
python analysis_code/figures/render_figures.py --figures 1 --roc-coordinates AUTHORIZED_DATA/figure1_roc_coordinates.csv --output-dir OUTPUTS/figure1
```

Its columns are `cohort`, `model`, `false_positive_rate`, and `true_positive_rate`. Cohort/model labels follow `figure1_auc_metrics.csv`. Generate ROC coordinates from the corresponding participant predictions; an AUC and confidence interval alone do not specify a curve. The figure/table index distinguishes included renderers, submitted aggregate visuals, and author-managed figure inputs.
