# Frozen-feature analysis protocol

## Representation and cohort

The analysis uses the mean 2,048-dimensional representation from five validation-selected image checkpoints and the corresponding mean image probability. Participant inputs are linked using identifiers and verified input checksums within the approved local environment.

Nine vessel features are measured in three regions (`disc`, `full`, `macula_dd2`), giving 27 targets. The probe analysis uses a common cohort of 5,187 participants with all 27 measurements available. Clinical predictor missingness is handled within training folds.

## Vessel-feature probes

Each vessel measurement is predicted from (1) the scalar image score, (2) the frozen image representation, or (3) age, sex, and HbA1c. Ridge regression uses shuffled five-fold outer cross-validation and five-fold inner tuning. The ridge parameter grid is specified in `config.example.json`. Imputation, centering, and scaling are learned only from the applicable training fold.

Reported metrics are Spearman correlation and cross-validated R-squared, with 2,000 paired participant-bootstrap resamples. P values use 10,000 conditional permutations against fixed out-of-fold predictions. Benjamini–Hochberg correction is applied across 27 targets within each predictor set and metric; an additional correction across all 81 comparisons is calculated. Paired contrasts compare the image representation with the scalar score and clinical predictors.

## Additional incremental analysis

The script also implements an exploratory cross-validated comparison of ten clinical covariates alone, clinical covariates with 27 vessel features, clinical covariates with the image score, and all predictors together. L2 logistic regression is tuned within five-fold outer stratified cross-validation. AUROC, Brier score, and log loss are calculated from outer-fold predictions, with paired participant-bootstrap intervals.

The image model remains fixed in both analyses. Probe associations describe information in the representation and do not establish causal mechanisms or prove that a particular feature drives the image-model prediction. All participant-level outputs remain within the approved institutional environment.
