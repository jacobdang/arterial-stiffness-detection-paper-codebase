# Clinical and image–clinical models

This directory contains validation-tuned clinical and fusion model workflows. `config.example.yaml` describes the authorized input tables and local paths. Clinical/fusion fitting uses scikit-learn 1.0.2 in a separate environment.

Coefficients for the 20 imputation-specific fusion logistic models are provided in `models/public_release/fusion_lr_20mi/`. Inputs must follow the feature order, encoding, imputation, and normalization described in the coefficient file. The final feature is the study image-model score. Apply each model using `sigmoid(intercept + coefficients @ features)` and average its probability across the 20 imputations.

For complete commands, input schemas, and output filenames, see [the workflow guide](../docs/TRAINING_AND_TESTING.md) and [the figure/table index](../FIGURES_AND_TABLES.md).
