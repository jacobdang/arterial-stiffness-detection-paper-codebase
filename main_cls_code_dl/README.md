# Bilateral image classification

Training and evaluation use the same image backbone for both eyes, mean feature aggregation, and a gated classification head. Set `PWV_IMAGE_ROOT`, `PWV_SPLIT_ROOT`, and `PWV_OUTPUT_ROOT` before running the Hydra entry points `train_main.py` and `test_main.py`.

The study image score averages probabilities from the five checkpoints ranked by validation AUROC. Internal-test metrics are also recorded after each epoch; checkpoint ranking uses validation AUROC. Selected checkpoint weights are provided under `models/public_release/`.

The backbone-comparison workflow is in `experiments/architecture_comparison/`.

For complete commands, input schemas, and output filenames, see [the workflow guide](../docs/TRAINING_AND_TESTING.md) and [the figure/table index](../FIGURES_AND_TABLES.md).
