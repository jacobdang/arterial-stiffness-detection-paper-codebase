# Retinal imaging and arterial stiffness in diabetes

Code accompanying **Deep learning detection and biological interpretation of arterial stiffness from retinal fundus images in diabetes**.

This repository provides the study's main training and testing workflows, retinal-structure and statistical analyses, most analysis code, selected main models, and aggregate results.

## Start here

| What you want to do | Guide |
|---|---|
| Find the code, results, or models for a figure/table | **[Complete figure and table index](FIGURES_AND_TABLES.md)** |
| Train an image model and test its checkpoints | **[Training and testing](docs/TRAINING_AND_TESTING.md)** |
| Run retinal-structure, backbone, or single-eye experiments | **[Ablation experiments](docs/ABLATION.md)** |
| Calculate model performance, comparisons, and associations | **[Statistical analysis](docs/STATISTICAL_ANALYSIS.md)** |
| Inspect the reported tables or numerical source results | **[Results directory guide](results/README.md)** |
| Prepare input files | **[Data formats](docs/DATA_FORMATS.md)** |
| Use the released weights and coefficients | **[Model files and inference](models/public_release/README.md)** |

## Run an included model

Use a dedicated Python 3.8 environment and run commands from the repository root:

```bash
python -m pip install -e '.[inference,test]'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python examples/run_epoch19_cpu_smoke.py --output /tmp/pwv_example.json
```

This applies the epoch-19 checkpoint to the included synthetic images. To use your own preprocessed images:

```bash
pwv-repro predict-images --left AUTHORIZED_DATA/left.png --right AUTHORIZED_DATA/right.png --device cpu
```

Each invocation returns a single-checkpoint score. The study image ensemble averages probabilities from five validation-selected checkpoints; [the training/testing guide](docs/TRAINING_AND_TESTING.md) explains that workflow and its outputs.

## Train, test, and analyze

The commands below illustrate the main entry points. Training commands require authorized images and split tables and perform model training when executed.

```bash
# Train the bilateral TinyNet-C model (CUDA environment).
python scripts/run_image_model.py train --image-root AUTHORIZED_DATA/images --split-root AUTHORIZED_DATA/splits --output-dir OUTPUTS/bilateral

# Test the validation-ranked checkpoints from that run.
python scripts/run_image_model.py test --run-dir OUTPUTS/bilateral

# Find the Figure 4 experiment definitions and numerical results.
python scripts/paper_index.py --item "Figure 4"

# Draw available aggregate plots and diagrams without fitting models.
python analysis_code/figures/render_figures.py --output-dir OUTPUTS/figures
```

The image launcher supports `--dry-run`; its training command also supports `--show-config`. Both inspect settings without training. See the guides for separate environment setup, exact input schemas, ablation indices, clinical model selection, and expected output files.

## Repository contents

| Directory | Contents |
|---|---|
| `main_cls_code_dl/` | Bilateral training, evaluation, and backbone configurations |
| `main_cls_code_dl_ablation_single_stream/` | Left-eye and right-eye training/evaluation |
| `main_cls_code_metric_and_fusion/` | Clinical and fusion model fitting and prediction |
| `preprocessing_code/` | Image/table preparation, imputation, and retinal masking |
| `analysis_code/` | Statistical routines, SHAP, attribution, probes, regression, and figure rendering |
| `models/public_release/` | Two TinyNet-C checkpoints and 20 fusion-model coefficient sets |
| `results/` | All 23 reported tables, numerical source tables, and aggregate supplementary visuals |
| `scripts/` | Workflow launchers, model lookup, and file/result checks |

## Verify the package

```bash
python scripts/paper_index.py --verify
make verify
make test-inference
```

These checks use file integrity, aggregate results, and synthetic examples. They do not train models.

## Data access and citation

Participant-level clinical data and retinal images are not publicly released under hospital policy. Requests may be considered by the corresponding author, Yifei Zhang (**zyf11192@rjh.com.cn**), subject to institutional review and approval. `AUTHORIZED_DATA/` denotes locally approved inputs; keep participant-level outputs in the approved environment.

See [CITATION.cff](CITATION.cff), [data access](docs/DATA_ACCESS.md), and [reuse terms](LICENSE.md).
