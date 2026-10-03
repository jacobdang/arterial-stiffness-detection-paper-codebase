# DeiT-Tiny matched-protocol top-five comparison

This runner reuses the `main_cls_code_dl` data loading, augmentation,
sampling, label construction, evaluation, and checkpoint-selection code. It
loads the bundled immutable `timm==0.9.2` package and fails closed on any other
imported version.

The production protocol is the study TinyNet-C protocol: the original
participant split; seed 0; augmentation level 2; half class-balanced plus half
uniform sampling with replacement; 45 epochs; batch size 64; BCE with logits;
AdamW (`lr=0.001`, `weight_decay=0.001`); CyclicLR (`0.0002` to `0.002`,
step-up 1500, gamma 0.99995); gradient clipping at 5; no EMA; one visible GPU;
and arithmetic averaging of probabilities from the five highest validation-
AUROC checkpoints. Test outcomes are not used for checkpoint selection.
For DeiT-Tiny, the class-token vector is mapped to 2,048 dimensions by
`Linear -> BatchNorm1d -> SiLU`; the bilateral mean, channel-attention, and
linear classification stages then match the study image classifier.

The public copy parameterizes site-local paths. Authorized split CSVs must match
the three SHA-256 values embedded in the runner, or execution stops.

```bash
CUDA_VISIBLE_DEVICES=0 python \
  experiments/architecture_comparison/run_top5.py \
  --model deit_tiny --precision amp \
  --data-root AUTHORIZED_DATA/preprocessed_images \
  --split-dir AUTHORIZED_DATA/authorized_splits \
  --output-dir OUTPUTS/deit_top5
```

Compare fixed ensembles only with trusted, authorized prediction artifacts:

```bash
python experiments/architecture_comparison/compare_top5.py \
  --candidate OUTPUTS/deit_top5/top_checkpoint_ensemble_test_result.pickle \
  --reference AUTHORIZED_DATA/tinynet_top5_test_result.pickle \
  --output-dir OUTPUTS/deit_top5/final_statistics \
  --require-candidate-run-manifest
```

## Aggregate comparison

The completed comparison used one seed-0 validation-ranked top-five ensemble
per backbone on the identical 7,331-participant internal-test split:

| Backbone | AUROC (participant-bootstrap 95% CI) |
|---|---:|
| TinyNet-C | 0.778997 (0.767871-0.790462) |
| DeiT-Tiny | 0.710318 (0.697817-0.722775) |

DeiT-Tiny minus TinyNet-C was -0.068679 (paired-bootstrap 95% CI,
-0.078373 to -0.057616; paired DeLong P=8.35e-33). These intervals condition
on the fitted ensembles and do not quantify between-training-seed variation.
The same ptflops 0.7.3 `pytorch` backend estimated 5.52 M parameters/0.88
two-eye GMACs for TinyNet-C and 10.20 M/7.39 two-eye GMACs for DeiT-Tiny.


The study code, protocol, and aggregate comparison tables are included in this release. Participant-level inputs are accessed through the authorized local environment.
