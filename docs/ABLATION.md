# Ablation experiments

The study includes retinal-structure experiments, backbone comparisons, and single-eye comparisons. These use distinct trained models. Apply the same prepared image transformation in training and evaluation for a structure-specific run.

## Retinal-structure images

[`masking.py`](../preprocessing_code/retinal_ablation/masking.py) contains the study's retention and inpainting functions. Upstream disc, macular, artery, and vein masks are supplied as approved local inputs. The command below consumes masks that already include the study's preprocessing and vessel dilation and match the 384 × 384 image. The module also contains `process_seg_file` for the study-format segmentation files.

```bash
python preprocessing_code/retinal_ablation/prepare_images.py --image AUTHORIZED_DATA/images/example.jpg --masks AUTHORIZED_DATA/masks/example.npz --output-root AUTHORIZED_DATA/occlusion_images --indices 1 2 3 4 5 6 8 17 18 19 20 21 22 24
```

The NPZ contains Boolean arrays named `artery_map`, `vein_map`, `disc_map`, `macula_map_dd1`, and `macula_map_dd2`. The command produces `<output-root>/<transformation>/<image-stem>.png`. Use the same image stems as the split CSVs. Prepare all images required by the training, validation, and test splits.

## Train and evaluate one structure

Example: artery-only images (index 2):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_image_model.py train --image-root AUTHORIZED_DATA/images --split-root AUTHORIZED_DATA/splits --occlusion-root AUTHORIZED_DATA/occlusion_images --occlusion-index 2 --output-dir OUTPUTS/artery_only
CUDA_VISIBLE_DEVICES=0 python scripts/run_image_model.py test --run-dir OUTPUTS/artery_only
```

For artery removal with biharmonic inpainting, use index 18 and a separate output directory. The principal Figure 4 comparisons use retained structures and biharmonic removal, as shown in [the numerical source table](../results/figure_source_data/figure4_occlusion_metrics.csv). Figure 5 then uses the corresponding participant-level prediction scores in adjusted regression models.

## Transformation index

The following names are the exact directories selected by the dataloader. Index 0 uses the original image in the occlusion directory; an ordinary full-image training run leaves the occlusion index unset.

| Index | Transformation directory |
|---|---|
| 0 | `orig_img` |
| 1 | `retain_vessel` |
| 2 | `retain_artery` |
| 3 | `retain_vein` |
| 4 | `retain_disc` |
| 5 | `retain_macular_1dd` |
| 6 | `retain_macular_2dd` |
| 7 | `retain_all_with_macular_1dd` |
| 8 | `retain_all_with_macular_2dd` |
| 9 | `remove_vessel` |
| 10 | `remove_artery` |
| 11 | `remove_vein` |
| 12 | `remove_disc` |
| 13 | `remove_macular_1dd` |
| 14 | `remove_macular_2dd` |
| 15 | `remove_all_with_macular_1dd` |
| 16 | `remove_all_with_macular_2dd` |
| 17 | `remove_vessel_inpaint_biharmonic` |
| 18 | `remove_artery_inpaint_biharmonic` |
| 19 | `remove_vein_inpaint_biharmonic` |
| 20 | `remove_disc_inpaint_biharmonic` |
| 21 | `remove_macular_1dd_inpaint_biharmonic` |
| 22 | `remove_macular_2dd_inpaint_biharmonic` |
| 23 | `remove_all_with_macular_1dd_inpaint_biharmonic` |
| 24 | `remove_all_with_macular_2dd_inpaint_biharmonic` |

Indices 9–16 use the OpenCV inpainting implementation; indices 17–24 use biharmonic inpainting. These are different image transformations and should be selected explicitly.

## Backbone comparisons (Tables S7–S8)

The image-model configuration choices are `image_only_tinynet_c`, `image_only_resnet18`, `image_only_efficientnet_b0`, `image_only_mobilenetv2`, and `image_only_mobilevitv2`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_image_model.py train --model image_only_resnet18 --image-root AUTHORIZED_DATA/images --split-root AUTHORIZED_DATA/splits --output-dir OUTPUTS/resnet18
```

This selects the architecture with the launcher's current optimization settings. The reported four-backbone sensitivity results used architecture-specific optimization; the released CSVs identify their reported estimates. The matched seed-0 TinyNet-C/DeiT-Tiny protocol has a dedicated [runner and guide](../experiments/architecture_comparison/README.md), including the 192-to-2,048-dimensional DeiT adapter, validation-ranked top-five evaluation, and paired comparisons. Its split-file hashes identify the study splits; configure authorized study inputs for that protocol.

## Single-eye comparisons (Table S9)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_image_model.py train --stream left --image-root AUTHORIZED_DATA/images --split-root AUTHORIZED_DATA/splits --output-dir OUTPUTS/left_eye
CUDA_VISIBLE_DEVICES=0 python scripts/run_image_model.py test --stream left --run-dir OUTPUTS/left_eye
CUDA_VISIBLE_DEVICES=0 python scripts/run_image_model.py train --stream right --image-root AUTHORIZED_DATA/images --split-root AUTHORIZED_DATA/splits --output-dir OUTPUTS/right_eye
CUDA_VISIBLE_DEVICES=0 python scripts/run_image_model.py test --stream right --run-dir OUTPUTS/right_eye
```

`SingleStreamWrapper` selects the corresponding eye from the paired input. For the averaged-single-eye comparison, align left/right `id_set` arrays and average their probabilities before computing metrics. Compare the resulting scores with the bilateral model using a paired DeLong test. [STATISTICAL_ANALYSIS.md](STATISTICAL_ANALYSIS.md) describes the common prediction-table interface.
