# Preprocessing

`image_and_table/` contains fundus-field cropping, square padding, image resizing, clinical-table preparation, and multiple-imputation routines.

Image inference uses preprocessed 384 × 384 RGB fundus images. Retinal-structure experiments additionally use prepared disc, macular, artery, and vein masks. The mask-to-image transformation definitions are in `main_cls_code_dl/dataloader.py`. Supply locally approved input images, clinical tables, and segmentation masks when running these workflows.
