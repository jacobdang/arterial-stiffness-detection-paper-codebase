import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
import cv2
import torch
import os.path as pathlib
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import albumentations as alb
import albumentations.augmentations.transforms as transforms
import albumentations.core.composition as composition
from utils import logging_info


occlusion_type = {0: 'orig_img', 1: 'retain_vessel', 2: 'retain_artery', 3: 'retain_vein', 4: 'retain_disc',
                  5: 'retain_macular_1dd', 6: 'retain_macular_2dd', 7: 'retain_all_with_macular_1dd',
                  8: 'retain_all_with_macular_2dd', 9: 'remove_vessel', 10: 'remove_artery', 11: 'remove_vein',
                  12: 'remove_disc', 13: 'remove_macular_1dd', 14: 'remove_macular_2dd',
                  15: 'remove_all_with_macular_1dd', 16: 'remove_all_with_macular_2dd',
                  17: 'remove_vessel_inpaint_biharmonic',  18: 'remove_artery_inpaint_biharmonic',
                  19: 'remove_vein_inpaint_biharmonic', 20: 'remove_disc_inpaint_biharmonic',
                  21: 'remove_macular_1dd_inpaint_biharmonic', 22: 'remove_macular_2dd_inpaint_biharmonic',
                  23: 'remove_all_with_macular_1dd_inpaint_biharmonic',
                  24: 'remove_all_with_macular_2dd_inpaint_biharmonic'}


def transform_gen_base(config):
    # referrence:
    # https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/
    # https://albumentations.ai/docs/api_reference/augmentations/transforms/

    train_trsfm = composition.Compose([composition.OneOf([
        alb.RandomRotate90(), transforms.Flip(), transforms.Transpose()],
        p=config['p_aug_rotate90_flip_transpose']),
        alb.ShiftScaleRotate(shift_limit=config['aug_shift_limit'], scale_limit=config['aug_scale_limit'],
                             rotate_limit=config['aug_rotate_limit'], border_mode=cv2.BORDER_WRAP,
                             p=config['p_aug_shiftscalerotate']),
        composition.OneOf([
            transforms.Blur(blur_limit=config['aug_blur_limit']),
            transforms.GaussianBlur(blur_limit=config['aug_blur_limit']),
            transforms.MedianBlur(blur_limit=config['aug_blur_limit']),
            transforms.MotionBlur(blur_limit=config['aug_blur_limit'])],
            p=config['p_aug_blur']),
        transforms.RandomBrightnessContrast(brightness_limit=config['aug_brightness_limit'],
                                            contrast_limit=config['aug_contrast_limit'],
                                            p=config['p_aug_brightness_contrast']),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0)], p=1)

    test_trsfm = composition.Compose([transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                                                           max_pixel_value=255.0)], p=1)
    return train_trsfm, test_trsfm


def transform_gen(orig_config):
    transform_config = dict()
    if orig_config['aug_level'] == 0:
        transform_config['p_aug_rotate90_flip_transpose'] = 0.5
        transform_config['p_aug_shiftscalerotate'] = 0
        transform_config['p_aug_blur'] = 0
        transform_config['p_aug_brightness_contrast'] = 0
        transform_config['aug_shift_limit'] = 0
        transform_config['aug_scale_limit'] = 0
        transform_config['aug_rotate_limit'] = 0
        transform_config['aug_blur_limit'] = 1
        transform_config['aug_brightness_limit'] = 0
        transform_config['aug_contrast_limit'] = 0

    if orig_config['aug_level'] == 1:
        transform_config['p_aug_rotate90_flip_transpose'] = 0.5
        transform_config['p_aug_shiftscalerotate'] = 0.5
        transform_config['p_aug_blur'] = 0.5
        transform_config['p_aug_brightness_contrast'] = 0.5
        transform_config['aug_shift_limit'] = 0.15
        transform_config['aug_scale_limit'] = 0.15
        transform_config['aug_rotate_limit'] = 30
        transform_config['aug_blur_limit'] = int(np.ceil(9.0 * orig_config['dataset']['img_size'] / 384) // 2 * 2 + 1)
        transform_config['aug_brightness_limit'] = 0.2
        transform_config['aug_contrast_limit'] = 0.2

    if orig_config['aug_level'] == 2:
        transform_config['p_aug_rotate90_flip_transpose'] = 0.5
        transform_config['p_aug_shiftscalerotate'] = 0.5
        transform_config['p_aug_blur'] = 0.5
        transform_config['p_aug_brightness_contrast'] = 0.5
        transform_config['aug_shift_limit'] = 0.3
        transform_config['aug_scale_limit'] = 0.3
        transform_config['aug_rotate_limit'] = 45
        transform_config['aug_blur_limit'] = int(np.ceil(17.0 * orig_config['dataset']['img_size'] / 384) // 2 * 2 + 1)
        transform_config['aug_brightness_limit'] = 0.4
        transform_config['aug_contrast_limit'] = 0.4
    train_trsfm, test_trsfm = transform_gen_base(transform_config)
    return train_trsfm, test_trsfm


class MMCDataset(Dataset):
    def __init__(self, img_file_list, metric_data, gt_data, patient_id_data, occlusion_seg_path=None, occlusion_seg_index = None, trsfm=None):
        self.trsfm = trsfm
        self.img_file_list = img_file_list
        self.metric_data = metric_data
        self.gt_data = gt_data
        self.patient_id_data = patient_id_data
        self.occlusion_seg_path = occlusion_seg_path
        self.occlusion_seg_index = occlusion_seg_index

    def _image_helper(self, img_file_name):
        img = Image.open(img_file_name)
        img = np.array(img.convert(mode='RGB'))
        augmented = self.trsfm(image=img)
        img = augmented['image']
        img = img.transpose(2, 0, 1)
        return img
                
    def __getitem__(self, idx):
        left_img_file = self.img_file_list[0][idx]
        right_img_file = self.img_file_list[1][idx]
        if self.occlusion_seg_path is None or self.occlusion_seg_index is None:
            left_file_name = left_img_file
            right_file_name = right_img_file
        else:
            curr_type = occlusion_type[self.occlusion_seg_index]
            left_file_name = pathlib.join(self.occlusion_seg_path, curr_type,
                                          pathlib.basename(left_img_file)[:-4] + '.png')
            right_file_name = pathlib.join(self.occlusion_seg_path, curr_type,
                                           pathlib.basename(right_img_file)[:-4] + '.png')
        left_img = self._image_helper(left_file_name)
        right_img = self._image_helper(right_file_name)
        img_full = torch.stack([torch.Tensor(left_img), torch.Tensor(right_img)], dim=0)
        gt = float(self.gt_data[idx])
        patient_id = self.patient_id_data[idx]
        if self.metric_data is not None:
            metric_data = self.metric_data.iloc[idx].to_list()
            return [img_full, torch.Tensor(metric_data)], gt, patient_id
        else:
            return img_full, gt, patient_id

    def __len__(self):
        return len(self.img_file_list[0])


def pwv_denormalize(normalized_pwv_value):
    return float(normalized_pwv_value) * 349.12 + 1634.14


def read_csv(root_img_path, csv_name, feature_list, th, top_ratio):
    pwv_variable_index = 35
    csv_file = pd.read_csv(csv_name)
    if top_ratio is not None:
        num_rows = int(len(csv_file) * top_ratio)
        csv_file = csv_file.head(num_rows)

    if feature_list is not None:
        final_csv = csv_file[['patient_id', 'left_img_id', 'right_img_id'] +
                             ['param' + str(i) for i in feature_list + [pwv_variable_index]]]
    else:
        final_csv = csv_file[['patient_id', 'left_img_id', 'right_img_id', 'param' + str(pwv_variable_index)]]
        
    left_img_file = []
    right_img_file = []

    for left_img_id, right_img_id in zip(final_csv['left_img_id'], final_csv['right_img_id']):
        left_img_id_confirmed = None
        right_img_id_confirmed = None
        if type(left_img_id) == str:
            left_img_id_confirmed = pathlib.join(root_img_path, left_img_id)
        if type(right_img_id) == str:
            right_img_id_confirmed = pathlib.join(root_img_path, right_img_id)
        if left_img_id_confirmed is None:
            left_img_id_confirmed = right_img_id_confirmed
        if right_img_id_confirmed is None:
            right_img_id_confirmed = left_img_id_confirmed

        left_img_file.append(left_img_id_confirmed)
        right_img_file.append(right_img_id_confirmed)

    gt_data = final_csv['param' + str(pwv_variable_index)].to_list()
    if th is not None:
        gt_data = [float(pwv_denormalize(normalized_pwv_value) >= th) for normalized_pwv_value in gt_data]
    if feature_list is not None:
        metric_data = final_csv[['param' + str(i) for i in feature_list]]
        for curr_col in metric_data.columns:
            metric_data[curr_col] = pd.to_numeric(metric_data[curr_col], errors='raise')
    else:
        metric_data = None

    patient_id_data = final_csv['patient_id'].to_list()
    return (left_img_file, right_img_file), metric_data, gt_data, patient_id_data


def get_data_loader(config, logger):
    train_trsfm, test_trsfm = transform_gen(config)
    cfg_data = config['dataset']
        
    train_img_file, train_metric_data, train_gt_data, train_patient_id_data = read_csv(
        cfg_data['root_path'], cfg_data['train_csv_name'], cfg_data['feature_list'], cfg_data['th'], config['train_ratio'])
    val_img_file, val_metric_data, val_gt_data, val_patient_id_data = read_csv(
        cfg_data['root_path'], cfg_data['val_csv_name'], cfg_data['feature_list'], cfg_data['th'],  None)
    test_img_file, test_metric_data, test_gt_data, test_patient_id_data = read_csv(
        cfg_data['root_path'], cfg_data['test_csv_name'], cfg_data['feature_list'], cfg_data['th'],  None)

    if 'occlusion_seg_path' in cfg_data.keys() and 'occlusion_seg_index' in config.keys():
        occlusion_seg_path=cfg_data['occlusion_seg_path']
        occlusion_seg_index=config['occlusion_seg_index']
    else:
        logging_info(logger, 'occlusion_seg_path or occlusion_seg_index not in config keys')
        occlusion_seg_path=None
        occlusion_seg_index=None
        
    train_dataset = MMCDataset(train_img_file, train_metric_data, train_gt_data, train_patient_id_data, trsfm=train_trsfm, occlusion_seg_path=occlusion_seg_path, occlusion_seg_index=occlusion_seg_index)
    val_dataset = MMCDataset(val_img_file, val_metric_data, val_gt_data, val_patient_id_data, trsfm=test_trsfm, occlusion_seg_path=occlusion_seg_path, occlusion_seg_index=occlusion_seg_index)
    test_dataset = MMCDataset(test_img_file, test_metric_data, test_gt_data, test_patient_id_data, trsfm=test_trsfm, occlusion_seg_path=occlusion_seg_path, occlusion_seg_index=occlusion_seg_index)
                      
    train_gt_data_counter = Counter(train_gt_data)
    weights_sum = np.sum(list(train_gt_data_counter.values()))
    balanced_weights = np.array([weights_sum / train_gt_data_counter[i] for i in train_gt_data])
    balanced_weights = balanced_weights / np.sum(balanced_weights)
    uniform_weights = np.ones(len(balanced_weights)) * 1.0 / len(balanced_weights)
           
    if cfg_data['th'] is not None and config['weighted_class_sampling_type'] == 0:
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True, drop_last=True)
    elif cfg_data['th'] is not None and config['weighted_class_sampling_type'] == 1:
        weights = 0.5 * balanced_weights + 0.5 * uniform_weights    
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=sampler, drop_last=True)
    elif cfg_data['th'] is not None and config['weighted_class_sampling_type'] == 2:
        weights = balanced_weights
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=sampler, drop_last=True)
    else:
        train_loader = None

    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'],
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'],
                             shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
