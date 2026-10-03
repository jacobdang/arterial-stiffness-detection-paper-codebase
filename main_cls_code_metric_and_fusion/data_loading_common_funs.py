import numpy as np
import pandas as pd
import os
import random
import math
from copy import deepcopy
import pickle


def basic_csv_read_cls(input_csv, feats, target, threshold):
    feats = ['param' + str(i) for i in feats]
    target = ['param' + str(target)]
    input_csv = input_csv[['patient_id'] + feats + target]
    for feat in feats:
        input_csv[feat] = input_csv[feat].astype(float)
    input_csv[target] = (input_csv[target] * 349.12) + 1634.14 >= threshold
    return input_csv

def merge_img_score_helper(data_frame, patient_id_list, pred_set_list):
    new_data =  np.array((patient_id_list, pred_set_list)).transpose()
    new_frame = pd.DataFrame(new_data, columns=('patient_id', 'img_score'))
    data_frame = data_frame.merge(new_frame, on='patient_id')
    return data_frame


def get_trainval_test_csv(train_csv_name, val_csv_name, test_csv_name, feats, target, threshold,  with_fusion_img_score, train_img_only_pickle_file, val_img_only_pickle_file, test_img_only_pickle_file):
    train_csv_0 = pd.read_csv(train_csv_name)
    val_csv_0 = pd.read_csv(val_csv_name)
    test_csv_0 = pd.read_csv(test_csv_name)

    train_csv_cls = basic_csv_read_cls(deepcopy(train_csv_0), feats, target, threshold).drop_duplicates()
    val_csv_cls = basic_csv_read_cls(deepcopy(val_csv_0), feats, target, threshold).drop_duplicates()
    test_csv_cls = basic_csv_read_cls(deepcopy(test_csv_0), feats, target, threshold).drop_duplicates()
    trainval_csv_cls = pd.concat([train_csv_cls, val_csv_cls], axis=0).reset_index(drop=True) 
    
    if with_fusion_img_score:
        train_image_only_record = pickle.load(open(train_img_only_pickle_file, 'rb'))
        val_image_only_record = pickle.load(open(val_img_only_pickle_file, 'rb'))
        test_image_only_record = pickle.load(open(test_img_only_pickle_file, 'rb'))
        train_csv_cls = merge_img_score_helper(train_csv_cls, list(train_image_only_record['id_set']), list(train_image_only_record['pred_set']))
        val_csv_cls = merge_img_score_helper(val_csv_cls, list(val_image_only_record['id_set']), list(val_image_only_record['pred_set']))
        test_csv_cls = merge_img_score_helper(test_csv_cls, list(test_image_only_record['id_set']), list(test_image_only_record['pred_set']))
        trainval_csv_cls = merge_img_score_helper(trainval_csv_cls, list(train_image_only_record['id_set']) + list(val_image_only_record['id_set']), list(train_image_only_record['pred_set']) + list(val_image_only_record['pred_set']))
    return train_csv_cls, val_csv_cls, test_csv_cls, trainval_csv_cls



