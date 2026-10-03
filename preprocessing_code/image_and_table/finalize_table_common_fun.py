import numpy as np
import pandas as pd
import os.path as pathlib
from copy import deepcopy

def table_one_hot_encoding(new_table):
    # param 1 [1,2]
    # param 2, 3, 4, 5, 6, 7 => numerical
    # param 8, 9, 10, 11, 12 [0, 1] 
    new_table.loc[:, 'param1'] = (new_table['param1'] - 1).astype(np.int64)
    new_table.loc[:, 'param8'] = new_table['param8'].astype(np.int64)
    new_table.loc[:, 'param9'] = new_table['param9'].astype(np.int64)
    new_table.loc[:, 'param10'] = new_table['param10'].astype(np.int64)
    new_table.loc[:, 'param11'] = new_table['param11'].astype(np.int64)
    new_table.loc[:, 'param12'] = new_table['param12'].astype(np.int64)
    return new_table

def merge_imputation_data(orig_table, imputation_data_dir, imputation_table_idx):
    imputation_table = pd.read_csv(pathlib.join(imputation_data_dir, 'mice_imputation_full_' + str(imputation_table_idx) + '.csv'))
    param_idx_array = list((np.arange(12) + 1).astype(np.int64))
    new_table = deepcopy(orig_table)
    for param_idx in param_idx_array:
        def rowwise_helper_fun(x):
            is_valid = (type(x['param' + str(param_idx)]) == int or type(x['param' + str(param_idx)]) == float) and (not np.isnan(x['param' + str(param_idx)]))
            if not is_valid:
                x['param' + str(param_idx)] = imputation_table[imputation_table['patient_id'] == x['patient_id']]['param' + str(param_idx)].values[0]
            return x
        new_table = new_table.apply(rowwise_helper_fun, axis = 1)
    new_table = table_one_hot_encoding(new_table)
    return new_table


def convert_img_to_patient_csv(per_img_data_csv, per_patient_data_csv, feats=[1,2,3,4,5,6,7,8,9,10,11,12], target = 35):
    data_csv = pd.read_csv(per_img_data_csv)
    img_id_list = list(data_csv['img_id'])
    patient_id_list = list(data_csv['patient_id'])
    eye_type_list = list(data_csv['left_right'])
    quality_type = list(data_csv['IQ'])
    
    patient_data_col_list = list(['province']) + ['param' + str(idx) for idx in feats + [target]]
    data_list = list()
    for col_name in patient_data_col_list:
        data_list.append(list(data_csv[col_name]))
    data_list = np.array(data_list)
    data_list = data_list.transpose()

    patient_data_dict = dict()
    for patient_id, img_id, eye_type_data, quality_type_data, patient_data in zip(patient_id_list, img_id_list, eye_type_list, quality_type, data_list):
        if patient_id not in patient_data_dict.keys():
            patient_data_dict[patient_id] = list()
            patient_data_dict[patient_id].append('') # left_img_id
            patient_data_dict[patient_id].append('') # right_img_id
            patient_data_dict[patient_id].append('') # left_img_quality
            patient_data_dict[patient_id].append('') # right_img_quality
            patient_data_dict[patient_id].extend(patient_data)
        if eye_type_data == 'left':
            patient_data_dict[patient_id][0] = img_id
            patient_data_dict[patient_id][2] = quality_type_data
        if eye_type_data == 'right':
            patient_data_dict[patient_id][1] = img_id
            patient_data_dict[patient_id][3] = quality_type_data
        assert np.sum(np.array(patient_data_dict[patient_id][4:]) != np.array(patient_data))==0, str(patient_data_dict[patient_id][6:]) + ' not equal to ' + str(patient_data)
    
    column_name = ['patient_id', 'left_img_id', 'right_img_id', 'left_img_quality', 'right_img_quality'] + patient_data_col_list
    patient_data_list = list()
    for key, value in patient_data_dict.items():
        patient_data_list.append([key] + list(value))
    patient_data_frame = pd.DataFrame(patient_data_list, columns=column_name)
    patient_data_frame.to_csv(per_patient_data_csv, index=False)