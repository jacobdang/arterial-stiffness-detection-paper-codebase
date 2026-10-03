import logging
import os.path as pathlib
import sys
import os
from os import makedirs
from copy import deepcopy
import yaml
from deco import concurrent, synchronized

def get_logger(logger_name, logging_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(logging_file_path, mode='a')
    fmt = '[%(asctime)s]: %(message)s'
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger

@concurrent
def main_helper_metric_and_fusion(orig_dir, orig_train_csv_name, orig_val_csv_name, orig_test_csv_name, mice_imputation_index, config, logger_name, train_and_test_model_main_fun):
    if mice_imputation_index is not None:
        config['output_dir'] = pathlib.join(deepcopy(orig_dir), 'mice_imputation_' + str(mice_imputation_index))
        config['train_csv_name'] = deepcopy(orig_train_csv_name).replace('$X$', str(mice_imputation_index))
        config['val_csv_name'] = deepcopy(orig_val_csv_name).replace('$X$', str(mice_imputation_index))
        config['test_csv_name'] = deepcopy(orig_test_csv_name).replace('$X$', str(mice_imputation_index))
    if 'fusion' not in config.keys():
        config['fusion'] = True
    
    output_dir_origin = config['output_dir']
    feat_set_list = config['feat_set_list']

    config['output_dir'] = pathlib.join(output_dir_origin, 'metric_only')
    makedirs(config['output_dir'], exist_ok=True)
    logger = get_logger(logger_name + '_metric_only', pathlib.join(config['output_dir'], 'train_and_test_main.log'))
    for feat_set in feat_set_list:
        train_and_test_model_main_fun(logger, config, feat_set, with_fusion_img_score=False,
                                      target=config['target'], threshold=config['threshold'])
  
    if config['fusion']:
        config['output_dir'] = pathlib.join(output_dir_origin, 'fusion')
        makedirs(config['output_dir'], exist_ok=True)
        logger = get_logger(logger_name + '_fusion', pathlib.join(config['output_dir'], 'train_and_test_main.log'))
        for feat_set in feat_set_list:
            train_and_test_model_main_fun(logger, config, feat_set, with_fusion_img_score=True,
                                          target=config['target'], threshold=config['threshold'])

@synchronized        
def main_helper_mice_imputation(config, method_name, train_and_test_model_main_fun):
    orig_dir = config['output_dir']
    orig_train_csv_name = config['train_csv_name']
    orig_val_csv_name = config['val_csv_name']
    orig_test_csv_name = config['test_csv_name']    
    if not config['use_mice_imputation']:
        main_helper_metric_and_fusion(orig_dir, orig_train_csv_name, orig_val_csv_name, orig_test_csv_name, None, config, 'main_' + method_name, train_and_test_model_main_fun)
    else:
        for mice_imputation_index in config['mice_imputation_index_list']:
            main_helper_metric_and_fusion(orig_dir, orig_train_csv_name, orig_val_csv_name, orig_test_csv_name, mice_imputation_index, config, 'main_' + str(mice_imputation_index) + '_' + method_name, train_and_test_model_main_fun)

def procedure_main(config_name, method, train_and_test_model_main_fun):
    with open(config_name, 'r') as config_stream:
        config = yaml.safe_load(config_stream)
    config['method'] = method
    config['output_dir'] = pathlib.join(config['output_dir'], 'method_' + str(method))
    main_helper_mice_imputation(config, method, train_and_test_model_main_fun)