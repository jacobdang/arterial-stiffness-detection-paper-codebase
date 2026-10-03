from data_loading_common_funs import get_trainval_test_csv
from procedure_common_funs import procedure_main
from sklearn.metrics import roc_auc_score as roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import pickle
import numpy as np
import pandas as pd
import os.path as pathlib
from copy import deepcopy
import sys
import itertools

ml_method_dict = dict()

lr_def_fun = LogisticRegression
ml_method_dict['lr'] = {'C':[1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]}

lda_def_fun = LinearDiscriminantAnalysis
ml_method_dict['lda'] = {'shrinkage':[1e-3, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 'auto'], 'solver':['eigen']}

lsvm_def_fun = SVC 
ml_method_dict['lsvm'] = {'C':[5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'kernel':['linear'], 'probability':[True]}

rbfsvm_def_fun = SVC
ml_method_dict['rbfsvm'] = {'C':[5e-4, 1e-3, 5e-3, 0.01, 0.1, 1, 5, 10], 'gamma':['auto', 'scale'], 'kernel':['rbf'], 'probability':[True]}

rf_def_fun = RandomForestClassifier
ml_method_dict['rf'] = {'criterion':['gini', 'entropy'], 'max_features':['sqrt', 'log2', 0.1, 0.2, 0.3, 0.4, 0.5], 'n_estimators':[1000], 'random_state':[0]}

gbc_def_fun = GradientBoostingClassifier
ml_method_dict['gbc'] = {'n_estimators': [20, 100, 500], 'max_depth': [1, 2, 3, 5, 7, 9, 11, 15, 19], 'learning_rate':[1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5], 'random_state':[0]}


def train_and_test_model_main_helper(logger, config, feat_set, with_fusion_img_score, cls_def_fun, cls_param_dict, target=35, threshold=1400):
    logger.info(str(dict(config)))
    logger.info(str(feat_set))
    logger.info('prepare training and validation data...')
    
    train_csv_cls, val_csv_cls, test_csv_cls, _ = get_trainval_test_csv(config['train_csv_name'], config['val_csv_name'], config['test_csv_name'], feat_set, target, threshold, with_fusion_img_score, config['train_img_only_pickle_file'], config['val_img_only_pickle_file'], config['test_img_only_pickle_file'])
    
    if with_fusion_img_score:
        feats_final = ['param' + str(i) for i in feat_set] + ['img_score']
    else:
        feats_final = ['param' + str(i) for i in feat_set]

    val_acc_all = list()
    param_config_all = list(itertools.product(*list(cls_param_dict.values())))
    param_name_all = list(cls_param_dict.keys())
    model_all = list()
    
    for param_config in param_config_all:
        print('param names are: ' + str(param_name_all) + ' , and curr_config is: ' + str(param_config))
        X = train_csv_cls[feats_final]
        y = train_csv_cls['param' + str(target)]
        param_dict = {key:value for key,value in zip(param_name_all, param_config)}
        clf = cls_def_fun(**param_dict)
        clf = clf.fit(X, y)
        X_val = val_csv_cls[feats_final]
        y_val = val_csv_cls['param' + str(target)]
        val_acc_all.append(roc_auc_score(y_val, clf.predict_proba(X_val)[:,1]))
        model_all.append(clf)
    
    best_idx = np.argmax(val_acc_all)
    best_model = model_all[best_idx]

    X_train = train_csv_cls[feats_final]
    y_train = train_csv_cls['param' + str(target)]
    y_train_pred = best_model.predict_proba(X_train)[:,1]
    
    X_val = val_csv_cls[feats_final]
    y_val = val_csv_cls['param' + str(target)]
    y_val_pred = best_model.predict_proba(X_val)[:,1]
    
    X_test = test_csv_cls[feats_final]
    y_test = test_csv_cls['param' + str(target)]
    y_test_pred = best_model.predict_proba(X_test)[:,1]

    train_pred_result = pd.DataFrame(np.array((train_csv_cls['patient_id'], y_train, y_train_pred)).transpose(), columns=['patient_id', 'gt', 'pred'])
    val_pred_result = pd.DataFrame(np.array((val_csv_cls['patient_id'], y_val, y_val_pred)).transpose(), columns=['patient_id', 'gt', 'pred'])
    test_pred_result = pd.DataFrame(np.array((test_csv_cls['patient_id'], y_test, y_test_pred)).transpose(), columns=['patient_id', 'gt', 'pred'])

    val_auc_score = roc_auc_score(val_pred_result['gt'], val_pred_result['pred'])
    logger.info('validation auc score is: ' + str(val_auc_score))
    test_auc_score = roc_auc_score(test_pred_result['gt'], test_pred_result['pred'])
    logger.info('test auc score is: ' + str(test_auc_score))
    
    train_pred_result.to_csv(pathlib.join(config['output_dir'], 'feat_' + str(feat_set) + '_train_pred_result.csv'), index=False)
    val_pred_result.to_csv(pathlib.join(config['output_dir'], 'feat_' + str(feat_set) + '_val_pred_result.csv'), index=False)
    test_pred_result.to_csv(pathlib.join(config['output_dir'], 'feat_' + str(feat_set) + '_test_pred_result.csv'), index=False)
    pickle.dump(best_model, open(pathlib.join(config['output_dir'], 'feat_' + str(feat_set) + '_trained_all_submodel.pickle'), 'wb'))



    
def train_and_test_model_main_lr(logger, config, feat_set, with_fusion_img_score, target=35, threshold=1400):
    train_and_test_model_main_helper(logger, config, feat_set, with_fusion_img_score, lr_def_fun, ml_method_dict['lr'], target, threshold)
    
def train_and_test_model_main_lda(logger, config, feat_set, with_fusion_img_score, target=35, threshold=1400):
    train_and_test_model_main_helper(logger, config, feat_set, with_fusion_img_score, lda_def_fun, ml_method_dict['lda'], target, threshold)
    
def train_and_test_model_main_lsvm(logger, config, feat_set, with_fusion_img_score, target=35, threshold=1400):
    train_and_test_model_main_helper(logger, config, feat_set, with_fusion_img_score, lsvm_def_fun, ml_method_dict['lsvm'], target, threshold)

def train_and_test_model_main_rbfsvm(logger, config, feat_set, with_fusion_img_score, target=35, threshold=1400):
    train_and_test_model_main_helper(logger, config, feat_set, with_fusion_img_score, rbfsvm_def_fun, ml_method_dict['rbfsvm'], target, threshold)

def train_and_test_model_main_rf(logger, config, feat_set, with_fusion_img_score, target=35, threshold=1400):
    train_and_test_model_main_helper(logger, config, feat_set, with_fusion_img_score, rf_def_fun, ml_method_dict['rf'], target, threshold)

def train_and_test_model_main_gbc(logger, config, feat_set, with_fusion_img_score, target=35, threshold=1400):
    train_and_test_model_main_helper(logger, config, feat_set, with_fusion_img_score, gbc_def_fun, ml_method_dict['gbc'], target, threshold)
    
if __name__ == '__main__':
    config_name = sys.argv[1]
    procedure_main(config_name, 'lr_basic', train_and_test_model_main_lr)
    #procedure_main(config_name, 'lda_basic', train_and_test_model_main_lda)
    #procedure_main(config_name, 'lsvm_basic', train_and_test_model_main_lsvm)
    #procedure_main(config_name, 'rbfsvm_basic', train_and_test_model_main_rbfsvm)
    #procedure_main(config_name, 'rf_basic', train_and_test_model_main_rf)
    #procedure_main(config_name, 'gbc_basic', train_and_test_model_main_gbc)
    

            
      


            























