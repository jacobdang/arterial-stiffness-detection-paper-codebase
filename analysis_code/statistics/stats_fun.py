import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import sklearn.metrics as sklm
import os.path as pathlib
import sklearn.metrics as metrics
import pandas as pd

def get_score_and_gt_helper(base_path, method_id, feature_id, fusion_type='metric_only'):
    input_dir = pathlib.join(base_path, method_id)
    train_pred_list = list()
    val_pred_list = list()
    test_pred_list = list()
    train_gt_list = list()
    val_gt_list = list()
    test_gt_list = list()
    for mice_id in range(20):
        curr_folder = pathlib.join(input_dir, 'mice_imputation_' + str(mice_id+ 1), fusion_type)
        train_score = pd.read_csv(pathlib.join(curr_folder, "feat_[" + str(feature_id) + "]_train_pred_result.csv"))
        val_score = pd.read_csv(pathlib.join(curr_folder, "feat_[" + str(feature_id) + "]_val_pred_result.csv"))
        test_score = pd.read_csv(pathlib.join(curr_folder, "feat_[" + str(feature_id) + "]_test_pred_result.csv"))
        train_pred_list.append(np.array(train_score['pred']))
        val_pred_list.append(np.array(val_score['pred']))
        test_pred_list.append(np.array(test_score['pred']))
        train_gt_list.append(np.array(train_score['gt']))
        val_gt_list.append(np.array(val_score['gt']))
        test_gt_list.append(np.array(test_score['gt']))   
        
    assert((val_gt_list[0] == val_gt_list[1]).all())
    assert((test_gt_list[0] == test_gt_list[1]).all())
    val_gt_list = val_gt_list[0]
    test_gt_list = test_gt_list[0]
    val_pred_list = np.mean(np.array(val_pred_list), axis = 0)
    test_pred_list = np.mean(np.array(test_pred_list), axis = 0)
    return train_gt_list, val_gt_list, test_gt_list, train_pred_list, val_pred_list, test_pred_list

def get_score_and_gt(base_path, feature_id, fusion_type='metric_only',  method_id_list = ['method_lr_basic', 'method_lda_basic','method_gbc_basic', 'method_rf_basic',  'method_lsvm_basic',  'method_rbfsvm_basic']):
    val_auc_list = []
    for method_id in method_id_list:
        _, val_gt_list, _, _, val_pred_list, _ = get_score_and_gt_helper(base_path, method_id, feature_id, fusion_type)
        fpr, tpr, threshold = metrics.roc_curve(val_gt_list, val_pred_list)
        auc = metrics.auc(fpr, tpr)
        val_auc_list.append(auc)
    
    best_val_idx = np.argmax(val_auc_list)
    best_method_id = np.array(method_id_list)[best_val_idx]
    train_gt_list, val_gt_list, test_gt_list, train_pred_list, val_pred_list, test_pred_list = get_score_and_gt_helper(base_path, best_method_id, feature_id, fusion_type)
    return best_method_id, train_gt_list, val_gt_list, test_gt_list, train_pred_list, val_pred_list, test_pred_list

def get_statistic_info(test_gt_list, test_pred_list):
    auc95, rec, sep, sep_90, global_auc, global_rec, global_sep, global_sep_90, _, _ = cal_ci95(test_gt_list, test_pred_list, 0.8)
    fpr, tpr, threshold = metrics.roc_curve(test_gt_list, test_pred_list)
    auc = metrics.auc(fpr, tpr)
    return auc95, fpr, tpr, auc

def cal_sep(test_labs_at_dim, rounded_preds_at_dim):
    cm = sklm.confusion_matrix(test_labs_at_dim, rounded_preds_at_dim)
    if len(cm)==2:
        specificity =cm[0,0]/(cm[0,0]+cm[0,1])
    else:
        specificity = 0
    return specificity

def sep_and_th_for_a_given_recall(y_test, y_pred_val, recall_thre):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_val)
    recall = tpr
    sep = 1 - fpr
    best_thre_index = np.argmin(np.abs(recall - recall_thre))
    best_thre = thresholds[best_thre_index]
    best_sep = sep[best_thre_index]
    result = ((y_pred_val <best_thre) & (y_test==0))
    return recall[best_thre_index], best_sep, best_thre

def recall_and_sep_and_th_largest_youden(y_test, y_pred_val):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_val)
    sep = 1 - fpr
    best_thre_index = np.argmax(tpr + sep - 1)
    best_thre = thresholds[best_thre_index]
    best_sep = sep[best_thre_index]
    best_recall = tpr[best_thre_index]
    return best_recall, best_sep, best_thre

def recall_and_sep(y_test, y_pred_val):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_val)
    sep = 1 - fpr
    best_thre_index = np.argmax(tpr + sep - 1)
    best_thre = thresholds[best_thre_index]
    best_sep = sep[best_thre_index]
    best_recall = tpr[best_thre_index]
    return best_recall, best_sep, best_thre
                            
def cal_ci95(y_true,y_pred,th):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    global_auc = roc_auc_score(y_true, y_pred)
    global_largest_youden_rec, global_largest_youden_sep, global_largest_youden_th = recall_and_sep_and_th_largest_youden(y_true, y_pred)
    global_given_recall_rec, global_given_recall_sep, global_given_recall_th = sep_and_th_for_a_given_recall(y_true, y_pred, th)
    
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_auc = []
    bootstrapped_largest_youden_recall = []
    bootstrapped_largest_youden_sep = []
    bootstrapped_given_recall_sep = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score_auc = roc_auc_score(y_true[indices], y_pred[indices])
        recall = sklm.recall_score(y_true[indices], y_pred[indices]>=global_largest_youden_th, average=None)
        recall = recall[1]
        sep = cal_sep(y_true[indices], y_pred[indices]>=global_largest_youden_th)
        sep_th = cal_sep(y_true[indices], y_pred[indices]>=global_given_recall_th)
        bootstrapped_auc.append(score_auc)
        bootstrapped_largest_youden_recall.append(recall)
        bootstrapped_largest_youden_sep.append(sep)
        bootstrapped_given_recall_sep.append(sep_th)

    auc=[]
    sorted_scores = np.array(bootstrapped_auc)
    sorted_scores.sort()
    auc.append(sorted_scores[int(0.025 * len(sorted_scores))])
    auc.append(sorted_scores[int(0.975 * len(sorted_scores))])
    
    rec=[]
    sorted_scores = np.array(bootstrapped_largest_youden_recall)
    sorted_scores.sort()
    rec.append(sorted_scores[int(0.025 * len(sorted_scores))])
    rec.append(sorted_scores[int(0.975 * len(sorted_scores))])
    
    sep = []
    sorted_scores = np.array(bootstrapped_largest_youden_sep)
    sorted_scores.sort()
    sep.append(sorted_scores[int(0.025 * len(sorted_scores))])
    sep.append(sorted_scores[int(0.975 * len(sorted_scores))])
    
    sep_th = []
    sorted_scores = np.array(bootstrapped_given_recall_sep)
    sorted_scores.sort()
    sep_th.append(sorted_scores[int(0.025 * len(sorted_scores))])
    sep_th.append(sorted_scores[int(0.975 * len(sorted_scores))])
    
    return auc, rec, sep, sep_th, global_auc, global_largest_youden_rec, global_largest_youden_sep, global_given_recall_sep, global_largest_youden_th, global_given_recall_th


