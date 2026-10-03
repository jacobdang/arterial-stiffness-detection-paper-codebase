import cv2
cv2.setNumThreads(1)
import sys
import numpy as np
import os.path as pathlib
from glob import glob

from dataloader import transform_gen, read_csv, MMCDataset
from torch.utils.data import DataLoader
from procedure import default_test_loop
from procedure import restore_checkpoint
from utils import get_logger, logging_info
from sklearn.metrics import roc_auc_score
from procedure import get_network, eval_main_numpy
from torch_ema import ExponentialMovingAverage
import pickle


def get_train_config(train_dir):
    log_file_path = pathlib.join(train_dir, 'train_main.log')
    with open(log_file_path, 'r') as log_file_obj:
        config_line = log_file_obj.readlines()[0]
        config_dict = eval((config_line[config_line.find('INFO')+8:-1]))
    return config_dict


def get_top_checkpoint_list(train_dir, no_checkpoints=5):
    checkpoint_dir = pathlib.join(train_dir, 'checkpoint')
    checkpoint_path_list = glob(pathlib.join(checkpoint_dir, 'epoch*.pth'))
    checkpoint_path_list.sort()

    score_set = []
    for checkpoint_path in checkpoint_path_list:
        filename = pathlib.basename(checkpoint_path)
        auc_score = float(filename[filename.find('auc') + 4:-4])
        score_set.append(auc_score)
    top_ranked_index = np.argsort(score_set)[::-1][0:no_checkpoints]
    top_ranked_checkpoint = np.array(checkpoint_path_list)[top_ranked_index]
    return top_ranked_checkpoint


def get_dataloader_helper(config, dataloader_type, logger):
    train_trsfm, test_trsfm = transform_gen(config)
    cfg_data = config['dataset']
    if dataloader_type == 'train':
        cfg_name = cfg_data['train_csv_name']
    elif dataloader_type == 'val':
        cfg_name = cfg_data['val_csv_name']
    elif dataloader_type == 'test':
        cfg_name = cfg_data['test_csv_name']
    else:
        cfg_name = None

    img_file, metric_data, gt_data, patient_id_data = read_csv(
        cfg_data['root_path'], cfg_name, cfg_data['feature_list'], cfg_data['th'], None)     
        
    if 'occlusion_seg_path' in cfg_data.keys() and 'occlusion_seg_index' in config.keys():
        occlusion_seg_path=cfg_data['occlusion_seg_path']
        occlusion_seg_index=config['occlusion_seg_index']
    else:
        logging_info(logger, 'occlusion_seg_path or occlusion_seg_index not in config keys')
        occlusion_seg_path=None
        occlusion_seg_index=None     
           
    dataset = MMCDataset(img_file, metric_data, gt_data, patient_id_data, trsfm=test_trsfm, occlusion_seg_path=occlusion_seg_path, occlusion_seg_index=occlusion_seg_index)
    num_workers=int(config.get("num_workers", 16))
    loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=num_workers, shuffle=False, drop_last=False)
    return loader


def test_helper(config, test_checkpoint_path, logger):
    def test_helper_subfun(config, dataloader_type, network, network_type, ema):
        dataloader = get_dataloader_helper(config, dataloader_type, logger)
        gt_set, pred_set, id_set, _, _, feat_set = default_test_loop(dataloader, network, network_type, ema=ema)
        auc = roc_auc_score(gt_set, pred_set)
        return gt_set, pred_set, id_set, auc, feat_set

    network, network_type = get_network(config['model'], config['dataset'])
    network = network.cuda()
    if config['ema_decay'] is not None:
        ema = ExponentialMovingAverage(network.parameters(), decay=config['ema_decay'])
    else:
        ema = None
    restore_checkpoint(test_checkpoint_path, network, None, None, None, ema, logger, True)

    train_result = test_helper_subfun(config, 'train', network, network_type, ema)
    val_result = test_helper_subfun(config, 'val', network, network_type, ema)
    test_result = test_helper_subfun(config, 'test', network, network_type, ema)
    return train_result, val_result, test_result


def save_helper(config, logger, dataloader_type, auc, gt_set, pred_set, id_set, feat_set, save_name):
    logging_info(logger, dataloader_type + ', patient_level auc %f' % auc)
    dump_dict = {'config': config, 'gt_set': gt_set, 'pred_set': pred_set, 'id_set': id_set,
                 'feat_set': feat_set}
    with open(save_name, 'wb') as handle:
        pickle.dump(dump_dict, handle)


def test_main_routine(train_dir):
    config = get_train_config(train_dir)
    logging_path = pathlib.join(train_dir, 'test_main.log')
    logger = get_logger(train_dir, logging_path)

    top_ranked_checkpoint_list = get_top_checkpoint_list(train_dir)
    logging_info(logger, 'top_ranked_checkpoint_list:' + str(top_ranked_checkpoint_list))
    best_checkpoint = top_ranked_checkpoint_list[0]
    logging_info(logger, 'best_checkpoint:' + str(best_checkpoint))
    train_result0, val_result0, test_result0 = test_helper(config, best_checkpoint, logger)
    save_helper(config, logger, 'train', train_result0[3], train_result0[0], train_result0[1],
                train_result0[2], train_result0[4],
                pathlib.join(train_dir, 'best_checkpoint_train_result.pickle'))
    save_helper(config, logger, 'val', val_result0[3], val_result0[0], val_result0[1],
                val_result0[2], val_result0[4],
                pathlib.join(train_dir, 'best_checkpoint_val_result.pickle'))
    save_helper(config, logger, 'test', test_result0[3], test_result0[0], test_result0[1],
                test_result0[2], test_result0[4],
                pathlib.join(train_dir, 'best_checkpoint_test_result.pickle'))

    logging_info(logger, 'top ranked 5 best checkpoint ensemble results.')
    train_pred_set_all = list()
    train_feat_set_all = list()
    val_pred_set_all = list()
    val_feat_set_all = list()
    test_pred_set_all = list()
    test_feat_set_all = list()
    for curr_checkpoint in top_ranked_checkpoint_list:
        logging_info(logger, 'curr tested checkpoint: ' + str(curr_checkpoint))
        train_result1, val_result1, test_result1 = test_helper(config, curr_checkpoint, logger)
        assert((train_result0[0] == train_result1[0]).all())
        assert((train_result0[2] == train_result1[2]).all())
        assert((val_result0[0] == val_result1[0]).all())
        assert((val_result0[2] == val_result1[2]).all())
        assert((test_result0[0] == test_result1[0]).all())
        assert((test_result0[2] == test_result1[2]).all())
        train_pred_set_all.append(train_result1[1])
        train_feat_set_all.append(train_result1[4])
        val_pred_set_all.append(val_result1[1])
        val_feat_set_all.append(val_result1[4])
        test_pred_set_all.append(test_result1[1])
        test_feat_set_all.append(test_result1[4])
    train_pred_set_all = np.mean(np.array(train_pred_set_all), axis=0)
    val_pred_set_all = np.mean(np.array(val_pred_set_all), axis=0)
    test_pred_set_all = np.mean(np.array(test_pred_set_all), axis=0)
    train_feat_set_all = np.mean(np.array(train_feat_set_all), axis=0)
    val_feat_set_all = np.mean(np.array(val_feat_set_all), axis=0)
    test_feat_set_all = np.mean(np.array(test_feat_set_all), axis=0)
    _, train_auc_all = eval_main_numpy(train_pred_set_all, train_result0[0])
    _, val_auc_all = eval_main_numpy(val_pred_set_all, val_result0[0])
    _, test_auc_all = eval_main_numpy(test_pred_set_all, test_result0[0])

    save_helper(config, logger, 'train', train_auc_all, train_result0[0], train_pred_set_all,
                train_result0[2], train_feat_set_all,
                pathlib.join(train_dir, 'top_checkpoint_ensemble_train_result.pickle'))
    save_helper(config, logger, 'val', val_auc_all, val_result0[0], val_pred_set_all,
                val_result0[2], val_feat_set_all,
                pathlib.join(train_dir, 'top_checkpoint_ensemble_val_result.pickle'))
    save_helper(config, logger, 'test', test_auc_all, test_result0[0], test_pred_set_all,
                test_result0[2], test_feat_set_all,
                pathlib.join(train_dir, 'top_checkpoint_ensemble_test_result.pickle'))


if __name__ == '__main__':
    train_dir = sys.argv[1]
    is_recursive_sub_dir = sys.argv[2]
    if bool(int(is_recursive_sub_dir)):
        sub_dir_list = glob(pathlib.join(train_dir, '*'))
        for sub_dir in sub_dir_list:
            train_logging_file = pathlib.join(sub_dir, 'train_main.log')
            test_output_file = pathlib.join(sub_dir, 'top_checkpoint_ensemble_test_result.pickle')
            if pathlib.exists(train_logging_file) and (not pathlib.exists(test_output_file)):
                test_main_routine(sub_dir)
    else:
        test_main_routine(train_dir)
