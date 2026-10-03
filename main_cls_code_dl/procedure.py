import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from utils import logging_info
from network import GenericBiStreamImgOnly
from torch.optim import AdamW, SGD
from sklearn.metrics import roc_auc_score, accuracy_score
import os
from torchmetrics.functional import accuracy, auroc
import os.path as pathlib


def eval_main_tensor(pred_prob_tensor, gt_tensor):
    acc = accuracy(pred_prob_tensor, gt_tensor.long())
    auc = auroc(pred_prob_tensor, gt_tensor.long(), pos_label=1)
    return acc, auc


def eval_main_numpy(pred_prob, gt_prob):
    acc = accuracy_score(gt_prob, pred_prob >= 0.5)
    auc = roc_auc_score(gt_prob, pred_prob)
    return acc, auc


def get_network(cfg_network, cfg_dataset):
    if cfg_network['network_arch'] == 'GenericBiStreamAttFusion':
        network = GenericBiStreamAttFusion(cfg_network['backbone_name'], cfg_network['approximate_head_dim'], list(cfg_dataset['meta_dims']))
        network_type = 'fusion'
    elif cfg_network['network_arch'] == 'GenericBiStreamImgOnly':
        network = GenericBiStreamImgOnly(cfg_network['backbone_name'], cfg_network['approximate_head_dim'])
        network_type = 'img_only'
    else:
        network = None
        network_type = ''
    return network, network_type


def get_loss_and_optimizer(config, network):
    if config['loss_type'] == 'bce_logits_loss':
        criterion = nn.BCEWithLogitsLoss()
        loss_need_sigmoid = False
    else:
        criterion = None
        loss_need_sigmoid = None

    if config['optimizer_type'] == 'adamw':
        optim = AdamW(network.parameters(), lr=config['lr'], weight_decay=config['model']['wd'])
    elif config['optimizer_type'] == 'sgd':
        optim = SGD(network.parameters(), lr=config['lr'], weight_decay=config['model']['wd'])
    else:
        optim = None
    return criterion, loss_need_sigmoid, optim


def network_inferrence_helper(network, network_type, data_input):
    if network_type == 'img_only':
        img = data_input.cuda(non_blocking=True)
        prd_logits, final_feat = network(img)
    elif network_type == 'fusion':
        img = data_input[0].cuda(non_blocking=True)
        metric = data_input[1].cuda(non_blocking=True)
        prd_logits, final_feat = network([img, metric])
    else:
        prd_logits, final_feat = None
    return prd_logits, final_feat


def default_train_loop(fp16_scaler, epoch, config, dataloader, network, network_type, ema, loss_fn,
                       loss_need_sigmoid, optim, scheduler):
    network.train()
    iter_losses = []
    iter_acc = []
    iter_auc = []
    with tqdm(total=len(dataloader)) as pbar:
        for data_input, gt, _ in dataloader:
            optim.zero_grad()
            gt = torch.unsqueeze(gt, dim=1).float().cuda(non_blocking=True)

            if network_type == 'fusion':
                metric = data_input[1]
                mask = torch.ones_like(metric)
                if epoch <= config['metric_masked_max_epoch'] and config['metric_masked_max_epoch'] > 0:
                    cur_mask_prob = 1.0 - epoch / config['metric_masked_max_epoch']
                else:
                    cur_mask_prob = 0.0
                mask_index = torch.randperm(metric.size(0))[:int(metric.size(0) * cur_mask_prob)]
                mask[mask_index] = 0
                metric = mask * metric
                data_input[1] = metric

            with torch.cuda.amp.autocast():
                prd_logits, _ = network_inferrence_helper(network, network_type, data_input)
                prd_prob = nn.Sigmoid()(prd_logits)
                if loss_need_sigmoid:
                    loss = loss_fn(prd_prob, gt)
                else:
                    loss = loss_fn(prd_logits, gt)
            

            fp16_scaler.scale(loss).backward()
            fp16_scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(network.parameters(), config['gradient_clip'])
            fp16_scaler.step(optim)
            fp16_scaler.update()
            
            if ema is not None:
                ema.update()
            lrs_config = config['lr_scheduler']
            if lrs_config['name'] == 'CyclicLR':
                scheduler.step()
                
            pbar.update(1)

            acc, auc = eval_main_tensor(prd_prob, gt)
            iter_losses.append(float(loss.cpu().detach().numpy()))
            iter_acc.append(float(acc.cpu().numpy()))
            iter_auc.append(float(auc.cpu().numpy()))

    return np.mean(iter_losses), np.mean(iter_acc), np.mean(iter_auc)


def default_test_loop(dataloader, network, network_type, ema):
    network.eval()
    gt_set = []
    pred_set = []
    final_feat_set = []
    patient_id_set = []
    with torch.no_grad():
        for data_input, gt, patient_id in dataloader:
            gt = torch.unsqueeze(gt, dim=1).float().cuda(non_blocking=True)
            if ema is not None:
                with ema.average_parameters():
                    prd_logits, final_feat = network_inferrence_helper(network, network_type, data_input)
            else:
                prd_logits, final_feat = network_inferrence_helper(network, network_type, data_input)
            prd_prob = nn.Sigmoid()(prd_logits)
            gt_set.extend(gt.cpu().numpy())
            pred_set.extend(prd_prob.cpu().numpy())
            final_feat_set.extend(final_feat.cpu().numpy())
            patient_id_set.extend([patient_id])
    acc, auc = eval_main_numpy(np.concatenate(pred_set), np.concatenate(gt_set))
    return np.concatenate(gt_set), np.concatenate(pred_set), np.concatenate(patient_id_set), acc, auc, np.array(final_feat_set)


def restore_checkpoint_helper(checkpoint_dict, item, item_string, logger, checkpoint_path):
    if item is not None:
        if item_string not in checkpoint_dict:
            logging_info(logger, 'warning: ' + item_string + ' state not found in the file :' + str(checkpoint_path) +
                         ' and cannot be loaded.')
        else:
            item.load_state_dict(checkpoint_dict[item_string])


def restore_checkpoint(checkpoint_path, network, optim, scheduler, scaler, ema, logger, resume_training_mode=True):
    checkpoint_dict = torch.load(checkpoint_path)
    curr_network_state_dict = network.state_dict()
    checkpoint_network_state_dict = checkpoint_dict['network']

    if curr_network_state_dict[list(curr_network_state_dict.keys())[-2]].shape == \
            checkpoint_network_state_dict[list(checkpoint_network_state_dict.keys())[-2]].shape \
            and resume_training_mode:
        network.load_state_dict(checkpoint_network_state_dict)
        restore_checkpoint_helper(checkpoint_dict, optim, 'optimizer', logger, checkpoint_path)
        restore_checkpoint_helper(checkpoint_dict, scheduler, 'scheduler', logger, checkpoint_path)
        restore_checkpoint_helper(checkpoint_dict, scaler, 'scaler', logger, checkpoint_path)
        restore_checkpoint_helper(checkpoint_dict, ema, 'ema', logger, checkpoint_path)
    else:
        checkpoint_network_state_dict.pop(list(checkpoint_network_state_dict)[-1])
        checkpoint_network_state_dict.pop(list(checkpoint_network_state_dict)[-1])
        network.load_state_dict(checkpoint_network_state_dict, strict=False)
        logging_info(logger, 'completed loading of network state dict for file: ' + str(checkpoint_path))

    checkpoint_epoch = 0
    if 'epoch' in checkpoint_dict:
        checkpoint_epoch = checkpoint_dict['epoch']
    return checkpoint_epoch


def save_checkpoint_helper(config, epoch, network, optim, scheduler, scaler, ema, ckpt_path):
    state = {'config': config, 'epoch': epoch, 'network': network.state_dict(), 'optimizer': optim.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict()}
    if ema is not None:
        state['ema'] = ema.state_dict()
    else:
        state['ema'] = None 
    torch.save(state, ckpt_path)


def save_checkpoint(config, logger, epoch, network, optim, scheduler, scaler, ema, best_score, val_auc):
    checkpoint_dir = pathlib.join(os.getcwd(), 'checkpoint')
    if epoch != 0 and epoch % config['epoch_saving_interval'] == 0:
        ckpt_path = os.path.join(checkpoint_dir, 'epoch%04d_auc_%04.4f.pth' % (epoch, val_auc))
        save_checkpoint_helper(config, epoch, network, optim, scheduler, scaler, ema, ckpt_path)
        logging_info(logger, 'checkpoint saved: ' + str(ckpt_path))
    if best_score is None:
        best_score = val_auc
    elif best_score < val_auc:
        best_score = val_auc
        ckpt_path1 = os.path.join(checkpoint_dir, 'best_model_epoch%04d_auc_%04.4f.pth' % (epoch, val_auc))
        save_checkpoint_helper(config, epoch, network, optim, scheduler, scaler, ema, ckpt_path1)
        logging_info(logger, 'checkpoint saved: ' + str(ckpt_path1))
        ckpt_path2 = os.path.join(checkpoint_dir, 'best_model_latest.pth')
        save_checkpoint_helper(config, epoch, network, optim, scheduler, scaler, ema, ckpt_path2)
        logging_info(logger, 'checkpoint saved: ' + str(ckpt_path2))
    return best_score


