import cv2
cv2.setNumThreads(1)
import os
import logging
import hydra
import numpy as np
import argparse
import torch
import os.path as pathlib
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from dataloader import get_data_loader
from procedure import get_network, get_loss_and_optimizer, default_train_loop, default_test_loop
from procedure import save_checkpoint, restore_checkpoint
from utils import logging_info
from torch_ema import ExponentialMovingAverage
import random

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self, config, logger):
        self.config = config
        logging_info(logger, str(config))
        network, network_type = get_network(config['model'], config['dataset'])
        logging_info(logger, str(network))
        self.network = network.cuda()
        self.network_type = network_type
        self.model = torch.nn.parallel.DataParallel(self.network)
        if config['ema_decay'] is not None:
            self.ema = ExponentialMovingAverage(network.parameters(), decay=config['ema_decay'])
        else:
            self.ema = None
        self.trn_dataloader, self.val_dataloader, self.test_dataloader = get_data_loader(config, logger)
        self.criterion, self.loss_need_sigmoid, self.optim = get_loss_and_optimizer(config, self.network)
        lrs_config = config['lr_scheduler']
        if lrs_config['name'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optim, factor=lrs_config['plateau_factor'],
                                               patience=lrs_config['plateau_patience'], threshold=0.001,
                                               cooldown=lrs_config['plateau_cooldown'],
                                               min_lr=1.0e-07, mode='max', verbose=True)
        elif lrs_config['name'] == 'CyclicLR':
            self.scheduler = CyclicLR(self.optim, base_lr=lrs_config['base_lr'],
                                      max_lr=lrs_config['max_lr'], step_size_up=lrs_config['step_size_up'], 
                                      mode='exp_range', gamma = lrs_config['gamma'], cycle_momentum=False)
        self.fp16_scaler = torch.cuda.amp.GradScaler(enabled=True)

        checkpoint_dir = pathlib.join(os.getcwd(), 'checkpoint')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.start_epoch = 0
        if os.path.exists(config['pretrained_checkpoint_path']):
            checkpoint_epoch = restore_checkpoint(config['pretrained_checkpoint_path'], self.network,
                                                  self.optim, self.scheduler, self.fp16_scaler, self.ema, logger,
                                                  config['resume_training'])
            self.start_epoch = checkpoint_epoch + 1
        self.logger = logger

    def main_routine(self):
        _, _, _, val_acc, val_auc, _ = default_test_loop(self.val_dataloader, self.model, self.network_type, ema=None)
        _, _, _, test_acc, test_auc, _ = default_test_loop(self.test_dataloader, self.model, self.network_type, ema=None)
        logging_info(self.logger, 'epoch -1, learning rate: %f, ave val batch acc: %f, ave val batch auc: %f, '
                                  'avg test batch acc: %f, avg test batch auc: %f' %
                     (self.optim.param_groups[0]['lr'], val_acc, val_auc, test_acc, test_auc))

        best_score = None
        for epoch in range(self.start_epoch, self.config['max_epochs']):
            trn_loss, trn_acc, trn_auc = default_train_loop(self.fp16_scaler, epoch, self.config, self.trn_dataloader,
                                                            self.model, self.network_type, self.ema,
                                                            self.criterion, self.loss_need_sigmoid, self.optim, self.scheduler)
            _, _, _, val_acc, val_auc, _ = default_test_loop(self.val_dataloader, self.model,
                                                          self.network_type, ema=None)
            _, _, _, test_acc, test_auc, _ = default_test_loop(self.test_dataloader, self.model,
                                                            self.network_type, ema=None)
            logging_info(self.logger, 'epoch %04d, learning rate: %f, avg train loss: %f'
                                      % (epoch, self.optim.param_groups[0]['lr'], trn_loss))
            logging_info(self.logger, 'avg train acc: %f, avg val acc: %f, '
                                      'avg test acc: %f' % (trn_acc, val_acc, test_acc))
            logging_info(self.logger, 'avg train auc: %f, avg val auc: %f, '
                                      'avg test auc: %f' % (trn_auc, val_auc, test_auc))  
            val_score = val_auc
            
            if self.ema is not None:       
                _, _, _, val_acc_ema, val_auc_ema, _ = default_test_loop(self.val_dataloader, self.model,
                                                                      self.network_type, ema=self.ema)
                _, _, _, test_acc_ema, test_auc_ema, _ = default_test_loop(self.test_dataloader, self.model,
                                                                        self.network_type, ema=self.ema)
                logging_info(self.logger, 'avg val acc(with ema): %f, avg test acc(with ema): %f' % (val_acc_ema,
                                                                                                     test_acc_ema))
                logging_info(self.logger, 'avg val auc(with ema): %f, avg test auc(with ema): %f' % (val_auc_ema,
                                                                                                     test_auc_ema))
                val_score = val_auc_ema

            lrs_config = self.config['lr_scheduler']
            if lrs_config['name'] == 'ReduceLROnPlateau':
                self.scheduler.step(val_score)

            torch.cuda.synchronize()
            best_score = save_checkpoint(self.config, self.logger, epoch, self.network, self.optim, self.scheduler,
                                         self.fp16_scaler, self.ema, best_score, val_score)


@hydra.main(config_path="conf", config_name='config')
def hydra_start(cfg):
    torch.cuda.empty_cache()
    seed = cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    trainer = Trainer(cfg, logger)
    trainer.main_routine()


if __name__ == '__main__':
    hydra_start()
    
    
    



