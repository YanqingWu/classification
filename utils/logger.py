import os
import torch
import logging
import argparse
from tensorboardX import SummaryWriter


class logger(logging.Logger):
    def __init__(self, args):
        self.args = args
        super(logger, self).__init__('logger')
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

        if not os.path.exists(self.args.log_file_path):
            os.makedirs(self.args.log_file_path)
        self.log_file = open(os.path.join(self.args.log_file_path, self.args.arch + '_train_logs.txt'), 'w')
        self.writer = SummaryWriter(log_dir=self.args.log_file_path + '/runs')

    def info(self, msg):
        if not isinstance(msg, str):
            msg = str(msg)
        print(msg)
        self.log_file.write(msg + '\n')

    def write(self, msg):
        if isinstance(msg, argparse.Namespace):
            for k, v in self.args.__dict__.items():
                self.log_file.write(str(k) + ' : ' + str(v) + '\n')
            return
        self.log_file.write(msg + '\n')

    def save_checkpoint(self, state, is_best=False):
        if not os.path.exists('trained_models'):
            os.mkdir('trained_models')
        if is_best:
            torch.save(state, 'trained_models/' + self.args.arch + '_model_best.pth')
        else:
            torch.save(state, 'trained_models/' + self.args.arch + '_model_latest.pth')








