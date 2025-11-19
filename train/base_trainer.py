import os
import torch
from tensorboardX import SummaryWriter
from transformers import Adafactor
from utils import colorize
import numpy as np

class BaseTrainer:
    def __init__(self, model, opts, device=None):
        self.model = model
        self.opts = opts
        self.device = device or torch.device('cpu')

        # --- optimizer setup ---
        if opts.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
        elif opts.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr)
        elif opts.optimizer == 'adafactor':
            self.optimizer = Adafactor(
                model.parameters(),
                lr=1e-3,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=0.0,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
        else:
            print('Choose optimizer: adam, adamw, or adafactor')

        print(colorize('%s' % self.optimizer, 'green'))

        self.model.to(self.device)

        # --- directories ---
        self.global_step = 0
        tb_path_train = os.path.join(opts.save_path, 'tb_logs', 'train')
        tb_path_val = os.path.join(opts.save_path, 'tb_logs', 'val')
        self.checkpoints_path = os.path.join(opts.save_path, "checkpoints")
        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(tb_path_train, exist_ok=True)
        os.makedirs(tb_path_val, exist_ok=True)

        self.tb_writer = SummaryWriter(tb_path_train)

    def save_checkpoint(self, ckpt_name):
        save_dict = {'state_dict': self.model.state_dict(), 'global_step': self.global_step}
        torch.save(save_dict, os.path.join(self.checkpoints_path, ckpt_name))

    def load_checkpoint(self, ckpt_paths):
        import glob

        for chkpt in ckpt_paths:
            if not chkpt.endswith('.pt'):
                checkpoints = sorted(glob.glob(f'{chkpt}/checkpoints/*'))
                if checkpoints:
                    chkpt = checkpoints[-1]
                else:
                    raise FileNotFoundError(f"No models found in {chkpt}")

            loaded_state = torch.load(chkpt, map_location=self.device)
            if 'state_dict' in loaded_state:
                if 'global_step' in loaded_state:
                    self.global_step = loaded_state['global_step']
                loaded_state = loaded_state['state_dict']

            self.load_model_params(self.model, loaded_state)
            print(colorize(f"Model {chkpt} loaded!", 'green'))

    @staticmethod
    def load_model_params(model, loaded_state):
        self_state = model.state_dict()
        for name, param in loaded_state.items():
            orig_name = name
            if name not in self_state:
                name = name.replace("module.", "")
            if name not in self_state:
                print(colorize(f"{orig_name} is not in the model.", 'red'))
                continue

            if self_state[name].shape != param.shape:
                if np.prod(param.shape) == np.prod(self_state[name].shape):
                    param = param.reshape(self_state[name].shape)
                else:
                    print(colorize(f"Wrong parameter length: {orig_name}", 'red'))
                    continue

            self_state[name].copy_(param)

    def train(self, dataloader=None, mode='train', epoch=0):
        """Override this in subclass"""
        print(f"Running train() mode={mode} on device={self.device}")
        raise NotImplementedError
