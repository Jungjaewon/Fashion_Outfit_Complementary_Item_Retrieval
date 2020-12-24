import os
import time
import datetime
import torch
import glob
import os.path as osp
import torch.nn as nn

from model import ConditionalSimNet


class Solver(object):

    def __init__(self, config, data_loader):
        """Initialize configurations."""
        self.config = config
        self.data_loader = data_loader
        self.img_size    = config['MODEL_CONFIG']['IMG_SIZE']

        assert self.img_size in [256]

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']
        self.base_lr       = float(config['TRAINING_CONFIG']['BASE_LR'])
        self.embed_lr      = float(config['TRAINING_CONFIG']['EMBED_LR'])
        self.mask_lr       = float(config['TRAINING_CONFIG']['MASK_LR'])
        self.margin        = float(config['TRAINING_CONFIG']['MARGIN'])
        self.lambda_back   = float(config['TRAINING_CONFIG']['LAMBDA_BACK'])
        self.lambda_sub    = float(config['TRAINING_CONFIG']['LAMBDA_SUB'])
        self.num_outfit = config['TRAINING_CONFIG']['NUM_OUTFIT']
        self.num_negative = config['TRAINING_CONFIG']['NUM_NEGATIVE']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        # torch.dist(x, y, p=2)

        self.cpu_seed = config['TRAINING_CONFIG']['CPU_SEED']
        self.gpu_seed = config['TRAINING_CONFIG']['GPU_SEED']
        #torch.manual_seed(config['TRAINING_CONFIG']['CPU_SEED'])
        #torch.cuda.manual_seed_all(config['TRAINING_CONFIG']['GPU_SEED'])

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.use_tensorboard = config['TRAINING_CONFIG']['USE_TENSORBOARD']

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = os.path.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = os.path.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = os.path.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']
        self.lr_decay_step  = config['TRAINING_CONFIG']['LR_DECAY_STEP']

        self.build_model()

        if self.use_tensorboard == 'True':
            self.build_tensorboard()

    def build_model(self):
        self.CSA = ConditionalSimNet(self.config).to(self.gpu)
        self.optimizer = torch.optim.Adam(self.CSA.parameters(), self.base_lr, (self.beta1, self.beta2))
        self.print_network(self.CSA, 'CSA')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(os.path.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, lr, optimizer):
        """Decay learning rates of the generator and discriminator."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def restore_model(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*-CSA.ckpt'))

        if len(ckpt_list) == 0:
            return 0

        ckpt_list = [int(x[0]) for x in ckpt_list]
        ckpt_list.sort()
        epoch = ckpt_list[-1]
        CSA_path = os.path.join(self.model_dir, '{}-CSA.ckpt'.format(epoch))
        self.CSA.load_state_dict(torch.load(CSA_path, map_location=lambda storage, loc: storage))
        self.CSA.to(self.gpu)

        return epoch

    def dict2gpu(self, data_dict):
        for key in data_dict:
            data_dict[key].to(self.gpu)

    def train(self):

        # Set data loader.
        data_loader = self.data_loader
        iterations = len(self.data_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        start_epoch = self.restore_model()
        start_time = time.time()
        print('Start training...')
        for e in range(start_epoch, self.epoch):

            for i in range(iterations):
                try:
                    data_dict = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    data_dict = next(data_iter)

                self.dict2gpu(data_dict)

                D_p = torch.zeros((self.batch_size, 1))
                D_n = torch.zeros((self.batch_size, 1))

                for n in range(self.num_outfit):
                    f_o = self.CSA(data_dict['outfit_image_{}'.format(n)], data_dict['outfit_onehot_{}'.format(n)])
                    f_p = self.CSA(data_dict['positive_image'], data_dict['outfit_onehot_{}'.format(n)])
                    D_p += torch.mean((f_o - f_p)**2, dim=1)

                for m in range(self.num_negative):
                    D_n_m = torch.zeros((self.batch_size, 1))
                    for n in range(self.num_outfit):
                        f_o = self.CSA(data_dict['outfit_image_{}'.format(n)], data_dict['outfit_onehot_{}'.format(n)])
                        f_n = self.CSA(data_dict['negative_image_{}'.format(n)], data_dict['negative_onehot_{}'.format(n)])
                        D_n_m += torch.mean((f_o - f_n) ** 2, dim=1)
                    D_n += D_n_m

                D_n = D_n / self.batch_size

                loss = self.ranking_loss(D_p, D_n, (torch.ones_like(D_p) * -1).sign())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_dict = dict()
                loss_dict['ranking_loss'] = loss.item()

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss_dict.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:
                G_path = os.path.join(self.model_dir, '{}-CSA.ckpt'.format(e + 1))
                torch.save(self.CSA.state_dict(), G_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

        print('Training is finished')

    def test(self):
        pass

