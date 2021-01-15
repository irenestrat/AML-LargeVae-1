import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.optim import Optimizer
import math
import os
import gzip

class AdamNormGrad(Optimizer):
    # author: Tomczak (stolen)
    #  A variation of classical Adam optimizer that normalizes the gradients // Code stolen from jmtomczak
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamNormGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                #############################################
                # normalize grdients
                grad = grad / ( torch.norm(grad,2) + 1.e-7 )
                #############################################

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


def he_init(m):
    # author: Tomczak (stolen)
    s = np.sqrt(2. / m.in_features)
    m.weight.data.normal_(0, s)

def set_seeds(seed):
    # author: Ioannis
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def load_dataset(args):
    # author: Ioannis
    if args.data_path.split("/")[1] == "Freyfaces":
        ##vampprior
        if args.vampprior:
            args.mean_pseudoinputs = 0.5
            args.var_pseudoinputs = 1e-4

        with open(args.data_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')[0]

        ## data normalization [half pixel trick] similarly to the paper implementation
        data = (data + 0.5)/256
        args.output_shape = np.asarray([28, 20])
        args.dataset_name = args.data_path.split("/")[1]


    elif args.data_path.split("/")[1] == "MNIST":

        ##vampprior
        if args.vampprior:
            args.mean_pseudoinputs = 0.05
            args.var_pseudoinputs = 1e-6

        data = []
        for file in os.listdir(args.data_path):
            with gzip.open(args.data_path+file, "r") as f:
                magic_number = int.from_bytes(f.read(4), "big")
                image_count = int.from_bytes(f.read(4), "big")
                row_count = int.from_bytes(f.read(4), "big")
                column_count = int.from_bytes(f.read(4), "big")

                image_data = f.read()
                image_data = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
                data.append(image_data)


        data = np.concatenate([data[0], data[1]])

        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2]) / 255
        args.output_shape = np.asarray([28, 28])
        args.dataset_name = args.data_path.split("/")[1]

    return data

def average_bit_per_d(ll):
    print("")