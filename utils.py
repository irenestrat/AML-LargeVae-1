import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np

def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def load_dataset(path):

    if path.split("/")[1] == "Freyfaces":
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')[0]

        ## data normalization [half pixel trick]
        data = (data + 0.5)/256

    return data