import torch
from types import SimpleNamespace
from utils import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import VAE


args = SimpleNamespace(data_split=[0.7964376590330788, 0.2, 0.2],
                       batch_size=100,
                       shuffle=True,
                       epochs=1000,
                       warm_up_epoch=100,
                       lr=5*1e-4,
                       latent_length=40,
                       use_gpu=torch.cuda.is_available())

## setting seeds
set_seeds(0)

## loading dataset
path = "datasets/Freyfaces/freyfaces.pkl"
dataset = load_dataset(path)

## shuffle data
idx = np.arange(dataset.shape[0])
np.random.shuffle(idx)

## construct train/val/test dataloaders
N = dataset.shape[0] ## length of dataset
args.data_length = dataset.shape[1]

# indices
train_idx = idx[:int(args.data_split[0]*N)]
val_idx = idx[int(args.data_split[0]*N):N-int(args.data_split[2]*N)]
test_idx = idx[N-int(args.data_split[2]*N):]

train_data = dataset[train_idx]
train_data = torch.Tensor(train_data)

val_data = dataset[val_idx]
val_data = torch.Tensor(val_data)

test_data = dataset[test_idx]
test_data = torch.Tensor(test_data)


vae = VAE(args.data_length, args.latent_length)
if args.use_gpu:
    vae.cuda()
    train_loader = DataLoader(train_data.cuda(), batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = DataLoader(val_data.cuda(), batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(test_data.cuda(), batch_size=args.batch_size, shuffle=args.shuffle)
else:
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=args.shuffle)

## initializing optimizer
optimizer = AdamNormGrad(vae.parameters(), lr=args.lr)
optimizer.zero_grad()

for epoch in range(args.epochs):

    ## annealing the KL_coef
    KL_coef = (epoch + 1) / args.warm_up_epoch  ## KL_coef: contribution of prior distribution
    ## done warm upping
    if epoch > args.warm_up_epoch:
        KL_coef = 1

    ## train model
    train_loss, train_R_loss, train_KL_loss = vae.train_model(train_loader, optimizer, KL_coef)

    ## val model
    val_loss, val_R_loss, val_KL_loss = vae.val_model(val_loader)
    print("")

print("")

