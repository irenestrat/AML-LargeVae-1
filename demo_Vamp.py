import torch
from types import SimpleNamespace
from utils_Vamp import *
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model_Vamp import VAE
import os

# author: Irene-Georgios-Ioannis Pair-Programming
args = SimpleNamespace(data_split=[0.7964376590330788, 0.10178117048346058,0.10178117048346058],
                       batch_size=100,
                       shuffle=True,
                       epochs=1,
                       warm_up_epoch=100,
                       early_stop_epoch=30,
                       lr=5*1e-4,
                       latent_length=40,
                       # data_path='datasets/Freyfaces/freyfaces.pkl',
                       # model_path='standard_Freyfaces/',
                       data_path='datasets/MNIST/',
                       model_path='standard_MNIST/',
                       use_gpu=torch.cuda.is_available(),
                       ## Vampprior
                       vampprior=True,
                       num_of_pseudoinputs=500,
                       use_training_data_init=False
                       )


## setting seeds
set_seeds(0)

## loading dataset
dataset = load_dataset(args)
# dataset, args.output_shape, args.dataset_name = load_dataset(args.data_path)


## shuffle data
idx = np.arange(dataset.shape[0])
np.random.shuffle(idx)

## construct train/val/test dataloaders
N = dataset.shape[0] ## length of dataset

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


vae = VAE(args)
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

train_loss_history = []
train_R_loss_history = []
train_KL_loss_history = []

val_loss_history = []
val_R_loss_history = []
val_KL_loss_history = []


test_loss_history = []
test_R_loss_history = []
test_KL_loss_history = []

best_val_loss = np.inf
early_stop_val_loss = np.inf
early_stop_counter = 0

### creating model folder
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

for epoch in range(args.epochs):
    # author: Ioannis

    ## annealing the KL_coef
    KL_coef = (epoch + 1) / args.warm_up_epoch  ## KL_coef: contribution of prior distribution
    ## done warm upping
    if epoch > args.warm_up_epoch:
        KL_coef = 1

    ## train model
    train_loss, train_R_loss, train_KL_loss = vae.train_model(train_loader, optimizer, KL_coef)


    ## val model
    val_loss, val_R_loss, val_KL_loss = vae.val_model(epoch, val_loader)

    ### testing
    test_loss, test_R_loss, test_KL_loss = vae.val_model(epoch, test_loader, test_mode=True)

    ### generate data similar to the dataset
    vae.generate(epoch, 12, args.use_gpu)

    ## storing loss per epoch
    train_loss_history.append(train_loss)
    train_R_loss_history.append(train_R_loss)
    train_KL_loss_history.append(train_KL_loss)

    val_loss_history.append(val_loss)
    val_R_loss_history.append(val_R_loss)
    val_KL_loss_history.append(val_KL_loss)

    test_loss_history.append(test_loss)
    test_R_loss_history.append(test_R_loss)
    test_KL_loss_history.append(test_KL_loss)

    ## saving weights if better model was found
    if val_loss_history[-1] < best_val_loss:
        best_val_loss = val_loss_history[-1]
        torch.save(vae.state_dict(), args.model_path + "/best_model.pth")
        print("Saving best model found at epoch:", epoch+1)

    ## early stopping
    if val_loss_history[-1] >= early_stop_val_loss:
        early_stop_counter += 1
    else:
        early_stop_val_loss = val_loss_history[-1]
        early_stop_counter = 0
        early_stop_val_loss = np.inf

    if early_stop_counter == args.early_stop_epoch:
        print("Validation loss has not decreased in a long time -- Early Stop -- ")
        break

    ### Printing epoch results
    print('Epoch: {}/{} - EarlyStopIndex: {} \n'
          'Train ~ loss: {:.2f} R_loss: {:.2f}  KL_loss: {:.2f}\n'
          'Val ~ loss: {:.2f} R_loss: {:.2f}  KL_loss: {:.2f}\n'
          'Test ~ loss: {:.2f} R_loss: {:.2f}  KL_loss: {:.2f}\n'.format(epoch+1, args.epochs, early_stop_counter,
                                                                           train_loss, train_R_loss, train_KL_loss,
                                                                           val_loss, val_R_loss, val_KL_loss,
                                                                           test_loss, test_R_loss, test_KL_loss))


## plotting losses
vae.plot_loss(train_loss_history, val_loss_history, test_loss_history, "Loss")
vae.plot_loss(train_R_loss_history, val_R_loss_history, test_R_loss_history, "Reconstruction Loss")
vae.plot_loss(train_KL_loss_history, val_KL_loss_history, test_KL_loss_history, "KL Loss")


## calculate likelihoods for tests when done
vae.calculate_likelihood(test_loader)
