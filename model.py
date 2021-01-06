import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import he_init

class VAE(nn.Module):




    def __init__(self,data_length,latent_length):
        super(VAE, self).__init__()


        ## encoder
        self.sigmoid = nn.Sigmoid()
        ## Gated_0 layer
        self.encoder_h_0 = nn.Linear(data_length, 300)
        self.encoder_gate_0 = nn.Linear(data_length, 300)
        ## Gated_1 layer
        self.encoder_h_1 = nn.Linear(300, 300)
        self.encoder_gate_1 = nn.Linear(300, 300)

        self.encoder_latent_mean = nn.Linear(300, latent_length)
        self.encoder_latent_sigma = nn.Linear(300, latent_length)
        self.encoder_sigma_act = nn.Hardtanh(-6, 2) ## why?

        ## decoder
        ## Gated_0 layer
        self.decoder_h_0 = nn.Linear(latent_length, 300)
        self.decoder_gate_0 = nn.Linear(latent_length, 300)
        ## Gated_1 layer
        self.decoder_h_1 = nn.Linear(300, 300)
        self.decoder_gate_1 = nn.Linear(300, 300)

        self.decoder_output_mean = nn.Linear(300, data_length)
        self.decoder_output_sigma = nn.Linear(300, data_length)
        self.decoder_sigma_act = nn.Hardtanh(-4.5, 0) ## why?

        # # weights initialization stolen from VampPrior github
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         he_init(m)

    def encoder(self, x):

        ## gated-0
        h_0 = self.encoder_h_0(x)
        gate_0 = self.encoder_gate_0(x)
        gate_0 = self.sigmoid(gate_0)
        x = h_0 * gate_0

        ## gated-1
        h_1 = self.encoder_h_1(x)
        gate_1 = self.encoder_gate_1(x)
        gate_1 = self.sigmoid(gate_1)
        x = h_1 * gate_1

        ## latent-distributions
        latent_mean = self.encoder_latent_mean(x)
        latent_sigma = self.encoder_latent_sigma(x)
        latent_sigma = self.encoder_sigma_act(latent_sigma)

        return latent_mean, latent_sigma

    def decoder(self, x):

        ## gated-0
        h_0 = self.decoder_h_0(x)
        gate_0 = self.decoder_gate_0(x)
        gate_0 = self.sigmoid(gate_0)
        x = h_0 * gate_0

        ## gated-1
        h_1 = self.decoder_h_1(x)
        gate_1 = self.decoder_gate_1(x)
        gate_1 = self.sigmoid(gate_1)
        x = h_1 * gate_1

        ## output-distributions
        output_mean = self.decoder_output_mean(x)
        output_mean = self.sigmoid(output_mean) ## 0-1 scale (why not softmax)??

        ## clamping for FrayFaces ##
        output_mean = torch.clamp(output_mean, min=0. + 1. / 512., max=1. - 1. / 512.) ## output ranging from 0.5/256 - (1-0.5/256)
        ##
        output_sigma = self.decoder_output_sigma(x)
        output_sigma = self.decoder_sigma_act(output_sigma)

        return output_mean, output_sigma



    def forward(self, x):

        ## encoding
        latent_mean, latent_sigma = self.encoder(x)

        # #### Sampling z_prior ## dikos mas papas
        # std = torch.exp(0.5 * latent_sigma)  # log(s^2) = 2log(s) : logvariance -> s=exp[0.5*2log(s)]
        # if latent_mean.device.type == 'cuda':
        #     eps = torch.normal(0, 1, size=std.size()).cuda()
        # else:
        #     eps = torch.normal(0, 1, size=std.size())
        # z_prior = eps.mul(std) + latent_mean  ## reparameterization trick

        ###  reparameterization trick VampPrior
        std = latent_sigma.mul(0.5).exp_()  # log(s^2) = 2log(s) : logvariance -> s=exp[0.5*2log(s)]
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        from torch.autograd import Variable
        eps = Variable(eps)
        z_prior = eps.mul(std).add_(latent_mean)  ## reparameterization trick

        ## decoding
        output_mean, output_sigma = self.decoder(z_prior)

        return output_mean, output_sigma, z_prior, latent_mean, latent_sigma

    def train_model(self, data_loader, optimizer, KL_coef=1):

        self.train()

        for index, data_batch in enumerate(data_loader):

            ## vizualize input
            # plt.imshow(data_batch.data.cpu().numpy()[0].reshape(28,20))
            # plt.show()

            output_mean, output_sigma, z_prior, latent_mean, latent_sigma = self.forward(data_batch)
            ### computing ELBO-loss
            ## KL divergence
            log_z_prior = torch.sum(-0.5 * torch.pow(z_prior, 2), 1) ## summing over the
            log_q_z = torch.sum(-0.5 * (latent_sigma + torch.pow(z_prior - latent_mean, 2) / torch.exp(latent_sigma)), 1)
            KL_loss = - (log_z_prior - log_q_z) ## KL(q||p)=Sum_over( q(z)*log[q(z)/p(z)])

            ### computing reconstruction loss
            ## Considering a discrete logistic distribution for FrayFace dataset (similarly to the paper)
            ## upper pixel of generated output
            data_batch = torch.floor(data_batch * 256) / 256

            ## applying the logistic distributions
            var = torch.exp(output_sigma) ## extracting variance
            logistic_CDF_left = 1 / (1 + torch.exp(- (data_batch - output_mean)/var))
            logistic_CDF_right = 1 / (1 + torch.exp(- (data_batch - output_mean + 1/256)/var))

            R_loss = torch.sum(- torch.log(logistic_CDF_right - logistic_CDF_left + 1e-7), dim=1)

            ## Overall loss considering both image reconstruction and respect to prior z
            loss = R_loss + KL_coef * KL_loss

            ## average over the batch's samples
            loss = torch.mean(loss)

            loss.backward()

            ## updating NN-weights
            optimizer.step()

            ## resetting optimizer
            optimizer.zero_grad()

        return loss, torch.mean(R_loss), torch.mean(KL_loss)

    def val_model(self, data_loader, KL_coef=1):

        self.eval()

        for index, data_batch in enumerate(data_loader):

            ## vizualize input
            # plt.imshow(data_batch.data.cpu().numpy()[0].reshape(28,20))
            # plt.show()

            output_mean, output_sigma, z_prior, latent_mean, latent_sigma = self.forward(data_batch)
            ### computing ELBO-loss
            ## KL divergence
            log_z_prior = torch.sum(-0.5 * torch.pow(z_prior, 2), 1) ## summing over the
            log_q_z = torch.sum(-0.5 * (latent_sigma + torch.pow(z_prior - latent_mean, 2) / torch.exp(latent_sigma)), 1)
            KL_loss = - (log_z_prior - log_q_z) ## KL(q||p)=Sum_over( q(z)*log[q(z)/p(z)])

            ### computing reconstruction loss
            ## Considering a discrete logistic distribution for FrayFace dataset (similarly to the paper)
            ## upper pixel of generated output
            data_batch = torch.floor(data_batch * 256) / 256

            ## applying the logistic distributions
            var = torch.exp(output_sigma) ## extracting variance
            logistic_CDF_left = 1 / (1 + torch.exp(- (data_batch - output_mean)/var))
            logistic_CDF_right = 1 / (1 + torch.exp(- (data_batch - output_mean + 1/256)/var))

            R_loss = torch.sum(- torch.log(logistic_CDF_right - logistic_CDF_left + 1e-7), dim=1)

            ## Overall loss considering both image reconstruction and respect to prior z
            loss = R_loss + KL_coef * KL_loss

            ## average over the batch's samples
            loss = torch.mean(loss)

        return loss, torch.mean(R_loss), torch.mean(KL_loss)
