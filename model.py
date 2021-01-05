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

    def train(self, data_loader, epochs, batch_size):

        for index, data_batch in enumerate(data_loader):

            self.forward(data_batch)

            # plt.imshow(data_batch.data.cpu().numpy()[0].reshape(28,20))
            # plt.show()

        print("")

        print("")