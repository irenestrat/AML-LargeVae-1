import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import he_init
import math

from torch.autograd import Variable

## Non-linear layers
# Slightly modified code from the VampPrior Github; always using an activation function for the layers.
# Also refer to: https://github.com/jmtomczak/vae_vampprior/blob/master/utils/nn.py
class NonLinearSimplified(nn.Module):
    def __init__(self, input_size, output_size, activation, bias=True):
        super(NonLinearSimplified, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        return self.activation( self.linear(x) )


class VAE(nn.Module):

    def __init__(self, use_gpu, model_path, data_shape, latent_length, dataset_name, model_name, num_of_pseudoinputs, output_shape, use_training_data_init, mean_pseudoinputs, var_pseudoinputs):
        # author: Irene-Georgios-Ioannis Pair-Programming

        super(VAE, self).__init__()

        self.data_shape = data_shape
        self.dataset_name = dataset_name
        self.latent_length = latent_length
        self.model_path = model_path
        self.model_name = model_name
        self.num_of_pseudoinputs = num_of_pseudoinputs
        self.image_size = output_shape
        self.use_training_data_init = use_training_data_init
        self.mean_pseudoinputs = mean_pseudoinputs
        self.var_pseudoinputs = var_pseudoinputs
        self.use_gpu = use_gpu

        ## encoder
        self.sigmoid = nn.Sigmoid()

        ## Gated_0 layer
        self.encoder_h_0 = nn.Linear(np.prod(data_shape), 300)
        self.encoder_gate_0 = nn.Linear(np.prod(data_shape), 300)

        ## Gated_1 layer
        self.encoder_h_1 = nn.Linear(300, 300)
        self.encoder_gate_1 = nn.Linear(300, 300)

        self.encoder_latent_mean = nn.Linear(300, self.latent_length)
        self.encoder_latent_sigma = nn.Linear(300, self.latent_length)
        self.encoder_sigma_act = nn.Hardtanh(-6, 2) ## why?

        ## decoder
        ## Gated_0 layer
        self.decoder_h_0 = nn.Linear(latent_length, 300)
        self.decoder_gate_0 = nn.Linear(latent_length, 300)

        ## Gated_1 layer
        self.decoder_h_1 = nn.Linear(300, 300)
        self.decoder_gate_1 = nn.Linear(300, 300)

        self.decoder_output_mean = nn.Linear(300, np.prod(data_shape))
        self.decoder_output_sigma = nn.Linear(300, np.prod(data_shape))
        self.decoder_sigma_act = nn.Hardtanh(-4.5, 0) ## why?

        # weights initialization similar to how the VampPrior paper does it
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)
        
        if(self.model_name == "VampPrior"):
            self.create_pseudoinputs()

    def encoder(self, x):
        # author: Irene-Georgios-Ioannis Pair-Programming

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
        # author: Irene-Georgios-Ioannis Pair-Programming

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
        output_mean = self.sigmoid(output_mean) ## 0-1 scale

        ## clamping for FreyFaces ##
        if self.dataset_name == "Freyfaces":
            output_mean = torch.clamp(output_mean, min=0. + 1. / 512., max=1. - 1. / 512.) ## output ranging from 0.5/256 - (1-0.5/256)
            output_sigma = self.decoder_output_sigma(x)
            output_sigma = self.decoder_sigma_act(output_sigma)
        elif self.dataset_name =="MNIST":
            output_mean = torch.clamp(output_mean, min=1e-5, max=1-1e-5) ## output ranging from 1/256 - 1-1/256
            output_sigma = 0

        return output_mean, output_sigma



    def forward(self, x):
        # author: Irene-Georgios-Ioannis Pair-Programming

        ## encoding
        latent_mean, latent_sigma = self.encoder(x)

        #### Sampling z_prior ##
        std = torch.exp(0.5 * latent_sigma)  # log(s^2) = 2log(s) : logvariance -> s=exp[0.5*2log(s)]
        if self.use_gpu:
            eps = torch.normal(0, 1, size=std.size()).cuda()
        else:
            eps = torch.normal(0, 1, size=std.size())

        ## reparameterization trick
        # We followed the intuition described in this blog post:
        # https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
        z_prior = eps * std + latent_mean 


        ## decoding
        output_mean, output_sigma = self.decoder(z_prior)

        return output_mean, output_sigma, z_prior, latent_mean, latent_sigma

    def generate(self, epoch, N, use_gpu):
        # author: Ioannis

        if use_gpu:
            z_priors = torch.normal(0, 1, size=(N, self.latent_length)).cuda()
        else:
            z_priors = torch.normal(0, 1, size=(N, self.latent_length))

        output_means, _ = self.decoder(z_priors)

        ## plotting outputs
        fig = plt.figure()
        columns = 4
        rows = int(N / 4)
        for index, i in enumerate(range(1, columns * rows + 1)):

            if self.use_gpu:
                imgGenerated = output_means.data.cpu().numpy()[index].reshape(self.data_shape)

            fig.add_subplot(rows, columns, i)
            plt.axis('off')
            plt.imshow(imgGenerated, cmap='gray')

        save_path = self.model_path + 'generate_' + str(epoch) + '.png'
        plt.savefig(save_path)
        plt.close()
        return output_means

    def generate_vamp_prior(self, num_of_generations, use_gpu):
        # Author: Irene-Georgios Pair-Programming
        gen_means = self.means(self.identity_mat)[0:num_of_generations]
        latent_sam_gen_mean, latent_sam_gen_logvar = self.encoder(gen_means)
        std = torch.exp(1/2 * latent_sam_gen_logvar)
        if use_gpu:
            eps = torch.normal(0, 1, size=std.size()).cuda()
        else:
            eps = torch.normal(0, 1, size=std.size())

        ## Reparameterization trick
        # We followed the intuition described in this blog post:
        # https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
        z_sample_rand = eps * std + latent_sam_gen_mean

        # Decoding
        samples_gen_mean, _ = self.decoder(z_sample_rand)
        return samples_gen_mean

    def plot_loss(self, train_loss_history, val_loss_history, test_loss_history, figure_name):
        # author: Ioannis

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(figure_name)
        plt.plot(np.arange(len(train_loss_history)), train_loss_history, label="train")
        plt.plot(np.arange(len(val_loss_history)), val_loss_history, label="test")
        plt.plot(np.arange(len(test_loss_history)), test_loss_history, label="val")

        plt.legend()
        plt.savefig(self.model_path+figure_name+".png")
        plt.close()
    
    def compute_VampPrior(self, latent_representation):
        # Author: Irene-Georgios Pair-Programming
        
        # calculate the pseudoinputs means
        pseudoinput_means = self.means(self.identity_mat)

        # calculate the latent representation for those given data (means)
        z_p_mean, z_p_log_var = self.encoder(pseudoinput_means)

        # expand the latent representations
        z_expand = latent_representation.unsqueeze(1)
        z_p_means = z_p_mean.unsqueeze(0)
        z_p_logvars = z_p_log_var.unsqueeze(0)

        a = torch.sum(-0.5 * (z_p_logvars + torch.pow(z_expand - z_p_means, 2) / torch.exp(z_p_logvars)), 2) - math.log(self.num_of_pseudoinputs)

        # calculate log-sum-exp
        log_prior = torch.max(a, 1)[0] + torch.log(torch.sum(torch.exp(a - (torch.max(a, 1)[0]).unsqueeze(1)), 1))  # MB x 1

        return log_prior

    def train_model(self, data_loader, optimizer, KL_coef=1):
        # author: Ioannis
        self.train()

        ### initializing overall_losses
        loss_all = 0
        R_loss_all = 0
        KL_loss_all = 0

        for index, data_batch in enumerate(data_loader):
            ## vizualize input
            # plt.imshow(data_batch.data.cpu().numpy()[0].reshape(28,20))
            # plt.show()

            output_mean, output_sigma, z_prior, latent_mean, latent_sigma = self.forward(data_batch)
            ### computing ELBO-loss
            ## KL divergence
            if(self.model_name == "standard"):
                log_z_prior = torch.sum(-0.5 * torch.pow(z_prior, 2), 1) ## loglikelihood of z_prior being generated by (0,1) Normal Distribution ## summing over the
            else:
                log_z_prior = self.compute_VampPrior(z_prior)

            log_q_z = torch.sum(-0.5 * (latent_sigma + torch.pow(z_prior - latent_mean, 2) / torch.exp(latent_sigma)), 1) ## loglikelihood of z_prior being generated by
                                                                                                                          ## predicted Normal Distributions
            KL_loss = - (log_z_prior - log_q_z) ## KL(q||p)=Sum_over( q(z)*log[q(z)/p(z)])

            ### computing reconstruction loss
            ## Considering a discrete logistic distribution for FrayFace dataset (similarly to the paper)
            ## upper pixel of generated output
            if self.dataset_name == 'Freyfaces':
                data_batch = torch.floor(data_batch * 256) / 256
                ## applying the logistic distributions ###as explained in https://medium.com/@smallfishbigsea/an-explanation-of-discretized-logistic-mixture-likelihood-bdfe531751f0
                var = torch.exp(output_sigma) ## extracting variance
                logistic_CDF_left = 1 / (1 + torch.exp(- (data_batch - output_mean)/var))
                logistic_CDF_right = 1 / (1 + torch.exp(- (data_batch - output_mean + 1/256)/var))

                R_loss = - torch.sum(torch.log(logistic_CDF_right - logistic_CDF_left + 1e-7), dim=1)

            elif self.dataset_name == 'MNIST':
                # author: Irene-Georgios-Ioannis Pair-Programming
                output_mean = torch.clamp(output_mean, min=1e-5, max=1-1e-5)
                log_bernoulli = data_batch * torch.log(output_mean) + (1 - data_batch) * torch.log(1 - output_mean)
                R_loss = - torch.sum(log_bernoulli, dim=1)

            R_loss = torch.mean(R_loss)
            KL_loss = torch.mean(KL_loss)

            ## Overall loss considering both image reconstruction and respect to prior z
            loss = R_loss + KL_coef * KL_loss

            ## summing losses of all batches
            if self.use_gpu:
                loss_all += loss.data.cpu().numpy()
                R_loss_all += R_loss.data.cpu().numpy()
                KL_loss_all += KL_loss.data.cpu().numpy()
            else:
                loss_all += loss.data.numpy()
                R_loss_all += R_loss.data.numpy()
                KL_loss_all += KL_loss.data.numpy()

            loss.backward()

            ## updating NN-weights
            optimizer.step()

            ## resetting optimizer
            optimizer.zero_grad()

        ## Average Loss over all batches
        loss_all /= len(data_loader)
        R_loss_all /= len(data_loader)
        KL_loss_all /= len(data_loader)

        return loss_all, R_loss_all, KL_loss_all

    def val_model(self, epoch, data_loader, KL_coef=1, test_mode=False):
        # author: Ioannis

        self.eval()

        ### initializing overall_losses
        loss_all = 0
        R_loss_all = 0
        KL_loss_all = 0

        for index, data_batch in enumerate(data_loader):

            ## vizualize input
            # plt.imshow(data_batch.data.cpu().numpy()[0].reshape(28,20))
            # plt.show()

            output_mean, output_sigma, z_prior, latent_mean, latent_sigma = self.forward(data_batch)
            ### computing ELBO-loss
            ## KL divergence
            log_z_prior = torch.sum(-0.5 * torch.pow(z_prior, 2),
                                    1)  ## loglikelihood of z_prior being generated by (0,1) Normal Distribution ## summing over the
            log_q_z = torch.sum(-0.5 * (latent_sigma + torch.pow(z_prior - latent_mean, 2) / torch.exp(latent_sigma)),
                                1)  ## loglikelihood of z_prior being generated by
            ## predicted Normal Distributions
            KL_loss = - (log_z_prior - log_q_z)  ## KL(q||p)=Sum_over( q(z)*log[q(z)/p(z)])

            ### computing reconstruction loss
            ## Considering a discrete logistic distribution for FrayFace dataset (similarly to the paper)
            ## upper pixel of generated output
            if self.dataset_name == 'Freyfaces':
                data_batch = torch.floor(data_batch * 256) / 256
                ## applying the logistic distributions###as explained in https://medium.com/@smallfishbigsea/an-explanation-of-discretized-logistic-mixture-likelihood-bdfe531751f0
                var = torch.exp(output_sigma)  ## extracting variance
                logistic_CDF_left = 1 / (1 + torch.exp(- (data_batch - output_mean) / var))
                logistic_CDF_right = 1 / (1 + torch.exp(- (data_batch - output_mean + 1 / 256) / var))

                R_loss = - torch.sum(torch.log(logistic_CDF_right - logistic_CDF_left + 1e-7), dim=1)

            elif self.dataset_name == 'MNIST':
                # author: Irene-Georgios-Ioannis Pair-Programming
                output_mean = torch.clamp(output_mean, min=1e-5, max=1 - 1e-5)
                log_bernoulli = data_batch * torch.log(output_mean) + (1 - data_batch) * torch.log(1 - output_mean)
                R_loss = - torch.sum(log_bernoulli, dim=1)

            R_loss = torch.mean(R_loss)
            KL_loss = torch.mean(KL_loss)

            ## Overall loss considering both image reconstruction and respect to prior z
            loss = R_loss + KL_coef * KL_loss

            if self.use_gpu:
                loss_all += loss.data.cpu().numpy()
                R_loss_all += R_loss.data.cpu().numpy()
                KL_loss_all += KL_loss.data.cpu().numpy()
            else:
                loss_all += loss.data.numpy()
                R_loss_all += R_loss.data.numpy()
                KL_loss_all += KL_loss.data.numpy()

            ### visualizing the reconstruction of a portion of the first batch for each epoch
            if index == 0:

                fig = plt.figure()
                columns = 4
                rows = 5
                for index,i in enumerate(range(1, columns * rows + 1, 2)):

                    if self.use_gpu:
                        imgGT = data_batch.data.cpu().numpy()[index].reshape(self.data_shape)
                        imgReconstructed = output_mean.data.cpu().numpy()[index].reshape(self.data_shape)
                    else:
                        imgGT = data_batch.data.numpy()[index].reshape(self.data_shape)
                        imgReconstructed = output_mean.data.numpy()[index].reshape(self.data_shape)

                    fig.add_subplot(rows, columns, i)
                    plt.axis('off')
                    plt.imshow(imgGT, cmap='gray')
                    fig.add_subplot(rows, columns, i+1)

                    plt.axis('off')
                    plt.imshow(imgReconstructed, cmap='gray')

                if test_mode:
                    save_path = self.model_path+'test_reconstruction_'+str(epoch)+'.png'
                else:
                    save_path = self.model_path+'val_reconstruction_'+str(epoch)+'.png'

                plt.savefig(save_path)
                plt.close()

        ## Average Loss over all batches
        loss_all /= len(data_loader)
        R_loss_all /= len(data_loader)
        KL_loss_all /= len(data_loader)

        # if test_mode:
            ### call log-likelihood

        return loss_all, R_loss_all, KL_loss_all
    
    def create_pseudoinputs(self):
        # Author: Irene-Georgios Pair-Programming

        # Computing a trainable non-linear layer for the pseudoinputs.
        # image_size stands for the data dimensions, i.e. [28, 20] for FrayFaces, [28, 28] for d-MNIST
        self.means = NonLinearSimplified(self.num_of_pseudoinputs, np.prod(self.image_size), bias=False, activation=nn.Hardtanh(min_val=0.0))

        # init pseudo-inputs
        if self.use_training_data_init:
            self.means.linear.weight.data = self.mean_pseudoinputs #TODO: check
        else:
            self.means.linear.weight.data.normal_(self.mean_pseudoinputs, np.sqrt(self.var_pseudoinputs))

        # create an idle input for calling pseudo-inputs
        self.identity_mat = Variable(torch.eye(self.num_of_pseudoinputs, self.num_of_pseudoinputs), requires_grad=False)
        if self.use_gpu:
            self.identity_mat = self.identity_mat.cuda()

    def calculate_KL_loss(self, chi_zero):
        ##############################################################
        # author: Ioannis
        output_mean, output_sigma, z_prior, latent_mean, latent_sigma = self.forward(chi_zero)
        ### computing ELBO-loss
        ## KL divergence
        log_z_prior = torch.sum(-0.5 * torch.pow(z_prior, 2),1)  ## loglikelihood of z_prior being generated by (0,1) Normal Distribution ## summing over the
        log_q_z = torch.sum(-0.5 * (latent_sigma + torch.pow(z_prior - latent_mean, 2) / torch.exp(latent_sigma)),1)  ## loglikelihood of z_prior being generated by
        ## predicted Normal Distributions
        KL_loss = - (log_z_prior - log_q_z)  ## KL(q||p)=Sum_over( q(z)*log[q(z)/p(z)])

        return KL_loss, output_mean, output_sigma, z_prior, latent_mean, latent_sigma

    def calculate_RE_loss(self, name, chi_zero, output_sigma, output_mean):
        # Author: Ioannis
        ### computing reconstruction loss
        ## Considering a discrete logistic distribution for FrayFace dataset (similarly to the paper)
        ## upper pixel of generated output
        if self.dataset_name == 'Freyfaces':
            chi_zero = torch.floor(chi_zero * 256) / 256
            ## applying the logistic distributions ###as explained in https://medium.com/@smallfishbigsea/an-explanation-of-discretized-logistic-mixture-likelihood-bdfe531751f0
            var = torch.exp(output_sigma)  ## extracting variance
            logistic_CDF_left = 1 / (1 + torch.exp(- (chi_zero - output_mean) / var))
            logistic_CDF_right = 1 / (1 + torch.exp(- (chi_zero - output_mean + 1 / 256) / var))

            R_loss = - torch.sum(torch.log(logistic_CDF_right - logistic_CDF_left + 1e-7), dim=1)

        elif self.dataset_name == 'MNIST':
            # author: Irene-Georgios-Ioannis Pair-Programming
            output_mean = torch.clamp(output_mean, min=1e-5, max=1 - 1e-5)
            log_bernoulli = data_batch * torch.log(output_mean) + (1 - data_batch) * torch.log(1 - output_mean)
            R_loss = - torch.sum(log_bernoulli, dim=1)

        return R_loss

    def calculate_likelihood(self, test_loader, numOfSamples=5000, MiniBatchSize=100, KL_coef=1):
        # Calculating the likelidood in test time.
        # Author: Georgios
        # ---------------------------------------------
        self.eval()

        testData = []
        for data in test_loader:
            testData.append(data)

        testData = torch.cat(testData, 0)

        # initializing the likelihood for all test data to find the max
        numOfData = testData.shape[0]
        likelihood_test = np.zeros(numOfData)
        ratio = max(1, numOfSamples / MiniBatchSize)
        dim = min(numOfSamples, MiniBatchSize)

        for datapoint in range(0, numOfData):
            if (datapoint % 100 == 0): print('{:.2f}%'.format(100 * datapoint / (float(numOfData))))
            chi_star = testData[datapoint].unsqueeze(0)

            losses = []
            for _ in range(0, int(ratio)):
                chi_zero = chi_star.expand(dim, chi_star.size(1))

                KL_loss, output_mean, output_sigma, z_prior, latent_mean, latent_sigma = self.calculate_KL_loss(chi_zero)
                R_loss = self.calculate_RE_loss(self.dataset_name, chi_zero, output_sigma, output_mean)

                ## Overall loss considering both image reconstruction and respect to prior z
                loss = R_loss + KL_coef * KL_loss

                ## Applying the logsumexp trick to avoid NANs in the overall loss computation.
                ## That is, we take into account all the overall losses considering both image reconstruction and respect to prior, for all 5000 samples.
                ## I describe the trick here: https://drive.google.com/drive/folders/1Cow0dU31nWNzeO8e1aMn3szpg_DIhMfH?usp=sharing
                losses.append(-loss.cpu().data.numpy())

            losses = np.asarray(losses)
            redim = losses.shape[0] * losses.shape[1]
            losses = np.reshape(losses, (redim, 1))

            # LOGSUMEXP TRICK
            logsumexp = 0
            losses = losses.ravel()
            loss_max = losses.max(axis=0)
            logsumexp = np.log(sum(np.exp(losses - loss_max)))
            logsumexp += loss_max

            likelihood_test[datapoint] = logsumexp - np.log(redim)

            ### plotting --george

        return -np.mean(likelihood_test)