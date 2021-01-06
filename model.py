import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import he_init

class VAE(nn.Module):

    def __init__(self, use_gpu, model_path, data_shape, latent_length, dataset_name):
        # author: Irene-Georgios-Ioannis Pair-Programming

        super(VAE, self).__init__()


        ## output_shape=
        self.data_shape = data_shape
        self.dataset_name = dataset_name
        self.latent_length = latent_length
        self.model_path = model_path
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
        output_mean = self.sigmoid(output_mean) ## 0-1 scale (why not softmax)??

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
        z_prior = eps * std + latent_mean  ## reparameterization trick


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
            log_z_prior = torch.sum(-0.5 * torch.pow(z_prior, 2), 1) ## loglikelihood of z_prior being generated by (0,1) Normal Distribution ## summing over the
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
                ## applying the logistic distributions ###as explained in https://medium.com/@smallfishbigsea/an-explanation-of-discretized-logistic-mixture-likelihood-bdfe531751f0
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

    def calculate_likelihood(self, test_loader, numOfSamples=5000, MiniBatchSize=100, KL_coef=1):
        # author: Georgios
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

                ##############################################################
                # author: Ioannis
                output_mean, output_sigma, z_prior, latent_mean, latent_sigma = self.forward(chi_zero)
                ### computing ELBO-loss
                ## KL divergence
                log_z_prior = torch.sum(-0.5 * torch.pow(z_prior, 2),1)  ## loglikelihood of z_prior being generated by (0,1) Normal Distribution ## summing over the
                log_q_z = torch.sum(-0.5 * (latent_sigma + torch.pow(z_prior - latent_mean, 2) / torch.exp(latent_sigma)),1)  ## loglikelihood of z_prior being generated by
                ## predicted Normal Distributions
                KL_loss = - (log_z_prior - log_q_z)  ## KL(q||p)=Sum_over( q(z)*log[q(z)/p(z)])
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


                ## Overall loss considering both image reconstruction and respect to prior z
                loss = R_loss + KL_coef * KL_loss
                ##############################################################
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

            ### plotting and why--george

        return -np.mean(likelihood_test)