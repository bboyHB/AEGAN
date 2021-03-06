import torch
from torch import nn
from torch.nn import functional as F


class VanillaVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_size,
                 hidden_dims: list = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        if isinstance(input_size, int):
            self.input_size = (int(input_size), int(input_size))
        elif isinstance(input_size, (tuple, list)) and len(input_size) == 2:
            self.input_size = tuple(input_size)
        else:
            print('input_size error!')
            exit(-1)

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            self.hidden_dims = [32, 64, 128, 256, 512]
        else:
            self.hidden_dims = hidden_dims

        self.beta = 2**len(self.hidden_dims)
        self.latent_fmap_size = (self.input_size[0]//self.beta, self.input_size[1]//self.beta)
        self.before_latent_dim = self.hidden_dims[-1]*self.latent_fmap_size[0]*self.latent_fmap_size[1]

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.before_latent_dim, latent_dim)
        self.fc_var = nn.Linear(self.before_latent_dim, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.before_latent_dim)

        for i in range(len(self.hidden_dims) - 1, 0, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i - 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i - 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[0],
                                               self.hidden_dims[0],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[0]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[0], out_channels=3,
                                      kernel_size=3, padding=1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.latent_fmap_size[0], self.latent_fmap_size[1])
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


if __name__ == '__main__':
    a = torch.rand((1, 3, 128, 128))
    vae = VanillaVAE(3, 2048, 128)
    b = vae(a)
    print(vae.loss_function(*b, M_N=1))


    pass
