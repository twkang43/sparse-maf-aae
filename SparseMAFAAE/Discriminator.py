import torch.nn as nn

# Simple Discriminator with Fully-Connected Layers
class SimpleDiscriminator(nn.Module):
    def __init__(self, latent_size):
        super(SimpleDiscriminator, self).__init__()
        self.latent_size = latent_size

        self.discriminator = nn.Sequential(
            nn.Linear(latent_size, latent_size<<1),
            nn.GELU(),
            nn.Linear(latent_size<<1, latent_size),
            nn.GELU(),
            nn.Linear(latent_size, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.discriminator(z)

# Discriminator with Deep Convolution Neural Networks
class DCDiscriminator(nn.Module):
    def __init__(self, latent_size):
        super(DCDiscriminator, self).__init__()
        self.latent_size = latent_size
        self.channel = 64
        self.kernel_size = 1

        self.discriminator = nn.Sequential(
            nn.Conv1d(self.latent_size, self.channel, self.kernel_size),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(self.channel, self.channel<<1, self.kernel_size),
            nn.BatchNorm1d(self.channel<<1),
            nn.LeakyReLU(0.2, True),

            nn.Conv1d(self.channel<<1, self.channel<<2, self.kernel_size),
            nn.BatchNorm1d(self.channel<<2),
            nn.LeakyReLU(0.2, True),

            nn.Flatten(),
            nn.Linear(self.channel<<2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = z.transpose(0,1).unsqueeze(dim=0)
        return self.discriminator(z.transpose(0,1))

# Discriminator with a Transformer
class TransDiscriminator(nn.Module):
    def __init__(self, latent_size):
        super(TransDiscriminator, self).__init__()
        self.latent_size = latent_size

    def forward(self, z):
        return z