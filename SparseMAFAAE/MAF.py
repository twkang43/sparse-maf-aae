import torch
import torch.nn as nn
from utils import set_device

DEVICE = set_device.set_device()

class MADE(nn.Module):
    def __init__(self, io_size):
        super(MADE, self).__init__()
        self.io_size = io_size # N
        self.latent_size = io_size # N

        # Binary mask matrix for Masked Autoencoders
        self.encoder_mask = torch.tril(torch.ones(self.io_size, self.latent_size)).to(DEVICE) # N x N
        self.encoder_mask.fill_diagonal_(0)

        self.decoder_mask = torch.tril(torch.ones(self.latent_size, self.io_size)).to(DEVICE) # N x N

       # Regular Autoencoder
        self.encoder = nn.Linear(self.io_size, self.latent_size, bias=False).to(DEVICE) # Encoder
        self.decoder = nn.Linear(self.latent_size, self.io_size, bias=False).to(DEVICE) # Decoder

        self.encoder.weight.data *= self.encoder_mask.transpose(0, 1)
        self.decoder.weight.data *= self.decoder_mask.transpose(0, 1)

        self.gelu = nn.GELU()

    def forward(self, z):
        self.encoder.weight.data *= self.encoder_mask.transpose(0, 1)
        self.decoder.weight.data *= self.decoder_mask.transpose(0, 1)
        
        # N x L => L x N => L x N//2
        z_latent = self.gelu(self.encoder(z.transpose(0,1)))
        # L x N//2 => L x N => N x L
        z_hat = self.decoder(z_latent).transpose(0,1)

        return z_hat

class MAF(nn.Module):
    def __init__(self, data_len, num_flows):
        super(MAF, self).__init__()
        self.made_layers = nn.ModuleList()

        for _ in range(num_flows):
            self.made_layers.append(MADE(data_len))

    def forward(self, z):
        for layer in self.made_layers:
            z = z + layer(z)

        return z