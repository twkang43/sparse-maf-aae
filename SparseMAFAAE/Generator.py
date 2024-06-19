import torch
import torch.nn as nn
from .MAF import MAF

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, num_layers, num_flows, data_len):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.mean = nn.Linear(hidden_size, latent_size)
        self.log_var = nn.Linear(hidden_size, latent_size)

        self.maf = MAF(data_len, num_flows)
    
    def forward(self, x):
        # D -> H
        h, _ = self.lstm(x)

        # H -> L
        mean = self.mean(h)
        log_var = self.log_var(h)

        z = self.reparameterization(mean, log_var)
        maf_z = self.maf(z)

        return maf_z, mean, log_var
    
    def reparameterization(self, mean, log_var):
        std = torch.exp(log_var/2)
        noise = torch.randn_like(std)

        return mean + noise*std

class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # self.gru = nn.GRU(latent_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(latent_size, hidden_size, num_layers, batch_first=True)
        self.reconstruction = nn.Linear(hidden_size, output_size)

    def forward(self, z):
        # L -> H
        h, _ = self.lstm(z)
        # H -> D
        x_hat = self.reconstruction(h)

        return x_hat
    
class Generator(nn.Module):
    def __init__(self, io_size, hidden_size, latent_size, num_layers, num_flows, data_len):
        super(Generator, self).__init__()
        self.io_size = io_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.data_len = data_len

        self.encoder = Encoder(io_size, hidden_size, latent_size, num_layers, num_flows, data_len)
        self.decoder = Decoder(latent_size, hidden_size, io_size, num_layers)

    def forward(self, x):
        maf_z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(maf_z)

        return x_hat, maf_z, (mean, log_var)