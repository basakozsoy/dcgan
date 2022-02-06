from torch.utils.data import DataLoader
from torch import nn
from celeba import CelebA
import pytorch_lightning as pl


# Lightning data module
class CelebADataModule(pl.LightningDataModule):
    def __init__(self, root="data/CelebA", batch_size=16, num_workers=4):
        super().__init__()
        self.data_dir = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = CelebA(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)


# Generator class
class Generator(nn.Module):
    def __init__(self):
        # latent_dim: Dimension of the latent space
        super().__init__()

        self.model = nn.Sequential(
            # input is z (noise), going into a transpose convolution:
            nn.ConvTranspose2d(in_channels = 100, out_channels = 512, 
                               kernel_size = 4, stride= 1, padding = 0, 
                               bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(in_channels = 512 , out_channels = 256,
                               kernel_size = 4, stride = 2, padding = 1, 
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, 
                               kernel_size =  4, stride =  2, padding = 1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(in_channels = 128, out_channels =  64,
                               kernel_size = 4, stride = 2, padding = 1,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(in_channels = 64, out_channels = 3,
                               kernel_size = 4, stride = 2, padding = 1,
                               bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )
    
    def forward(self, input):
        return self.model(input)


# Discriminator class: 
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(in_channels = 3, out_channels = 64,
                      kernel_size = 4, stride = 2, padding = 1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(in_channels = 64, out_channels = 128,
                      kernel_size = 4, stride = 2, padding = 1,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 16 x 16
            nn.Conv2d(in_channels = 128, out_channels = 256,
                      kernel_size = 4, stride = 2, padding = 1,
                      bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 8 x 8
            nn.Conv2d(in_channels = 256, out_channels = 512,
                      kernel_size = 4, stride = 2, padding = 1,
                      bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 4 x 4
            nn.Conv2d(in_channels = 512, out_channels = 1, 
                      kernel_size = 4, stride = 1, padding = 0,
                      bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.model(input).view(-1, 1)