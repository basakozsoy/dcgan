from modules import Generator, Discriminator
import torch, pdb
import numpy as np
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import PIL

# loss functions and optimizers
class DCGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.validation_z = torch.randn(8, 100, 1, 1)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs = batch

        # sample noise
        z = torch.randn(imgs.shape[0], 100, 1, 1, requires_grad = True) # 100: latent dimension
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # ground truth result: i.e all fake (1)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy (BCELoss)
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)

            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx == 1:
            # measure discriminator's ability to classify real
            # from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)
                     
            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,            
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            return output

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=4)
        grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
        grid = grid.cpu().numpy()
        grid = (grid*255).astype(np.uint8)
        PIL.Image.fromarray(grid).save(f"samples/imgs_epoch_{self.current_epoch}.png")

        
