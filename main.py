# import required libraries
from pytorch_lightning.accelerators import accelerator
from modules import CelebADataModule
from models import DCGAN
import torch, pdb
from pytorch_lightning.trainer import Trainer

data = CelebADataModule(batch_size=512, num_workers=8)
model = DCGAN()
trainer = Trainer(accelerator="gpu", devices=1, max_epochs=50)
trainer.fit(model, data)