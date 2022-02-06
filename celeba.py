import os, PIL
from torchvision import transforms
from torch.utils.data import Dataset


class CelebA(Dataset):
    def __init__(self, root="data/CelebA"):
        super().__init__()
        self.root = root
        self.transforms = transforms.Compose([transforms.Resize(64),
                                              transforms.CenterCrop(64),
                                              transforms.ToTensor()])
        self.img_files = os.listdir(os.path.join(root, "img_align_celeba"))
        self._length = len(self.img_files)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, "img_align_celeba", self.img_files[idx])
        img = PIL.Image.open(img_name)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = self.transforms(img)
        return img
        