import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from einops import rearrange
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from torchvision import transforms


img_normalize = transforms.Compose([transforms.Normalize(mean=[0.8525, 0.8530, 0.8474], std=[0.3088, 0.3043, 0.3135])])


class DataData(Dataset):
    def __init__(self, root_dir="./cat_dog", image_size=224):
        
        path = Path(root_dir)
        self.data_path = list(path.rglob("*.[jp][pn]g"))

        self.image_size = image_size


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):

        image_path = self.data_path[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print("error")
            return torch.zeros((3, self.image_size, self.image_size))
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        image = np.array(image).astype(float)
        image /= 255

        image = torch.tensor(image)

        image = rearrange(image, 'h w c -> c h w')
        
        image = image.to(torch.float32)

        image = img_normalize(image)

        return image, torch.tensor(1) if "cat" in str(image_path.stem) else torch.tensor(0)
