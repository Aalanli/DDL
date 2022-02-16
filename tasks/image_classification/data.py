# %%
import torch
from torch.utils.data.dataset import random_split
from torchvision.datasets import Caltech101

import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms.autoaugment as ag

class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target

class CompleteAugment:
    def __init__(self):
        self.augment = T.AutoAugment()
    
    def __call__(self, img, target):
        return self.augment(img), target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def collate_fn(x):
    label = torch.tensor([i[1] for i in x])
    img = [i[0] for i in x]
    return img, label


def img_transforms(x):
    x = x.convert('RGB')
    img = F.normalize(F.to_tensor(x), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img


def build_loader(batch_size, eval_batch_size, workers, train_split_percent=0.8):
    from torch.utils.data import random_split
    augment = ag.AutoAugment(ag.AutoAugmentPolicy.CIFAR10)
    trf = lambda x: img_transforms(augment(x))

    cal = Caltech101('datasets', transform=trf)
    print(len(cal))
    train_size = int(train_split_percent * len(cal))
    eval_size = len(cal) - train_size
    train_data, eval_data = random_split(cal, [train_size, eval_size])

    train_data = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=workers,
                                            collate_fn=collate_fn)
    eval_data = torch.utils.data.DataLoader(eval_data,
                                            batch_size=eval_batch_size,
                                            shuffle=True,
                                            num_workers=workers,
                                            collate_fn=collate_fn)
    return train_data, eval_data

