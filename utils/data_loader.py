import torch
from os.path import join
import torch.utils.data as data
from utils.augment import augmentation
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class DataLoader:
    def __init__(self, args, input_size):
        self.args = args
        self.train_path = join(args.data, 'train')
        self.val_path = join(args.data, 'val')
        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize(input_size, interpolation=0)

    @staticmethod
    def augment(img):
        return augmentation(img)

    def load_data(self):
        train_dataset = datasets.ImageFolder(self.train_path, transforms.Compose([
                # transforms.Lambda(lambda img: self.augment(img)),
                self.resize,
                transforms.ToTensor(),
                # normalize:batch by batch or picture by picture.
                self.normalization,
            ]))
        self.num_classes = len(train_dataset.classes)
        self.classes = train_dataset.classes
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.val_path, transforms.Compose([
                self.resize,
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                self.normalization,
            ])),
            batch_size=self.args.val_batch, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)

        return train_loader, val_loader