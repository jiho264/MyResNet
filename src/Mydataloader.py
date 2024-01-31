import copy
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    Compose,
    RandomCrop,
    RandomShortestSize,
    AutoAugment,
    Normalize,
    CenterCrop,
    ToImage,
    ToDtype,
)
from torchvision.transforms.autoaugment import AutoAugmentPolicy
import matplotlib.pyplot as plt
import numpy as np


class LoadDataset:
    """
    input :
        - root : datasets folder path
        - seceted_data :
            - "CIFAR10" or "CIFAR100" : Load ~ from torchvision.datasets
            - "ImageNet2012" : Load ~ from Local
    pre-processing:
        - CIFAR10, CIFAR100 :
            - Option : split train/valid with (split_ratio):(1-split_ratio) ratio (default split_ratio = 0)
            - train :
                - Compose([ToImage(), ToDtype(scale=True)])
                - Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[1, 1, 1], inplace=True)
                - AutoAugment(interpolation=InterpolationMode.NEAREST, policy=AutoAugmentPolicy.CIFAR10)
                - RandomCrop(size=(32, 32), padding=[4, 4, 4, 4], pad_if_needed=False, fill=0, padding_mode=constant)
                - RandomHorizontalFlip(p=0.5)
            - valid, test :
                - Compose([ToImage(), ToDtype(scale=True)])
        - ImageNet2012 :
            - train :
                - RandomShortestSize(min_size=range(256, 480), antialias=True),
                - RandomCrop(size=224),
                - RandomHorizontalFlip(self.Randp),
                - Compose([ToImage(), ToDtype(scale=True)])
                - Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True),
                - AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            - valid (center croped valid set) :
                - RandomShortestSize(min_size=range(256, 480), antialias=True),
                - CenterCrop(size=368),
                - Compose([ToImage(), ToDtype(scale=True)])
                - Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True),
            - test (10-croped valid set):
                - Define another location. Find [/src/Prediction_for_MultiScaleTest.ipynb]
    output :
        - self.train_data
        - self.valid_data (default : None)
        - self.test_data (default : None)
        - num of classes
    """

    def __init__(self, root, seceted_dataset, split_ratio=0):
        self.Randp = 0.5
        self.dataset_name = seceted_dataset
        self.split_ratio = split_ratio

        if self.dataset_name[:5] == "CIFAR":
            dataset_mapping = {
                "CIFAR100": datasets.CIFAR100,
                "CIFAR10": datasets.CIFAR10,
            }
            cifar_default_transforms = Compose(
                [
                    Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
                    Normalize(
                        mean=[0.49139968, 0.48215827, 0.44653124],
                        std=[1, 1, 1],
                        inplace=True,
                    ),
                    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                    RandomCrop(
                        size=32,
                        padding=4,
                        fill=0,
                        padding_mode="constant",
                    ),
                    RandomHorizontalFlip(self.Randp),
                    # exject mean and std
                    # https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
                    # std=1로 하면 submean
                    # https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Normalize.html#torchvision.transforms.v2.Normalize
                    # Submean(),
                ],
            )
            """CIFAR10, CIFAR100에서는 ref_train에 split ratio대로 적용해서 잘라냄."""
            ref_train = dataset_mapping[self.dataset_name](
                root=root,
                train=True,
                download=False,
                transform=cifar_default_transforms,
            )
            self.test_data = dataset_mapping[self.dataset_name](
                root=root,
                train=False,
                download=False,
                transform=Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
            )

            if self.split_ratio != 0:
                # Split to train and valid set
                total_length = len(ref_train)
                train_length = int(total_length * self.split_ratio)
                valid_length = total_length - train_length
                self.train_data, self.valid_data = random_split(
                    ref_train, [train_length, valid_length]
                )
                # Apply transform at each dataset
                self.train_data.transform = copy.deepcopy(cifar_default_transforms)
                self.valid_data.transform = (
                    Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
                )

                self.train_data.classes = ref_train.classes
                self.valid_data.classes = ref_train.classes

                self.train_data.class_to_idx = ref_train.class_to_idx
                self.valid_data.class_to_idx = ref_train.class_to_idx

            else:
                self.train_data = ref_train
                self.valid_data = None

            #######################################################

        elif self.dataset_name == "ImageNet2012":
            self.ImageNetRoot = root + "/" + self.dataset_name + "/"

            self.train_data = datasets.ImageFolder(
                root=self.ImageNetRoot + "train",
                transform=Compose(
                    [
                        RandomShortestSize(min_size=range(256, 480), antialias=True),
                        RandomCrop(size=224),
                        RandomHorizontalFlip(self.Randp),
                        Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True
                        ),
                        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                    ]
                ),
            )
            """
            train transform 이렇게 하면 시간 2배걸림. 48분걸렸음.
            Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
            Normalize(
                mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True
            ),
            RandomShortestSize(min_size=range(256, 480), antialias=True),
            RandomCrop(size=224),
            AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            RandomHorizontalFlip(self.Randp),
            """
            self.valid_data = datasets.ImageFolder(
                root=self.ImageNetRoot + "val",
                transform=Compose(
                    [
                        RandomShortestSize(min_size=range(256, 480), antialias=True),
                        # VGG에서 single scale로 했을 때는 두 range의 median 값으로 crop함.
                        CenterCrop(size=368),
                        Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True
                        ),
                    ]
                ),
            )
            self.test_data = None

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return

    def Unpack(self, print_info=True):
        if print_info == True:
            print(
                "-----------------------------------------------------------------------"
            )
            print("Dataset : ", self.dataset_name)
            print("- Length of Train Set : ", len(self.train_data))
            if self.valid_data != None:
                print("- Length of Valid Set : ", len(self.valid_data))
            if self.test_data != None:
                print("- Length of Test Set : ", len(self.test_data))
            print("- Count of Classes : ", len(self.train_data.classes))
            print(
                "-----------------------------------------------------------------------"
            )
        return (
            self.train_data,
            self.valid_data,
            self.test_data,
            len(self.train_data.classes),
        )

    def get_dataloader(
        self,
        # dataset,
        batch_size=1,
        shuffle=None,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device="",
        print_info=False,
    ):
        train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory_device=pin_memory_device,
        )

        print("train.transforms =", self.train_data.transform, train_loader.batch_size)

        if self.valid_data != None:
            valid_loader = DataLoader(
                self.valid_data,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                pin_memory_device=pin_memory_device,
            )
            print(
                "valid.transforms =", self.valid_data.transform, valid_loader.batch_size
            )

        else:
            valid_loader = None
        if self.test_data != None:
            test_loader = DataLoader(
                self.test_data,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                pin_memory_device=pin_memory_device,
            )
            print(
                "test.transforms =",
                self.test_data.transform,
                test_loader.batch_size,
            )
            if print_info == True:
                for X, y in test_loader:
                    print(f"Shape of X [N, C, H, W]: {X.shape}")
                    print("mean of X", X.mean(dim=(0, 2, 3)))
                    print(f"Shape of y: {y.shape} {y.dtype}")
                    break

                class_names = test_loader.dataset.classes
                count = 0
                _, axs = plt.subplots(2, 5, figsize=(8, 4))

                for images, labels in test_loader:
                    images = images.numpy()

                    for i in range(len(images)):
                        image = images[i]
                        label = labels[i]
                        image = np.transpose(image, (1, 2, 0))
                        image = np.clip(image, 0, 1)
                        ax = axs[count // 5, count % 5]
                        ax.imshow(image)
                        ax.set_title(f"{class_names[label], label}")
                        ax.axis("off")
                        count += 1

                        if count == 10:
                            break
                    if count == 10:
                        break
                plt.tight_layout()
                plt.show()

        else:
            test_loader = None
        return (train_loader, valid_loader, test_loader)
