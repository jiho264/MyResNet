# %% [markdown]
# # Loading Trained Model

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

from src.Mymodel import MyResNet34
from src.Mymodel import MyResNet_CIFAR
import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.v2 import (
    ToTensor,
    Compose,
    RandomShortestSize,
    Normalize,
    TenCrop,
    ToImage,
    ToDtype,
)

# %%
"""Dataset selection"""
# DATASET = "CIFAR10"
# DATASET = "CIFAR100"
DATASET = "ImageNet2012"

"""Model selection for CIFAR"""
NUM_LAYERS_LEVEL = 5

"""Dataset parameters"""
BATCH = 256
SPLIT_RATIO = 0

"""optimizer parameters"""
OPTIMIZER = "SGD"
# OPTIMIZER = "Adam"
# OPTIMIZER = "Adam_decay"
SCHEDULER = "MultiStepLR"

file_path = ""
if DATASET == "ImageNet2012":
    file_path = f"MyResNet34_{BATCH}_{OPTIMIZER}_{SCHEDULER}"

else:
    file_path = f"MyResNet{NUM_LAYERS_LEVEL*6+2}_{BATCH}_{OPTIMIZER}"

if SPLIT_RATIO != 0:
    file_path += f"_{int(SPLIT_RATIO*100)}"


# %%
class LoadDataset:
    def __init__(self, root, seceted_dataset):
        self.dataset_name = seceted_dataset

        if self.dataset_name[:5] == "CIFAR":
            pass

        elif self.dataset_name == "ImageNet2012":
            self.ImageNetRoot = root + "/" + self.dataset_name + "/"

            """
            각 지정된 스케일에 따라 10 crop해야하는데, 5개 scale들의 평균을 내야하니까 좀 번거로움.
            그치만, 학습 중엔 center crop으로 eval하니, 지금 당장 필요하지는 않음.
            """

            test_data_list = list()
            scales = [224, 256, 384, 480, 640]
            # scales = [640, 480, 384, 256, 224]
            for scale in scales:
                test_data_list.append(
                    datasets.ImageFolder(
                        root=self.ImageNetRoot + "val",
                        transform=Compose(
                            [
                                RandomShortestSize(min_size=scale + 1, antialias=True),
                                TenCrop(size=scale),
                                Compose(
                                    [ToImage(), ToDtype(torch.float32, scale=True)]
                                ),
                                # Normalize(
                                #     mean=[0.485, 0.456, 0.406],
                                #     std=[1, 1, 1],
                                #     inplace=True,
                                # ),
                            ]
                        ),
                    )
                )
            self.test_data_list = test_data_list
            self.classes = 1000
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return


# %%
class MyShortCut:
    def __init__(self) -> None:
        self.preprocessing_test = torch.nn.Sequential(
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[1.0, 1.0, 1.0],
                inplace=True,
            )
        )

        pass


# %%
tmp = LoadDataset(root="../../data", seceted_dataset=DATASET)
COUNT_OF_CLASSES = tmp.classes
test_data = tmp.test_data_list

# %%
# Redirect the output to a file
sys.stdout = open(f"MultiScaleTestLog_{DATASET}_{BATCH}_{OPTIMIZER}.txt", "w")

# %%
scales = [224, 256, 384, 480, 640]
test_dataloader_list = list()
batch_size_list = [256, 128, 96, 64, 28]
num_workers_list = [8, 8, 8, 8, 8]

for i in range(5):
    test_dataloader_list.append(
        DataLoader(
            test_data[i],
            batch_size=batch_size_list[i],
            shuffle=False,
            num_workers=num_workers_list[i],
            pin_memory=True,
            pin_memory_device="cuda",
            persistent_workers=True,
        )
    )
    # print(
    #     test_dataloader_list[i].dataset,
    #     len(test_dataloader_list[i]),
    #     len(test_dataloader_list[i].dataset),
    #     test_dataloader_list[i].batch_size,
    # )

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DATASET == "ImageNet2012":
    model = MyResNet34(num_classes=COUNT_OF_CLASSES, Downsample_option="B").to(device)
    # model = models.resnet34(pretrained=True).to(device)
    # model = models.resnet34(pretrained=False).to(device)
    print(f"ResNet-34 for {DATASET} is loaded.")
else:
    model = MyResNet_CIFAR(
        num_classes=COUNT_OF_CLASSES, num_layer_factor=NUM_LAYERS_LEVEL
    ).to(device)


model.load_state_dict(torch.load(file_path + ".pth"))
print(f"Model is loaded from {file_path}.pth")
# %%
_MyShortCut = MyShortCut()

# %%
criterion = nn.CrossEntropyLoss()
avg_loss = 0
avg_top1_acc = 0
avg_top5_acc = 0

for i in range(len(scales)):
    with torch.no_grad():
        test_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        for images, labels in tqdm.tqdm(
            test_dataloader_list[i], desc=f"Test {scales[i]}", ncols=55
        ):
            for img in images:
                img, labels = img.to(device), labels.to(device)
                _MyShortCut.preprocessing_test(img)

                outputs = model(img)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                # Top-1 accuracy
                _, predicted_top1 = outputs.max(1)
                total += labels.size(0)
                correct_top1 += predicted_top1.eq(labels).sum().item()

                # Top-5 accuracy
                _, predicted_top5 = outputs.topk(5, 1, largest=True, sorted=True)
                correct_top5 += predicted_top5.eq(labels.view(-1, 1)).sum().item()

        test_loss /= len(test_dataloader_list[i])
        test_loss /= 10  # TenCrop
        test_top1_acc = correct_top1 / total
        test_top5_acc = correct_top5 / total

        print(
            f"Dataset {batch_size_list[i]}: Loss: {test_loss}, Top-1 Acc: {test_top1_acc}, Top-5 Acc: {test_top5_acc}"
        )

        avg_loss += test_loss
        avg_top1_acc += test_top1_acc
        avg_top5_acc += test_top5_acc

avg_loss /= len(scales)
avg_top1_acc /= len(scales)
avg_top5_acc /= len(scales)

print(
    f"Avg Loss: {avg_loss}, Avg Top-1 Acc: {avg_top1_acc}, Avg Top-5 Acc: {avg_top5_acc}"
)

# Close the file
sys.stdout.close()
