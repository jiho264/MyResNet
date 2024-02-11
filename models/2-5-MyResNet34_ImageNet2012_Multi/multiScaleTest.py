# %% [markdown]
# # Loading Trained Model

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

from src.Mymodel import MyResNet34
import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.v2 import (
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

file_path = f"MyResNet34_{DATASET}_rezero"

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MyResNet34(num_classes=1000, Downsample_option="B").to("cuda")
criterion = nn.CrossEntropyLoss()
file_name = "MyResNet34_ImageNet2012_rezero"


model.load_state_dict(torch.load(file_path + ".pth"))
print(f"Model is loaded from {file_path}.pth")
model.eval()
print("-" * 50)

# %%
criterion = nn.CrossEntropyLoss()
avg_loss = 0
avg_top1_acc = 0
avg_top5_acc = 0

scales = [224, 256, 384, 480, 640]
batch_size_list = [256, 128, 64, 16, 16]
num_workers_list = [8, 8, 8, 8, 8]

for i in range(5):
    test_dataloader =  DataLoader(
        datasets.ImageFolder(
            root="../../data/ImageNet2012/val",
            transform=Compose(
                    [
                        RandomShortestSize(min_size=scales[i] + 1, antialias=True),
                        Compose(
                            [ToImage(), ToDtype(torch.float32, scale=True)]
                        ),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True
                        ),
                        TenCrop(size=scales[i]),
                    ]
                ),
            ),
        batch_size=batch_size_list[i],
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        pin_memory_device="cuda",
    )
    print(f"Dataset {scales[i]}: {len(test_dataloader.dataset)}")
    print(test_dataloader.dataset)
    print("-" * 50)
    with torch.no_grad():
        test_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        for images, labels in tqdm.tqdm(
            test_dataloader, desc=f"Test {scales[i]}", ncols=55
        ):
            for image in images:
                image, labels = image.to(device), labels.to(device)

                outputs = model(image)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                # Top-1 accuracy
                _, predicted_top1 = outputs.max(1)
                total += labels.size(0)
                correct_top1 += predicted_top1.eq(labels).sum().item()

                # Top-5 accuracy
                _, predicted_top5 = outputs.topk(5, 1, largest=True, sorted=True)
                correct_top5 += predicted_top5.eq(labels.view(-1, 1)).sum().item()

        test_loss /= len(test_dataloader)
        test_loss /= 10  # TenCrop
        test_top1_acc = correct_top1 / total
        test_top5_acc = correct_top5 / total

        print(
            f"Dataset {scales[i]}: Loss: {test_loss:.6f}, Top-1 Acc: {test_top1_acc*100:.2f}%, Top-5 Acc: {test_top5_acc*100:.2f}%"
        )
        print("-" * 50)
        avg_loss += test_loss
        avg_top1_acc += test_top1_acc
        avg_top5_acc += test_top5_acc

avg_loss /= len(scales)
avg_top1_acc /= len(scales)
avg_top5_acc /= len(scales)

print(
    f"Avg Loss: {avg_loss:.6f}, Avg Top-1 Acc: {avg_top1_acc*100:.2f}%, Avg Top-5 Acc: {avg_top5_acc*100:.2f}%"
)
