import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch
import sys, os, tqdm, time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))
from src.resnet import resnet32

# from MyImageNetdataloader import LoadDataset, MyShortCut
# from src.Mydataloader import LoadDataset
from src.utils import SingleModelTrainingProcess
from src.Earlystopper import EarlyStopper
from src.Mymodel import MyResNet_CIFAR
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
from torchvision import datasets

batch_size = 128
trainset = torchvision.datasets.CIFAR10(
    root="../../../data",
    train=True,
    download=True,
    transform=Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
)
train_dataloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    pin_memory_device="cuda",
    persistent_workers=True,
)

testset = torchvision.datasets.CIFAR10(
    root="../../../data",
    train=False,
    download=True,
    transform=Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(
                mean=[0.49139968, 0.48215827, 0.44653124],
                std=[1.0, 1.0, 1.0],
                inplace=True,
            ),
        ]
    ),
)
test_dataloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    pin_memory_device="cuda",
    persistent_workers=True,
)

# Load the ResNet model
# model = MyResNet_CIFAR(num_classes=10, num_layer_factor=5, Downsample_option="A")
model = resnet32().to("cuda")


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
file_name = "resnet32"
earlystopper = EarlyStopper(patience=999, model=model, file_name=file_name)


# %%
class MyShortCut:
    """https://bongjasee.tistory.com/2"""

    def __init__(self) -> None:
        self.preprocessing_train = torch.nn.Sequential(
            Normalize(
                mean=[0.49139968, 0.48215827, 0.44653124],
                std=[1.0, 1.0, 1.0],
                inplace=True,
            ),
            RandomCrop(
                size=32,
                padding=4,
                fill=0,
                padding_mode="constant",
            ),
            RandomHorizontalFlip(p=0.5),
        )
        pass


# checkpoint = torch.load(
#     file_name + ".pth.tar",
#     map_location=lambda storage, loc: storage.cuda("cuda"),
# )
# model.load_state_dict(checkpoint["model"])
# optimizer.load_state_dict(checkpoint["optimizer"])
# scheduler.load_state_dict(checkpoint["scheduler"])

_MyShortCut = MyShortCut()
train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
test_loss = []
test_acc = []
lr_log = []

logs = {
    "train_loss": train_loss,
    "train_acc": train_acc,
    "valid_loss": valid_loss,
    "valid_acc": valid_acc,
    "test_loss": test_loss,
    "test_acc": test_acc,
    "lr_log": lr_log,
}
# Train the model
for epoch in range(200):
    model.train()
    running_loss = 0.0
    running_total = 0.0
    running_correct = 0.0
    for images, labels in tqdm.tqdm(train_dataloader, desc=f"{epoch} Train", ncols=55):
        images, labels = images.to("cuda"), labels.to("cuda")
        _MyShortCut.preprocessing_train(images)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        running_total += labels.size(0)
        running_correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_dataloader)
    train_acc = running_correct / running_total
    print(
        f"   epoch : {epoch+1} | train_loss : {train_loss:.4f} | train_acc : {train_acc*100:.2f}%"
    )

    # Test the model
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_loss = test_loss / len(test_dataloader)
    test_acc = test_correct / test_total
    print(
        f"   epoch : {epoch+1} |  test_loss : {test_loss:.4f} |  test_acc : {test_acc*100:.2f}%"
    )

    scheduler.step()
    earlystopper.check(train_loss)
    logs["train_loss"].append(train_loss)
    logs["train_acc"].append(train_acc)
    logs["test_loss"].append(test_loss)
    logs["test_acc"].append(test_acc)
    logs["lr_log"].append(optimizer.param_groups[0]["lr"])
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "earlystopper": earlystopper.state_dict(),
        "logs": logs,
    }
    torch.save(checkpoint, file_name + ".pth.tar")

print("Finished training")
