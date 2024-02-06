import torch
import torch.nn as nn
import torch.optim as optim
import torch
import sys, os, tqdm, time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

from src.Earlystopper import EarlyStopper
from src.Mymodel import MyResNet34
from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    Compose,
    RandomCrop,
    RandomShortestSize,
    Normalize,
    CenterCrop,
    ToImage,
    ToDtype,
)
from torchvision import datasets
import torch
from torch.utils.data import DataLoader


from sklearn.decomposition import PCA


class PCAColorAugmentation_for_CIFAR10(object):
    """
    Apply PCA color augmentation to the input image.

    Args:
    - n_components (int): Number of principal components to keep for PCA transformation.

    """

    def __init__(self, n_components=3):
        self.n_components = n_components

    def __call__(self, image):
        """
        Apply PCA color augmentation to the input image.

        Args:
        - image (torch.Tensor): Input image tensor of shape (C, H, W) where C is the number of channels (e.g., 3 for RGB).

        Returns:
        - transformed_image (torch.Tensor): Transformed image tensor after applying PCA color augmentation.
        """

        # Ensure the image tensor is in the format (H, W, C)
        image = image.permute(1, 2, 0)

        # Flatten the image tensor
        flattened_image = image.view(-1, image.shape[-1]).numpy()

        # Apply PCA to the flattened image data
        pca = PCA(n_components=self.n_components)
        pca.fit(flattened_image)
        transformed_flattened_image = pca.transform(flattened_image)

        # Inverse transform to obtain the image in the original space
        inverted_transformed_image = pca.inverse_transform(transformed_flattened_image)

        # Reshape the inverted transformed image to the original shape
        transformed_image = torch.tensor(
            inverted_transformed_image.reshape(image.shape)
        )

        # Ensure the transformed image tensor is in the format (C, H, W)
        transformed_image = transformed_image.permute(2, 0, 1)

        return transformed_image


BATCH = 256
EPOCHS = 120
# %%
train_dataloader = DataLoader(
    datasets.ImageFolder(
        root="../../data/ImageNet2012/train",
        transform=Compose(
            [
                RandomShortestSize(min_size=range(256, 480), antialias=True),
                RandomCrop(size=224),
                Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True
                ),
                PCAColorAugmentation_for_CIFAR10(),
                RandomHorizontalFlip(),
            ]
        ),
    ),
    batch_size=BATCH,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    pin_memory_device="cuda",
)

valid_dataloader = DataLoader(
    datasets.ImageFolder(
        root="../../data/ImageNet2012/val",
        transform=Compose(
            [
                RandomShortestSize(min_size=range(256, 480), antialias=True),
                # VGG에서 single scale로 했을 때는 두 range의 median 값으로 crop함.
                CenterCrop(size=368),
                Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True
                ),
            ]
        ),
    ),
    batch_size=BATCH,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    pin_memory_device="cuda",
)
print("-" * 50)
print("train_dataset : ", train_dataloader.dataset)
print("-" * 50)
print("valid_dataset : ", valid_dataloader.dataset)
print("-" * 50)
# Load the ResNet model
model = MyResNet34(num_classes=1000, Downsample_option="B").to("cuda")


# %% Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
file_name = "MyResNet34_ImageNet2012_rezero"
earlystopper = EarlyStopper(patience=999, model=model, file_name=file_name)

"""loading log file"""
if os.path.exists(file_name + ".pth.tar"):

    checkpoint = torch.load(
        file_name + ".pth.tar",
        map_location=lambda storage, loc: storage.cuda("cuda"),
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    logs = checkpoint["logs"]
    earlystopper.load_state_dict(checkpoint["earlystopper"])

    print("Suceessfully loaded the All setting and Log file.")
    print(file_name)
    print(f"Current epoch is {len(logs['train_loss'])}")
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Current best valid loss: {min(logs['valid_loss'])}")
    print(f"Current best model loss: {earlystopper.best_eval_loss}")
else:
    # Create a dictionary to store the variables
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    lr_log = []

    logs = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "lr_log": lr_log,
    }
    print(file_name, "does not exist. Created a new log.")
print("-" * 50)
# %%
print(f" - file_name : ", file_name)
print(f" - optimizer : ", optimizer)
print(f" - scheduler : ", scheduler.__class__.__name__)
print(f" - scheduler milestone : ", scheduler.milestones)
print(f" - scheduler gamma : ", scheduler.gamma)
print("-" * 50)

# %% Train the model
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_total = 0.0
    running_correct = 0.0
    for images, labels in tqdm.tqdm(train_dataloader, desc=f"{epoch} Train", ncols=55):
        images, labels = images.to("cuda"), labels.to("cuda")
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
    valid_correct = 0
    valid_total = 0
    valid_loss = 0
    with torch.no_grad():
        for images, labels in tqdm.tqdm(
            valid_dataloader, desc=f"{epoch} Valid", ncols=55
        ):
            images, labels = images.to("cuda"), labels.to("cuda")
            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            valid_total += labels.size(0)
            valid_correct += predicted.eq(labels).sum().item()
    valid_loss = valid_loss / len(valid_dataloader)
    valid_acc = valid_correct / valid_total
    print(
        f"   epoch : {epoch+1} |  valid_loss : {valid_loss:.4f} |  valid_acc : {valid_acc*100:.2f}%"
    )

    scheduler.step()
    earlystopper.check(train_loss)
    logs["train_loss"].append(train_loss)
    logs["train_acc"].append(train_acc)
    logs["valid_loss"].append(valid_loss)
    logs["valid_acc"].append(valid_acc)
    logs["lr_log"].append(optimizer.param_groups[0]["lr"])
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "earlystopper": earlystopper.state_dict(),
        "logs": logs,
    }
    torch.save(checkpoint, file_name + ".pth.tar")
    if epoch % 5 == 0:
        torch.save(model.state_dict(), file_name + f"_{epoch}.pth")
        print("Save the model at ", file_name + f"_{epoch}.pth")


print("Finished training")
