import torch
import sys, os, tqdm, time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

# from MyImageNetdataloader import LoadDataset, MyShortCut
# from src.Mydataloader import LoadDataset
from src.utils import SingleModelTrainingProcess
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


# %%
"""Dataset selection"""
DATASET = "CIFAR10"
# DATASET = "CIFAR100"
# DATASET = "ImageNet2012"

"""Dataset parameters"""
BATCH = 128

"""optimizer parameters"""
optim_list = [
    # "Adam",
    # "Adam_decay",
    "SGD",
    # "SGD_nasterov",
    # "AdamW",
    # "AdamW_amsgrad",
    # "NAdam",
]
PRINT_PAD_OPTIM = max([len(i) for i in optim_list])

scheduler_list = [
    # "ExponentialLR",
    "MultiStepLR",
    # "ReduceLROnPlateau",
    # "CosineAnnealingLR",
    # "CosineAnnealingWarmUpRestarts",
    # "CycleLR",
    # "ConstantLR",
]
PRINT_PAD_SCHDULER = max([len(i) for i in scheduler_list])

"""Learning rate scheduler parameters"""
NUM_EPOCHS = 180

"""Early stopping parameters"""
EARLYSTOPPINGPATIENCE = 9999

train_dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="../../../data",
        train=True,
        transform=Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
        download=False,
    ),
    batch_size=BATCH,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    pin_memory_device="cuda",
    persistent_workers=True,
)
test_dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root="../../../data",
        train=False,
        transform=Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
        download=False,
    ),
    batch_size=BATCH,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    pin_memory_device="cuda",
    persistent_workers=True,
)
# %%
each_trainings = list()
print("-" * 50)
each_trainings.append(
    SingleModelTrainingProcess(
        dataset=DATASET,
        batch_size=BATCH,
        optimizer_name="SGD",
        schduler_name="MultiStepLR",
        device="cuda",
        use_amp=False,
        train_dataloader=train_dataloader,
        valid_dataloader=None,
        test_dataloader=test_dataloader,
        Earlystopping_patiance=EARLYSTOPPINGPATIENCE,
        MultiStepLR_milestones=[82, 123],
    )
)
print("-" * 50)
print(f"optimizer : ", each_trainings[0].optimizer)
print(f"scheduler : ", each_trainings[0].scheduler)
print(f"scheduler milestone : ", each_trainings[0].scheduler.milestones)
print(f"scheduler gamma : ", each_trainings[0].scheduler.gamma)
print("-" * 50)
"""
=======================================================
if batch = 256
=======================================================
non-split [single epoch = 196 iter] : milestones = [164, 246]
- 1 ~ 164 epochs == 1 ~ 32k iter >> lr = 0.1
- 165~246 epochs == 32k ~ 48k iter >> lr = 0.01
- 247~328(?) epochs == 48k ~ 64k iter >> lr = 0.001
=======================================================
split to 45k/5k [single epoch = 176 iter]: milestones = [182, 273]
- 1~182 epochs == 1 ~ 32k iter >> lr = 0.1
- 182~273 epochs == 32k ~ 48k iter >> lr = 0.01
- 273~364(?) epochs == 48k ~ 64k iter >> lr = 0.001
=======================================================
if batch = 128
=======================================================
non-split [signle epoch = 391 iter]: milestones = [82, 123]
- 1 ~ 82 epochs == 1 ~ 32k iter >> lr = 0.1
- 83~123 epochs == 32k ~ 48k iter >> lr = 0.01
- 124~(164) epochs == 48k ~ 64k iter >> lr = 0.001
=======================================================
split to 45k/5k [signle epoch = 352 iter]: milestones = [91, 137]
- 1~91 epochs == 1 ~ 32k iter >> lr = 0.1
- 92~137 epochs == 32k ~ 48k iter >> lr = 0.01
- 138~(183) epochs == 48k ~ 64k iter >> lr = 0.001
=======================================================
"""


# %%
class MyShortCut:
    """https://bongjasee.tistory.com/2"""

    def __init__(self) -> None:
        self.preprocessing_train = torch.nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomCrop(
                size=32,
                padding=4,
                fill=0,
                padding_mode="constant",
            ),
            RandomHorizontalFlip(p=0.5),
        )
        self.preprocessing_test = torch.nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        pass


_MyshortCut = MyShortCut()
_training = each_trainings[0]
pre_epochs = len(each_trainings[0].logs["train_loss"])
for epoch in range(NUM_EPOCHS):
    now_epochs = epoch + 1 + pre_epochs
    if now_epochs > NUM_EPOCHS:
        break
    # %% Forward_train ######################################################################################################
    for images, labels in tqdm.tqdm(
        train_dataloader, desc=f"{now_epochs} Train", ncols=55
    ):

        images, labels = images.to("cuda"), labels.to("cuda")
        # if _training.is_completed() == True:
        #     pass
        _MyshortCut.preprocessing_train(images)

        # _training.forward_train(images, labels)
        _training.model.train()
        outputs = _training.model(images)  # A
        loss = _training.criterion(outputs, labels)  # B

        _training.optimizer.zero_grad()  # C
        loss.backward()  # D
        _training.optimizer.step()  # E

        _training.train_loss += loss.item()  # F
        _, predicted = outputs.max(1)  # G
        _training.train_total += labels.size(0)  # H
        _training.train_corrects += predicted.eq(labels).sum().item()  # I

    # %% Forward_test ######################################################################################################
    # if test_dataloader != None:
    for images, labels in tqdm.tqdm(
        test_dataloader, desc=f"{now_epochs} Test", ncols=55
    ):
        images, labels = images.to("cuda"), labels.to("cuda")
        # if _training.is_completed() == True:
        #     pass
        _MyshortCut.preprocessing_test(images)

        # _training.forward_eval(images, labels, mode="test")
        _training.model.eval()
        with torch.no_grad():
            outputs = _training.model(images)  # A
            _, predicted = outputs.max(1)  # B
            _training.test_total += labels.size(0)  # C
            _training.test_corrects += predicted.eq(labels).sum().item()  # D
            _training.test_loss += _training.criterion(outputs, labels).item()  # E

    # %% summary.. ######################################################################################################
    # _training.compute_epoch_results()
    _training.train_loss /= len(train_dataloader)
    _training.train_acc = _training.train_corrects / _training.train_total
    _training.test_loss /= len(test_dataloader)
    _training.test_acc = _training.test_corrects / _training.test_total

    # scheduler
    _training.scheduling()
    # print
    _training.print_info(
        now_epochs=now_epochs,
        num_epochs=NUM_EPOCHS,
        print_pad_optim=PRINT_PAD_OPTIM,
        print_pad_scheduler=PRINT_PAD_SCHDULER,
    )

    _training.train_loss = 0
    _training.train_corrects = 0
    _training.train_total = 0
    _training.test_loss = 0
    _training.test_corrects = 0
    _training.test_total = 0
    # Save checkpoint
    # _training.save_model()
    # Early stopping
    # _ = _training.select_earlystopping_loss_and_check()
    # set zeros
    # _training.set_zeros_for_next_epoch()
    print("-" * 50)
