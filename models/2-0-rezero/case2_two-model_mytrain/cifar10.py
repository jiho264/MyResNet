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
    "NAdam",
]
PRINT_PAD_OPTIM = max([len(i) for i in optim_list])

scheduler_list = [
    # "ExponentialLR",
    "MultiStepLR",
    "ReduceLROnPlateau",
    # "CosineAnnealingLR",
    # "CosineAnnealingWarmUpRestarts",
    # "CycleLR",
    # "ConstantLR",
]
PRINT_PAD_SCHDULER = max([len(i) for i in scheduler_list])

"""Learning rate scheduler parameters"""
NUM_EPOCHS = 200

"""Early stopping parameters"""
EARLYSTOPPINGPATIENCE = 9999

train_data = datasets.CIFAR10(
    root="../../data",
    train=True,
    transform=Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
    download=True,
)
test_data = datasets.CIFAR10(
    root="../../data",
    train=False,
    transform=Compose([ToImage(), ToDtype(torch.float32, scale=True)]),
    download=True,
)
train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    pin_memory_device="cuda",
    persistent_workers=True,
)
test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=BATCH,
    shuffle=True,
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
        train_dataloader=train_dataloader,
        valid_dataloader=None,
        test_dataloader=test_dataloader,
        Earlystopping_patiance=EARLYSTOPPINGPATIENCE,
    )
)
print("-" * 50)
each_trainings.append(
    SingleModelTrainingProcess(
        dataset="other",
        batch_size=BATCH,
        optimizer_name="SGD",
        schduler_name="MultiStepLR",
        device="cuda",
        train_dataloader=train_dataloader,
        valid_dataloader=None,
        test_dataloader=test_dataloader,
        Earlystopping_patiance=EARLYSTOPPINGPATIENCE,
    )
)
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
            Normalize(
                # mean=[0.49139968, 0.48215827, 0.44653124],
                # std=[1.0, 1.0, 1.0],
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
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
        self.preprocessing_test = torch.nn.Sequential(
            Normalize(
                # mean=[0.49139968, 0.48215827, 0.44653124],
                # std=[1.0, 1.0, 1.0],
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                inplace=True,
            )
        )
        pass


_MyshortCut = MyShortCut()

pre_epochs = len(each_trainings[0].logs["train_loss"])
for epoch in range(NUM_EPOCHS):
    now_epochs = epoch + 1 + pre_epochs
    if now_epochs > NUM_EPOCHS:
        break
    # %% Forward_train ######################################################################################################
    for images, labels in tqdm.tqdm(
        train_dataloader, desc=f"{now_epochs} Train", ncols=55
    ):
        for _training in each_trainings:
            if _training.is_completed() == True:
                pass
            _training.model.train()
            _MyshortCut.preprocessing_train(images.to("cuda"))
            _training.forward_train(images, labels)

    # %% Forward_test ######################################################################################################
    if test_dataloader != None:
        for images, labels in tqdm.tqdm(
            test_dataloader, desc=f"{now_epochs} Test", ncols=55
        ):
            for _training in each_trainings:
                if _training.is_completed() == True:
                    pass
                _training.model.eval()
                _MyshortCut.preprocessing_test(images.to("cuda"))
                _training.forward_eval(images, labels, mode="test")

    # %% summary.. ######################################################################################################
    for _training in each_trainings:
        _training.compute_epoch_results()
        # scheduler
        _training.scheduling()
        # print
        _training.print_info(
            now_epochs=now_epochs,
            num_epochs=NUM_EPOCHS,
            print_pad_optim=PRINT_PAD_OPTIM,
            print_pad_scheduler=PRINT_PAD_SCHDULER,
        )
        # Save checkpoint
        _training.save_model()
        # Early stopping
        _ = _training.select_earlystopping_loss_and_check()
        # set zeros
        _training.set_zeros_for_next_epoch()
    print("-" * 50)
