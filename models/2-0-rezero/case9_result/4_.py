import torch
import sys, os, tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

# from MyImageNetdataloader import LoadDataset, MyShortCut
# from src.Mydataloader import LoadDataset
from src.utils import SingleModelTrainingProcess
from src.Mydataloader import LoadDataset
from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    Compose,
    RandomCrop,
    Normalize,
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
    "SGD",
]
PRINT_PAD_OPTIM = max([len(i) for i in optim_list])

scheduler_list = [
    "MultiStepLR",
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
        ).to("cuda")
        self.preprocessing_test = torch.nn.Sequential(
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ).to("cuda")
        pass


_MyshortCut = MyShortCut()


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

        _training.forward_train(images, labels)

    # %% Forward_test ######################################################################################################
    # if test_dataloader != None:
    for images, labels in tqdm.tqdm(
        test_dataloader, desc=f"{now_epochs} Test", ncols=55
    ):
        images, labels = images.to("cuda"), labels.to("cuda")
        # if _training.is_completed() == True:
        #     pass
        _MyshortCut.preprocessing_test(images)

        _training.forward_eval(images, labels, mode="test")

    # %% summary.. ######################################################################################################
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
    # _training.save_model()
    # Early stopping
    # _ = _training.select_earlystopping_loss_and_check()
    # set zeros
    _training.set_zeros_for_next_epoch()
    print("-" * 50)
