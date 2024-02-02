import torch
import sys, os, tqdm, time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

# from MyImageNetdataloader import LoadDataset, MyShortCut
from src.Mydataloader import LoadDataset
from src.utils import SingleModelTrainingProcess

# %% memo
"""

NAdam + ReduceLROnPlateau 
patience = 5
cooldown = 5
earlystopping = 15

NAdam + MultiStepLR
milestones = [30, 60]
earlystopping = 15

SGD + MultiStepLR
milestones = [30, 60]
earlystopping = 15

"""

ReduceLROnPlateau_patiance = 5
ReduceLROnPlateau_cooldown = 5


# %%
"""Dataset selection"""
# DATASET = "CIFAR10"
# DATASET = "CIFAR100"
DATASET = "ImageNet2012"

"""Dataset parameters"""
BATCH = 256

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
NUM_EPOCHS = 150

"""Early stopping parameters"""
EARLYSTOPPINGPATIENCE = 15

tmp = LoadDataset(root="../../data", seceted_dataset=DATASET)
_, _, _, _ = tmp.Unpack()


train_dataloader, valid_dataloader, test_dataloader = tmp.get_dataloader(
    batch_size=BATCH,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    pin_memory_device="cuda",
    persistent_workers=True,
)
print("-" * 50)

# %%
each_trainings = list()
# for optim_name in optim_list:
#     for schduler_name in scheduler_list:
#         each_trainings.append(
#             Single_training(
#                 optimizer_name=optim_name, schduler_name=schduler_name, device="cuda"
#             )
#         )
#         print("-" * 50)
each_trainings.append(
    SingleModelTrainingProcess(
        dataset=DATASET,
        batch_size=BATCH,
        optimizer_name="NAdam",
        schduler_name="ReduceLROnPlateau",
        device="cuda",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        Earlystopping_patiance=EARLYSTOPPINGPATIENCE,
        ReduceLROnPlateau_patiance=ReduceLROnPlateau_patiance,
        ReduceLROnPlateau_cooldown=ReduceLROnPlateau_cooldown,
    )
)
print("-" * 50)
each_trainings.append(
    SingleModelTrainingProcess(
        dataset=DATASET,
        batch_size=BATCH,
        optimizer_name="NAdam",
        schduler_name="MultiStepLR",
        device="cuda",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        Earlystopping_patiance=EARLYSTOPPINGPATIENCE,
    )
)
print("-" * 50)
each_trainings.append(
    SingleModelTrainingProcess(
        dataset=DATASET,
        batch_size=BATCH,
        optimizer_name="SGD",
        schduler_name="MultiStepLR",
        device="cuda",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        Earlystopping_patiance=EARLYSTOPPINGPATIENCE,
    )
)
print("-" * 50)

# %%

# _MyshortCut = MyShortCut()

pre_epochs = len(each_trainings[0].logs["train_loss"])
for epoch in range(NUM_EPOCHS):
    now_epoch = epoch + 1 + pre_epochs
    if now_epoch > NUM_EPOCHS:
        break
    # %% Forward_train ######################################################################################################
    for images, labels in tqdm.tqdm(
        train_dataloader, desc=f"{now_epoch} Train", ncols=55
    ):
        for _training in each_trainings:
            if _training.is_completed() == True:  # early stop flag
                continue
            _training.model.train()
            _training.forward_train(images, labels)

    # %% Forward_valid ######################################################################################################
    if valid_dataloader != None:
        for images, labels in tqdm.tqdm(
            valid_dataloader, desc=f"{now_epoch} Valid", ncols=55
        ):
            for _training in each_trainings:
                if _training.is_completed() == True:  # early stop flag
                    continue
                _training.model.eval()
                _training.forward_eval(images, labels, mode="valid")

    # %% Forward_test ######################################################################################################
    if test_dataloader != None:
        for images, labels in tqdm.tqdm(
            test_dataloader, desc=f"{now_epoch} Test", ncols=55
        ):
            for _training in each_trainings:
                if _training.is_completed() == True:  # early stop flag
                    continue
                _training.model.eval()
                _training.forward_eval(images, labels, mode="test")

    # %% summary.. ######################################################################################################
    for _training in each_trainings:
        _training.compute_epoch_results()
        # scheduler
        _training.scheduling()
        # print
        _training.print_info(
            num_epochs=NUM_EPOCHS,
            print_pad_optim=PRINT_PAD_OPTIM,
            print_pad_scheduler=PRINT_PAD_SCHDULER,
        )
        # Save checkpoint
        _training.save_model()
        # Early stopping
        if _training.earlystopping(now_epoch=now_epoch) == True:
            _training.save_model(stopflag=True)
            continue
        # set zeros
        _training.set_zeros_for_next_epoch()
    print("-" * 50)
