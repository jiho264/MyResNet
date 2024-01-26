import sys, os
import torch
import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))
from MyResNet34_256_models_case import ModelCase
from src.Mydataloader import LoadDataset


def single_iter_train(Training):
    # Training loop @@@@
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        images, labels = images.to(Training.device), labels.to(Training.device)
        Training.optimizer.zero_grad()
        outputs = Training.model(images)
        loss = Training.criterion(outputs, labels)

    Training.scaler.scale(loss).backward()
    Training.scaler.step(Training.optimizer)
    Training.scaler.update()

    running_loss += loss.item()
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()

    return running_loss, correct, total


def single_iter_eval(Training):
    for images, labels in tqdm.tqdm(
        valid_dataloader, desc=f"{now_epoch} Eval", ncols=55
    ):
        images, labels = images.to(Training.device), labels.to(Training.device)

        outputs = Training.model(images)
        loss = Training.criterion(outputs, labels)

        eval_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        return eval_loss, correct, total


"""Dataset parameters"""
BATCH = 256
SHUFFLE = True
NUMOFWORKERS = 8
PIN_MEMORY = True
SPLIT_RATIO = 0

"""optimizer parameters"""
OPTIMIZER = "SGD"
# OPTIMIZER = "Adam"
# OPTIMIZER = "Adam_decay"

"""Learning rate scheduler parameters"""
NUM_EPOCHS = 1000

"""Learning rate scheduler parameters"""
SCHEDULER_PARIENCE = 10
COOLDOWN = 20

"""Early stopping parameters"""
EARLYSTOPPINGPATIENCE = 20


tmp = LoadDataset(
    root="../../data", seceted_dataset="ImageNet2012", split_ratio=SPLIT_RATIO
)
train_data, valid_data, _, _ = tmp.Unpack()

train_dataloader, valid_dataloader, _ = tmp.get_dataloader(
    batch_size=BATCH, shuffle=True, num_workers=NUMOFWORKERS, pin_memory=PIN_MEMORY
)

# %%
modelcase1 = ModelCase(
    batch_size=BATCH,
    optimizer="Adam_decay",
    earlystoppingpatience=EARLYSTOPPINGPATIENCE,
    scheduler_parience=SCHEDULER_PARIENCE,
    cooldown=COOLDOWN,
)
Training1 = modelcase1.getTraining()
is_stop1 = False

modelcase2 = ModelCase(
    batch_size=BATCH,
    optimizer="AdamW",
    earlystoppingpatience=EARLYSTOPPINGPATIENCE,
    scheduler_parience=SCHEDULER_PARIENCE,
    cooldown=COOLDOWN,
)
Training2 = modelcase2.getTraining()
is_stop2 = False

# %% [markdown]
for epoch in range(NUM_EPOCHS):
    # 1
    if is_stop1 == False:
        eval_loss1 = Training1.SingleEpoch(train_dataloader, valid_dataloader)
        Training1.Save()
        is_stop1 = Training1.earlystopper.check(eval_loss1)

    # 2
    if is_stop2 == False:
        eval_loss2 = Training2.SingleEpoch(train_dataloader, valid_dataloader)
        Training2.Save()
        is_stop2 = Training2.earlystopper.check(eval_loss2)

    print("-" * 50)


# %% [markdown]
for epoch in range(NUM_EPOCHS):
    # 1
    now_epoch = len(Training1.logs["train_loss"]) + 1
    # Training loop @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    Training1.model.train()
    running_loss1 = 0.0
    correct1 = 0
    total1 = 0
    eval_loss1 = 0.0
    running_loss2 = 0.0
    correct2 = 0
    total2 = 0
    eval_loss2 = 0.0
    for images, labels in tqdm.tqdm(
        train_dataloader, desc=f"{now_epoch} Train", ncols=55
    ):
        if is_stop1 == False:
            running_loss1, correct1, total1 = single_iter_train(Training1)
            train_loss1 = running_loss1 / len(train_dataloader)
            train_acc1 = correct1 / total1
            Training1.logs["train_loss"].append(train_loss1)
            Training1.logs["train_acc"].append(train_acc1)
            Training1.logs["lr_log"].append(Training1.optimizer.param_groups[0]["lr"])
            print(f"Train Loss: {train_loss1:.4f} | Train Acc: {train_acc1*100:.2f}%")
        if is_stop2 == False:
            running_loss2, correct2, total2 = single_iter_train(Training2)
            train_loss2 = running_loss2 / len(train_dataloader)
            train_acc2 = correct2 / total2
            Training2.logs["train_loss"].append(train_loss2)
            Training2.logs["train_acc"].append(train_acc2)
            Training2.logs["lr_log"].append(Training2.optimizer.param_groups[0]["lr"])
            print(f"Train Loss: {train_loss2:.4f} | Train Acc: {train_acc2*100:.2f}%")
    ###########################################################################
    for images, labels in tqdm.tqdm(
        valid_dataloader, desc=f"{now_epoch} Eval", ncols=55
    ):
        if is_stop1 == False:
            eval_loss1, correct1, total1 = single_iter_eval(Training1)
            eval_loss1 = eval_loss1 / len(valid_dataloader)
            valid_acc1 = correct1 / total1
            Training1.logs["valid_loss"].append(eval_loss1)
            Training1.logs["valid_acc"].append(valid_acc1)
            print(f"Valid Loss: {eval_loss1:.4f} | Valid Acc: {valid_acc1*100:.2f}%")
        if is_stop2 == False:
            eval_loss2, correct2, total2 = single_iter_eval(Training2)
            eval_loss2 = eval_loss2 / len(valid_dataloader)
            valid_acc2 = correct2 / total2
            Training2.logs["valid_loss"].append(eval_loss2)
            Training2.logs["valid_acc"].append(valid_acc2)
            print(f"Valid Loss: {eval_loss2:.4f} | Valid Acc: {valid_acc2*100:.2f}%")
    ###########################################################################
    if is_stop1 == False:
        Training1.scheduler.step(eval_loss1)
        Training1.Save()
        is_stop1 = Training1.earlystopper.check(eval_loss1)
    if is_stop2 == False:
        Training2.scheduler.step(eval_loss2)
        Training2.Save()
        is_stop2 = Training2.earlystopper.check(eval_loss2)

    print("-" * 50)
