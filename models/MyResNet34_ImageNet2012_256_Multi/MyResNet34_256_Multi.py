import sys, os
import torch
import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))
from MyResNet34_256_models_case import ModelCase
from src.Mydataloader import LoadDataset


def single_iter_train(Training, images, labels, running_loss, correct, total):
    Training.model.train()
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


def single_iter_eval(Training, images, labels, eval_loss, correct, total):
    Training.model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
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

"""Learning rate scheduler parameters"""
NUM_EPOCHS = 150

"""Learning rate scheduler parameters"""
SCHEDULER_PARIENCE = 5
COOLDOWN = 20

"""Early stopping parameters"""
EARLYSTOPPINGPATIENCE = 30


tmp = LoadDataset(
    root="../../data", seceted_dataset="ImageNet2012", split_ratio=SPLIT_RATIO
)
train_data, valid_data, _, _ = tmp.Unpack()

train_dataloader, valid_dataloader, test_dataloader = tmp.get_dataloader(
    batch_size=BATCH, shuffle=True, num_workers=NUMOFWORKERS, pin_memory=PIN_MEMORY
)

if valid_dataloader == None:
    eval_dataloader = test_dataloader
elif test_dataloader == None:
    eval_dataloader = valid_dataloader
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
    if is_stop1 == True and is_stop2 == True:
        break
    now_epoch = len(Training1.logs["train_loss"]) + 1
    # Training loop @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    running_loss1 = 0.0
    correct1_train = 0
    total1_train = 0
    running_loss2 = 0.0
    correct2_train = 0
    total2_train = 0
    for images, labels in tqdm.tqdm(
        train_dataloader, desc=f"{now_epoch} Train", ncols=55
    ):
        if is_stop1 == False:
            running_loss1, correct1_train, total1_train = single_iter_train(
                Training1, images, labels, running_loss1, correct1_train, total1_train
            )
        if is_stop2 == False:
            running_loss2, correct2_train, total2_train = single_iter_train(
                Training2, images, labels, running_loss2, correct2_train, total2_train
            )
    ###########################################################################
    eval_loss1 = 0.0
    correct1_test = 0
    total1_test = 0
    eval_loss2 = 0.0
    correct2_test = 0
    total2_test = 0
    for images, labels in tqdm.tqdm(
        eval_dataloader, desc=f"{now_epoch} Eval", ncols=55
    ):
        if is_stop1 == False:
            eval_loss1, correct1_test, total1_test = single_iter_eval(
                Training1, images, labels, eval_loss1, correct1_test, total1_test
            )

        if is_stop2 == False:
            eval_loss2, correct2_test, total2_test = single_iter_eval(
                Training2, images, labels, eval_loss2, correct2_test, total2_test
            )

    ###########################################################################
    if is_stop1 == False:
        train_loss1 = running_loss1 / len(train_dataloader)
        train_acc1 = correct1_train / total1_train
        Training1.logs["train_loss"].append(train_loss1)
        Training1.logs["train_acc"].append(train_acc1)
        Training1.logs["lr_log"].append(Training1.optimizer.param_groups[0]["lr"])

        eval_loss1 = eval_loss1 / len(eval_dataloader)
        valid_acc1 = correct1_test / total1_test
        Training1.logs["valid_loss"].append(eval_loss1)
        Training1.logs["valid_acc"].append(valid_acc1)
        Training1.scheduler.step(eval_loss1)
        Training1.Save()
        is_stop1 = Training1.earlystopper.check(eval_loss1)
        print(
            f"case1 | Train Loss: {train_loss1:.4f} | Train Acc: {train_acc1*100:.2f}%"
        )
        print(
            f"case1 | Valid Loss: {eval_loss1:.4f} | Valid Acc: {valid_acc1*100:.2f}%"
        )
    else:
        print("case1 | Early Stopping")

    if is_stop2 == False:
        train_loss2 = running_loss2 / len(train_dataloader)
        train_acc2 = correct2_train / total2_train
        Training2.logs["train_loss"].append(train_loss2)
        Training2.logs["train_acc"].append(train_acc2)
        Training2.logs["lr_log"].append(Training2.optimizer.param_groups[0]["lr"])

        eval_loss2 = eval_loss2 / len(eval_dataloader)
        valid_acc2 = correct2_test / total2_test
        Training2.logs["valid_loss"].append(eval_loss2)
        Training2.logs["valid_acc"].append(valid_acc2)
        Training2.scheduler.step(eval_loss2)
        Training2.Save()
        is_stop2 = Training2.earlystopper.check(eval_loss2)
        print(
            f"case2 | Train Loss: {train_loss2:.4f} | Train Acc: {train_acc2*100:.2f}%"
        )
        print(
            f"case2 | Valid Loss: {eval_loss2:.4f} | Valid Acc: {valid_acc2*100:.2f}%"
        )
    else:
        print("case2 | Early Stopping")

    print("-" * 50)
