import torch
from torch import nn
from torch.optim.lr_scheduler import (
    ExponentialLR,
    CosineAnnealingLR,
    CyclicLR,
    MultiStepLR,
    ReduceLROnPlateau,
    ConstantLR,
)
import sys, os, tqdm, time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))
from src.CumstomCosineAnnealingwarmRestarts import CosineAnnealingWarmUpRestarts

# from MyImageNetdataloader import LoadDataset, MyShortCut
from src.Mydataloader import LoadDataset
from src.Mymodel import MyResNet_CIFAR, MyResNet34
from src.Earlystopper import EarlyStopper

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
    # "SGD",
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

# %%

tmp = LoadDataset(root="../../data", seceted_dataset=DATASET)
_, _, _, COUNT_OF_CLASSES = tmp.Unpack()


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
class Single_model:
    def __init__(self, optimizer_name, schduler_name, device="cuda") -> None:
        if DATASET == "ImageNet2012":
            self.file_name = f"MyResNet34_{BATCH}_{optimizer_name}_{schduler_name}"
            self.model = MyResNet34(
                num_classes=COUNT_OF_CLASSES, Downsample_option="B"
            ).to(device)
        elif DATASET == "CIFAR10":
            self.file_name = f"MyResNet32_{BATCH}_{optimizer_name}_{schduler_name}"
            self.model = MyResNet_CIFAR(
                num_classes=COUNT_OF_CLASSES, num_layer_factor=5
            ).to(device)

        self.optim_name = optimizer_name
        self.scheduler_name = schduler_name
        """define model"""

        """define loss function"""
        self.criterion = nn.CrossEntropyLoss()

        """define optimizer"""
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters())
        elif optimizer_name == "Adam_decay":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), weight_decay=1e-4
            )
        elif optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
            )
        elif optimizer_name == "SGD_nasterov":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True,
            )
        elif optimizer_name == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), weight_decay=1e-4
            )
        elif optimizer_name == "AdamW_amsgrad":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), weight_decay=1e-4, amsgrad=True
            )
        elif optimizer_name == "NAdam":
            self.optimizer = torch.optim.NAdam(
                self.model.parameters(), weight_decay=1e-4
            )

        """define earlystopper"""
        self.earlystopper = EarlyStopper(
            patience=EARLYSTOPPINGPATIENCE, model=self.model, file_name=self.file_name
        )

        """define learning rate scheduler"""
        if schduler_name == "ExponentialLR":
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        elif schduler_name == "MultiStepLR":
            self.scheduler = MultiStepLR(self.optimizer, milestones=[30, 60], gamma=0.1)
        elif schduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, patience=5, factor=0.1, cooldown=5
            )
        elif schduler_name == "CosineAnnealingLR":
            """
            - T_max : half of single pariod
            - eta_min : min_lr
            """
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0.001)
        elif schduler_name == "CosineAnnealingWarmUpRestarts":
            """
            초기 lr = near zero여야함.
            - T_0 : single period, 14로 했다가, 10도 테스트 해봄.
            - T_mult : period multiply factor. 2면 다음부터 주기 2배욈
            - eta_max : max_lr. 처음 튀어 오를 lr
            - T_up : warmup period. 튀어오르는데 필요한 epochs.
            - gamma : eta_max decay factor.
            """

            if schduler_name == "CosineAnnealingWarmUpRestarts":
                self.optimizer.param_groups[0]["lr"] = 1e-8
                if optimizer_name == "NAdam":
                    self.scheduler = CosineAnnealingWarmUpRestarts(
                        self.optimizer,
                        T_0=10,
                        T_mult=2,
                        eta_max=0.002,
                        T_up=2,
                        gamma=0.5,
                    )
                elif optimizer_name[:3] == "SGD":
                    self.scheduler = CosineAnnealingWarmUpRestarts(
                        self.optimizer, T_0=10, T_mult=2, eta_max=0.1, T_up=2, gamma=0.5
                    )
                elif optimizer_name[:4] == "Adam":
                    self.scheduler = CosineAnnealingWarmUpRestarts(
                        self.optimizer,
                        T_0=10,
                        T_mult=2,
                        eta_max=0.001,
                        T_up=2,
                        gamma=0.5,
                    )
        elif schduler_name == "ConstantLR":
            self.scheduler = ConstantLR(
                self.optimizer, factor=1, total_iters=NUM_EPOCHS
            )
            pass

        """define scaler"""
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)


# %%
class Single_training(Single_model):
    def __init__(self, optimizer_name, schduler_name, device="cuda"):
        super().__init__(
            optimizer_name=optimizer_name, schduler_name=schduler_name, device=device
        )

        self.device = device
        """loading log file"""
        if os.path.exists(self.file_name + ".pth.tar"):
            # Read checkpoint as desired, e.g.,
            checkpoint = torch.load(
                self.file_name + ".pth.tar",
                map_location=lambda storage, loc: storage.cuda(self.device),
            )
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.earlystopper.load_state_dict(checkpoint["earlystopper"])
            self.logs = checkpoint["logs"]

            print("Suceessfully loaded the All setting and Log file.")
            print(self.file_name)
            print(f"Current epoch is {len(self.logs['train_loss'])}")
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")
        else:
            # Create a dictionary to store the variables
            train_loss = []
            train_acc = []
            valid_loss = []
            valid_acc = []
            test_loss = []
            test_acc = []
            lr_log = []
            self.logs = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr_log": lr_log,
            }
            print(self.file_name, "does not exist. Created a new log.")

        self.train_loss = 0.0
        self.running_loss = 0.0
        self.running_corrects = 0.0
        self.running_total = 0.0
        self.train_acc = 0.0

        self.valid_loss = 0.0
        self.valid_corrects = 0.0
        self.valid_total = 0.0
        self.valid_acc = 0.0

        self.test_loss = 0.0
        self.test_corrects = 0.0
        self.test_total = 0.0
        self.test_acc = 0.0

    def set_zeros_for_next_epoch(self, set_value=0.0):
        self.running_loss = set_value
        self.running_corrects = set_value
        self.running_total = set_value
        self.valid_loss = set_value
        self.valid_corrects = set_value
        self.valid_total = set_value
        self.test_loss = set_value
        self.test_corrects = set_value
        self.test_total = set_value

    def compute_epoch_results(self):
        self.train_loss = self.running_loss / len(train_dataloader)
        self.train_acc = self.running_corrects / self.running_total
        if valid_dataloader != None:
            self.valid_loss = self.valid_loss / len(valid_dataloader)
            self.valid_acc = self.valid_corrects / self.valid_total
        if test_dataloader != None:
            self.test_loss = self.test_loss / len(test_dataloader)
            self.test_acc = self.test_corrects / self.test_total

    def save_model(self, stopflag=False):
        if stopflag == True:
            self.logs["train_loss"].append(-999)
            self.logs["train_acc"].append(-999)
            self.logs["valid_loss"].append(-999)
            self.logs["valid_acc"].append(-999)
            self.logs["test_loss"].append(-999)
            self.logs["test_acc"].append(-999)
            self.logs["lr_log"].append(-999)
        else:
            self.logs["train_loss"].append(self.train_loss)
            self.logs["train_acc"].append(self.train_acc)
            self.logs["valid_loss"].append(self.valid_loss)
            self.logs["valid_acc"].append(self.valid_acc)
            self.logs["test_loss"].append(self.test_loss)
            self.logs["test_acc"].append(self.test_acc)
            self.logs["lr_log"].append(self.optimizer.param_groups[0]["lr"])

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "earlystopper": self.earlystopper.state_dict(),
            "logs": self.logs,
        }
        torch.save(checkpoint, self.file_name + ".pth.tar")

    def print_info(self):
        _length = len(self.logs["train_loss"])
        _epoch_print = f"{_length}/{NUM_EPOCHS} | {self.optim_name.ljust(PRINT_PAD_OPTIM)} - {self.scheduler_name.ljust(PRINT_PAD_SCHDULER)}"
        _epoch_print += (
            " | " + f"Train : {self.train_loss:.4f} / {self.train_acc*100:.2f}%"
        )
        if valid_dataloader != None:
            _valid_print = f"Valid : {self.valid_loss:.4f} / {self.valid_acc*100:.2f}%"
            _epoch_print += " | " + _valid_print
        if test_dataloader != None:
            _test_print = f"Test : {self.test_loss:.4f} / {self.test_acc*100:.2f}%"
            _epoch_print += " | " + _test_print
        print(_epoch_print)

    def scheduling(self):
        if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
            if valid_dataloader == None and test_dataloader != None:
                self.scheduler.step(self.valid_loss)
            else:
                self.scheduler.step(self.train_loss)
        elif self.scheduler.__class__.__name__ in (
            "ExponentialLR",
            "MultiStepLR",
            "CosineAnnealingWarmUpRestarts",
            "CosineAnnealingLR",
            "ConstantLR",
        ):
            self.scheduler.step()
        else:
            raise NotImplementedError

    def earlystopping(self):
        if valid_dataloader != None:
            if self.earlystopper.check(self.valid_loss) == True:
                print(f"Early stopping at {now_epoch} epoch.")
                return True
        elif valid_dataloader == None and test_dataloader != None:
            # testset이 학습에 관여해서는 안 됨.
            if self.earlystopper.check(self.train_loss) == True:
                print(f"Early stopping at {now_epoch} epoch.")
                return True
        elif valid_dataloader == None and test_dataloader == None:
            if self.earlystopper.check(self.train_loss) == True:
                print(f"Early stopping at {now_epoch} epoch.")
                return True
        else:
            return False

    def is_completed(self):
        if len(_training.logs["train_loss"]) > 1:
            if _training.logs["train_loss"][-1] == -999:
                return True
        return False


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
    Single_training(
        optimizer_name="NAdam", schduler_name="ReduceLROnPlateau", device="cuda"
    )
)
print("-" * 50)
each_trainings.append(
    Single_training(optimizer_name="NAdam", schduler_name="MultiStepLR", device="cuda")
)
print("-" * 50)
each_trainings.append(
    Single_training(optimizer_name="SGD", schduler_name="MultiStepLR", device="cuda")
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
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                images, labels = images.to(_training.device), labels.to(
                    _training.device
                )
                """preprocessing"""
                # _MyshortCut.preprocessing_train(images)

                _training.optimizer.zero_grad()
                outputs = _training.model(images)
                loss = _training.criterion(outputs, labels)

            _training.scaler.scale(loss).backward()
            _training.scaler.step(_training.optimizer)
            _training.scaler.update()

            _training.running_loss += loss.item()
            _, predicted = outputs.max(1)
            _training.running_total += labels.size(0)
            _training.running_corrects += predicted.eq(labels).sum().item()
    # %% Forward_valid ######################################################################################################
    if valid_dataloader != None:
        for images, labels in tqdm.tqdm(
            valid_dataloader, desc=f"{now_epoch} Valid", ncols=55
        ):
            for _training in each_trainings:
                if _training.is_completed() == True:  # early stop flag
                    continue
                _training.model.eval()
                with torch.no_grad():
                    images, labels = images.to(_training.device), labels.to(
                        _training.device
                    )
                    """preprocessing"""
                    # _MyshortCut.preprocessing_valid(images)

                    outputs = _training.model(images)
                    loss = _training.criterion(outputs, labels)

                    _training.valid_loss += loss.item()
                    _, predicted = outputs.max(1)
                    _training.valid_total += labels.size(0)
                    _training.valid_corrects += predicted.eq(labels).sum().item()
    # %% Forward_test ######################################################################################################
    if test_dataloader != None:
        for images, labels in tqdm.tqdm(
            test_dataloader, desc=f"{now_epoch} Test", ncols=55
        ):
            for _training in each_trainings:
                if _training.is_completed() == True:  # early stop flag
                    continue
                _training.model.eval()
                with torch.no_grad():
                    images, labels = images.to(_training.device), labels.to(
                        _training.device
                    )
                    outputs = _training.model(images)
                    loss = _training.criterion(outputs, labels)

                    _training.test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    _training.test_total += labels.size(0)
                    _training.test_corrects += predicted.eq(labels).sum().item()

    # %% summary.. ######################################################################################################
    for _training in each_trainings:
        _training.compute_epoch_results()
        # scheduler ######################################################################################################
        _training.scheduling()
        # print ######################################################################################################
        _training.print_info()
        # Save checkpoint ######################################################################################################
        _training.save_model()
        # Early stopping ######################################################################################################
        if _training.earlystopping() == True:
            _training.save_model(stopflag=True)
            continue
        # set zeros ######################################################################################################
        _training.set_zeros_for_next_epoch()
    print("-" * 50)
