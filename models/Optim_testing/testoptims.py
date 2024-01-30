import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import sys, os
import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

from src.Mydataloader import LoadDataset
from src.Mymodel import MyResNet_CIFAR
from src.Earlystopper import EarlyStopper

# %%
"""Dataset selection"""
DATASET = "CIFAR10"
# DATASET = "CIFAR100"
# DATASET = "ImageNet2012"

"""Dataset parameters"""
BATCH = 128
SHUFFLE = True
NUMOFWORKERS = 8
PIN_MEMORY = True

"""optimizer parameters"""
optim_list = [
    "Adam",
    "Adam_decay",
    "SGD",
    "SGD_nasterov",
    "AdamW",
    "AdamW_amsgrad",
    "NAdam",
]
print_pad_len = max([len(i) for i in optim_list])
SCHDULER = "ExponentialLR"
"""Learning rate scheduler parameters"""
NUM_EPOCHS = 100

"""Early stopping parameters"""
EARLYSTOPPINGPATIENCE = NUM_EPOCHS

# %%

tmp = LoadDataset(root="../../data", seceted_dataset=DATASET)
train_data, valid_data, test_data, COUNT_OF_CLASSES = tmp.Unpack()


train_dataloader, valid_dataloader, test_dataloader = tmp.get_dataloader(
    batch_size=BATCH, shuffle=SHUFFLE, num_workers=NUMOFWORKERS, pin_memory=PIN_MEMORY
)


# %%
class Single_model:
    def __init__(self, optimizer, schduler, device="cuda") -> None:
        self.file_name = f"MyResNet32_{BATCH}_{optimizer}_{SCHDULER}"
        self.optim_name = optimizer
        """define model"""
        self.model = MyResNet_CIFAR(
            num_classes=COUNT_OF_CLASSES, num_layer_factor=5
        ).to(device)

        """define loss function"""
        self.criterion = nn.CrossEntropyLoss()

        """define optimizer"""
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters())
        elif optimizer == "Adam_decay":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), weight_decay=1e-4
            )
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
            )
        elif optimizer == "SGD_nasterov":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True,
            )
        elif optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), weight_decay=1e-4
            )
        elif optimizer == "AdamW_amsgrad":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), weight_decay=1e-4, amsgrad=True
            )
        elif optimizer == "NAdam":
            self.optimizer = torch.optim.NAdam(
                self.model.parameters(), weight_decay=1e-4
            )

        """define earlystopper"""
        self.earlystopper = EarlyStopper(
            patience=EARLYSTOPPINGPATIENCE, model=self.model, file_name=self.file_name
        )

        """define learning rate scheduler"""
        if schduler == "ExponentialLR":
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)

        """define scaler"""
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)


# %%
class Single_training(Single_model):
    def __init__(self, optimizer, schduler, device="cuda"):
        super().__init__(optimizer, schduler, device)
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
            eval_loss = []
            valid_acc = []
            test_loss = []
            test_acc = []
            lr_log = []
            self.logs = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": eval_loss,
                "valid_acc": valid_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr_log": lr_log,
            }
            print("File does not exist. Created a new log.")

        self.train_loss = 0.0
        self.running_loss = 0.0
        self.running_corrects = 0
        self.running_total = 0
        self.train_acc = 0

        self.test_loss = 0.0
        self.test_corrects = 0
        self.test_total = 0
        self.test_acc = 0.0
        pass


# %%
each_trainings = list()
for optim in optim_list:
    each_trainings.append(
        Single_training(optimizer=optim, schduler=SCHDULER, device="cuda")
    )

# %%
pre_epochs = len(each_trainings[0].logs["train_loss"])

for epoch in range(NUM_EPOCHS):
    now_epoch = epoch + 1 + pre_epochs
    if now_epoch > NUM_EPOCHS:
        break
    print(f"[Epoch {now_epoch}/{NUM_EPOCHS}] :")
    # %% Forward_train ######################################################################################################
    for images, labels in tqdm.tqdm(
        train_dataloader, desc=f"{now_epoch} Train", ncols=55
    ):
        for _training in each_trainings:
            _training.running_loss = 0.0
            _training.running_corrects = 0
            _training.running_total = 0
            _training.model.train()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                images, labels = images.to(_training.device), labels.to(
                    _training.device
                )
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

    # %% Forward_eval ######################################################################################################
    for (
        images,
        labels,
    ) in test_dataloader:
        for _training in each_trainings:
            _training.model.eval()
            _training.test_loss = 0.0
            _training.test_corrects = 0
            _training.test_total = 0

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
        _training.train_loss = _training.running_loss / len(train_dataloader)
        _training.train_acc = _training.running_corrects / _training.running_total
        _training.logs["train_loss"].append(_training.train_loss)
        _training.logs["train_acc"].append(_training.train_acc)
        _training.test_loss = _training.test_loss / len(test_dataloader)
        _training.test_acc = _training.test_corrects / _training.test_total
        _training.logs["test_loss"].append(_training.test_loss)
        _training.logs["test_acc"].append(_training.test_acc)
        # Save checkpoint #######
        _training.logs["lr_log"].append(_training.optimizer.param_groups[0]["lr"])

        # scheduler #######
        if _training.scheduler.__class__.__name__ == "ReduceLRonPlateau":
            _training.scheduler.step(_training.test_loss)
        elif _training.scheduler.__class__.__name__ in (
            "ExponentialLR",
            "MultiStepLR",
            "CosineAnnealingLR",
        ):
            _training.scheduler.step()
        else:
            pass
        # print #######
        print(
            f"{_training.optim_name.ljust(print_pad_len)} | train : {_training.train_loss:.4f} / {_training.train_acc*100:.2f}% | test : {_training.test_loss:.4f} / {_training.test_acc*100:.2f}%"
        )

        # Save checkpoint ####### save는 제일 나중에
        checkpoint = {
            "model": _training.model.state_dict(),
            "optimizer": _training.optimizer.state_dict(),
            "scaler": _training.scaler.state_dict(),
            "scheduler": _training.scheduler.state_dict(),
            "earlystopper": _training.earlystopper.state_dict(),
            "logs": _training.logs,
        }
        torch.save(checkpoint, _training.file_name + ".pth.tar")

        # Early stopping #######
        if _training.earlystopper.check(_training.test_loss) == True:
            break
    print("-" * 50)

# %%
# add viewer
