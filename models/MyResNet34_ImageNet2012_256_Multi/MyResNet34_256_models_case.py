import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from fvcore.nn import FlopCountAnalysis, flop_count_table
import sys, os

from src.Mydataloader import LoadDataset
from src.Mymodel import MyResNet34
from src.Mymodel import MyResNet_CIFAR
from src.Mytraining import DoTraining
from src.Earlystopper import EarlyStopper
from src.LogViewer import LogViewer


class ModelCase:
    def __init__(
        self,
        batch_size,
        optimizer,
        earlystoppingpatience,
        scheduler_parience,
        cooldown,
    ):
        model = MyResNet34(num_classes=1000, Downsample_option="B").to("cuda")
        print(f"ResNet-34 for ImageNet2012 is loaded.")
        file_name = f"MyResNet34_{batch_size}_{optimizer}"
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters())
        elif optimizer == "Adam_decay":
            optimizer = torch.optim.Adam(
                model.parameters(), weight_decay=1e-4, foreach=True
            )
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001
            )
        elif optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(), weight_decay=1e-4, amsgrad=True, foreach=True
            )
        else:
            raise ValueError("Optimizer is not defined.")

        criterion = nn.CrossEntropyLoss()

        earlystopper = EarlyStopper(
            patience=earlystoppingpatience, model=model, file_name=file_name
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=scheduler_parience,
            factor=0.1,
            verbose=True,
            threshold=1e-4,
            min_lr=1e-6,
            cooldown=cooldown,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        if os.path.exists(file_name + ".pth.tar"):
            # Read checkpoint as desired, e.g.,
            checkpoint = torch.load(
                file_name + ".pth.tar",
                map_location=lambda storage, loc: storage.cuda("cuda"),
            )
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            earlystopper.load_state_dict(checkpoint["earlystopper"])
            logs = checkpoint["logs"]

            print("Suceessfully loaded the All setting and Log file.")
            print(file_name)
            print(f"Current epoch is {len(logs['train_loss'])}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        else:
            # Create a dictionary to store the variables
            train_loss = []
            train_acc = []
            eval_loss = []
            valid_acc = []
            test_loss = []
            test_acc = []
            lr_log = []
            logs = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": eval_loss,
                "valid_acc": valid_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "lr_log": lr_log,
            }
            print("File does not exist. Created a new log.")

        self.Training = DoTraining(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            earlystopper=earlystopper,
            device="cuda",
            logs=logs,
            file_path=file_name,
        )

    pass

    def getTraining(self):
        print("now lr:", self.Training.optimizer.param_groups[0]["lr"])
        print("earlystop counter:", self.Training.earlystopper.early_stop_counter)
        print("bad epoch counter:", self.Training.scheduler.num_bad_epochs)
        print("scheduler parience:", self.Training.scheduler.patience)
        print("scheduler cooldown counter:", self.Training.scheduler.cooldown_counter)

        return self.Training
