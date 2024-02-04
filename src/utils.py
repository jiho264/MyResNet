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

# from MyImageNetdataloader import LoadDataset, MyShortCut
from src.CumstomCosineAnnealingwarmRestarts import CosineAnnealingWarmUpRestarts
from src.Mymodel import MyResNet_CIFAR, MyResNet34
from src.Earlystopper import EarlyStopper


class Single_model:
    def __init__(
        self,
        dataset,
        batch_size,
        optimizer_name,
        schduler_name,
        device="cuda",
        **kwargs,
    ) -> None:
        if dataset == "ImageNet2012":
            self.file_name = f"MyResNet34_{batch_size}_{optimizer_name}_{schduler_name}"
            self.model = MyResNet34(num_classes=1000, Downsample_option="B").to(device)
        elif dataset == "CIFAR10":
            self.file_name = f"MyResNet32_{batch_size}_{optimizer_name}_{schduler_name}"
            self.model = MyResNet_CIFAR(num_classes=10, num_layer_factor=5).to(device)

        self.optimizer_name = optimizer_name
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
            patience=kwargs["kwargs"]["Earlystopping_patiance"],
            model=self.model,
            file_name=self.file_name,
        )

        """define learning rate scheduler"""
        if schduler_name == "ExponentialLR":
            self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        elif schduler_name == "MultiStepLR":
            self.scheduler = MultiStepLR(self.optimizer, milestones=[30, 60], gamma=0.1)
        elif schduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                patience=kwargs["kwargs"]["ReduceLROnPlateau_patiance"],
                factor=0.1,
                cooldown=kwargs["kwargs"]["ReduceLROnPlateau_cooldown"],
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
            self.scheduler = ConstantLR(self.optimizer, factor=1, total_iters=999)
            pass

        """define scaler"""
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)


class SingleModelTrainingProcess(Single_model):
    def __init__(
        self,
        dataset,
        batch_size,
        optimizer_name,
        schduler_name,
        device="cuda",
        train_dataloader=None,
        valid_dataloader=None,
        test_dataloader=None,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size,
            optimizer_name,
            schduler_name,
            device="cuda",
            kwargs=kwargs,
        )
        # self.earlystopper.end_flag = False
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

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
        self.train_loss = self.running_loss / len(self.train_dataloader)
        self.train_acc = self.running_corrects / self.running_total
        if self.valid_dataloader != None:
            self.valid_loss = self.valid_loss / len(self.valid_dataloader)
            self.valid_acc = self.valid_corrects / self.valid_total
        if self.test_dataloader != None:
            self.test_loss = self.test_loss / len(self.test_dataloader)
            self.test_acc = self.test_corrects / self.test_total

        self.logs["train_loss"].append(self.train_loss)
        self.logs["train_acc"].append(self.train_acc)
        self.logs["valid_loss"].append(self.valid_loss)
        self.logs["valid_acc"].append(self.valid_acc)
        self.logs["test_loss"].append(self.test_loss)
        self.logs["test_acc"].append(self.test_acc)
        self.logs["lr_log"].append(self.optimizer.param_groups[0]["lr"])

    def save_model(self):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "earlystopper": self.earlystopper.state_dict(),
            "logs": self.logs,
        }
        torch.save(checkpoint, self.file_name + ".pth.tar")

    def print_info(self, now_epochs, num_epochs, print_pad_optim, print_pad_scheduler):
        _epoch_print = f"{now_epochs}/{num_epochs} | {self.optimizer_name.ljust(print_pad_optim)} - {self.scheduler_name.ljust(print_pad_scheduler)}"
        _epoch_print += (
            " | " + f"Train : {self.train_loss:.4f} / {self.train_acc*100:.2f}%"
        )
        if self.valid_dataloader != None:
            _valid_print = f"Valid : {self.valid_loss:.4f} / {self.valid_acc*100:.2f}%"
            _epoch_print += " | " + _valid_print
        if self.test_dataloader != None:
            _test_print = f"Test : {self.test_loss:.4f} / {self.test_acc*100:.2f}%"
            _epoch_print += " | " + _test_print
        print(_epoch_print)

    def scheduling(self):
        if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
            if self.valid_dataloader != None:
                # valid ok , train non
                self.scheduler.step(self.valid_loss)
                # print("reduce check", self.valid_loss)
            else:
                # valid non이라면 무조건 train loss로 스케줄링
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

    def select_earlystopping_loss_and_check(self):
        if self.valid_dataloader != None:
            return self.earlystopper.check(self.valid_loss)
        elif self.valid_dataloader == None:
            return self.earlystopper.check(self.train_loss)
        else:
            raise NotImplementedError

    def is_completed(self):
        return self.earlystopper.end_flag

    def forward_train(self, images, labels):
        self.model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            images, labels = images.to(self.device), labels.to(self.device)
            """preprocessing"""
            # _MyshortCut.preprocessing_train(images)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.running_loss += loss.item()
        _, predicted = outputs.max(1)
        self.running_total += labels.size(0)
        self.running_corrects += predicted.eq(labels).sum().item()

    def forward_eval(self, images, labels, mode):
        self.model.eval()
        with torch.no_grad():
            images, labels = images.to(self.device), labels.to(self.device)
            """preprocessing"""
            # _MyshortCut.preprocessing_valid(images)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            if mode == "valid":
                self.valid_loss += loss.item()
                _, predicted = outputs.max(1)
                self.valid_total += labels.size(0)
                self.valid_corrects += predicted.eq(labels).sum().item()
            elif mode == "test":
                self.test_loss += loss.item()
                _, predicted = outputs.max(1)
                self.test_total += labels.size(0)
                self.test_corrects += predicted.eq(labels).sum().item()
            else:
                raise NotImplementedError
