import torch, sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

from src.LogViewer import LogViewer


class logs:
    def __init__(self, file_name):
        self.file_name = file_name
        self.device = "cuda"
        checkpoint = torch.load(
            self.file_name + ".pth.tar",
            map_location=lambda storage, loc: storage.cuda(self.device),
        )

        self.logs = checkpoint["logs"]

        print("Suceessfully loaded the All setting and Log file.")
        print(self.file_name)


#     def __call__(self):
#         for i in range(len(self.logs["train_loss"])):
#             print(
#                 i + 1,
#                 " | train_loss",
#                 self.logs["train_loss"][i],
#                 ", train_acc ",
#                 self.logs["train_acc"][i],
#                 "test_loss ",
#                 self.logs["test_loss"][i],
#                 "test_acc",
#                 self.logs["test_acc"][i],
#             )


file_name = "MyResNet32_128_SGD_MultiStepLR_myresnet"

dja = logs(file_name=file_name)


__log = LogViewer(dja.logs)

__log.print_all()
