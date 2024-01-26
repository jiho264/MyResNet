# %% [markdown]
# # Loading Trained Model

# %%
import torch
from src.LogViewer import LogViewer
from src.Earlystopper import EarlyStopper
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.Mymodel import MyResNet34
from src.Mymodel import MyResNet_CIFAR

# %%
"""Dataset selection"""
# DATASET = "CIFAR10"
# DATASET = "CIFAR100"
DATASET = "ImageNet2012"

"""Model selection for CIFAR"""
NUM_LAYERS_LEVEL = 5

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


file_path = ""
if DATASET == "ImageNet2012":
    file_path = f"MyResNet34_{BATCH}_{OPTIMIZER}"
    _model_name = f"MyResNet34_{DATASET}_{BATCH}_{OPTIMIZER}"
else:
    file_path = f"MyResNet{NUM_LAYERS_LEVEL*6+2}_{BATCH}_{OPTIMIZER}"
    _model_name = f"MyResNet{NUM_LAYERS_LEVEL*6+2}_{DATASET}_{BATCH}_{OPTIMIZER}"

if SPLIT_RATIO != 0:
    _model_name += f"_{int(SPLIT_RATIO*100)}"
    file_path += f"_{int(SPLIT_RATIO*100)}"

# %%
checkpoint = torch.load(
    "models/" + _model_name + "/" + file_path + ".pth.tar",
    map_location=lambda storage, loc: storage.cuda("cuda"),
)

logs = checkpoint["logs"]

print("Suceessfully loaded the All setting and Log file.")

# %%
if DATASET == "CIFAR10" or DATASET == "CIFAR100":
    """ResNet{20, 32, 44, 56, 110, 1202} for CIFAR"""
    model = MyResNet_CIFAR(
        num_classes=10,
        num_layer_factor=NUM_LAYERS_LEVEL,
        Downsample_option="A",
    ).to("cuda")
    print(f"ResNet-{5*6+2} for {DATASET} is loaded.")

elif DATASET == "ImageNet2012":
    """ResNet34 for ImageNet 2012"""
    model = MyResNet34(num_classes=1000, Downsample_option="B").to("cuda")
    # model = models.resnet34(pretrained=True).to(device)
    # model = models.resnet34(pretrained=False).to(device)
    print(f"ResNet-34 for {DATASET} is loaded.")

if OPTIMIZER == "Adam":
    optimizer = torch.optim.Adam(model.parameters())
elif OPTIMIZER == "Adam_decay":
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
elif OPTIMIZER == "SGD":
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001
    )

# %% [markdown]
# ## (3) Define Early Stopping

# %%
earlystopper = EarlyStopper(patience=777, model=model, file_name=file_path)

# %% [markdown]
# ## (4) Define Learning Rate schedualer

# %%
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=777,
    factor=0.1,
    verbose=True,
    threshold=1e-4,
    cooldown=777,
)

optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
earlystopper.load_state_dict(checkpoint["earlystopper"])

# %%
print("now lr:", optimizer.param_groups[0]["lr"])
print("earlystop counter:", earlystopper.early_stop_counter)
print("bad epoch counter:", scheduler.num_bad_epochs)
print("scheduler parience:", scheduler.patience)
print("scheduler cooldown counter:", scheduler.cooldown_counter)


# %%


# %%
viewer = LogViewer(logs)
# viewer.draw(start=4980, range=200)
# viewer.draw()

# %%
viewer.print_len()

# %%
viewer.print_all()

# %% [markdown]
# - 주피터노트북 output set
# - @tag:notebookOutputLayout
