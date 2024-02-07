import torch

print("torch version:", torch.__version__)
print("cudnn version:", format(torch.backends.cudnn.version()))
print("cuda version:", format(torch.version.cuda))
