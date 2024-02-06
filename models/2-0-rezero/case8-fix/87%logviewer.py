import torch
import sys, os, tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("src"))))

from src.LogViewer import LogViewer

file_name = "MyResNet32_128_SGD.pth.tar"


checkpoint = torch.load(file_name, map_location=lambda storage, loc: storage.cuda(0))
logs = checkpoint["logs"]

log_viewer = LogViewer(logs)
log_viewer.print_all()
