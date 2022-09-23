import json

model_file = "Injector_Surrogate_NN_PyTorch.pth"
model_info = json.load(open("configs/model_info.json"))
pv_info = json.load(open("configs/pv_info.json"))

import torch
torch_model = torch.jit.load("model_scripted.pt")
