import torch_rocm_win as torch_rocm
torch = torch_rocm.torch

if torch_rocm.zluda_device:
    print("F")
print("E")