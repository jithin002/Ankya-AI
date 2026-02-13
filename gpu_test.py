# gpu_test.py
import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    # small allocation test
    x = torch.rand(512,512).to('cuda')
    print("Allocated tensor on GPU, shape:", x.shape)
    del x
else:
    print("CUDA not available - running on CPU.")
