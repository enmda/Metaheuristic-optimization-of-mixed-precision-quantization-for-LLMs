"""Sanity check: libraries, CUDA, and free GPU memory."""
import torch
import transformers
import datasets

print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"datasets     {datasets.__version__}")
print(f"CUDA ok      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU          {torch.cuda.get_device_name(0)}")
    print(f"VRAM free    {free / 2**30:.2f} / {total / 2**30:.2f} GiB")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    print(f"fp16 matmul  ok ({(x @ x).shape})")
