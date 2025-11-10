# src/utils.py
import math, random, numpy as np, torch

def require_cuda():
    if not torch.cuda.is_available():
        raise SystemError("❌ GPU not detected. CPU is required by spec.")
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def loss_to_ppl(loss: float) -> float:
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")
