import platform
import torch

CURRENT_OS = platform.system()
NUM_CUDA_DEVICES = torch.cuda.device_count()

def set_device():
    device = torch.device("cpu")

    if CURRENT_OS == "Linux":
        if 1 < NUM_CUDA_DEVICES:
            torch.cuda.set_device(3)
            device = torch.device("cuda:3")
        elif NUM_CUDA_DEVICES == 1:
            device = torch.device("cuda")

    elif CURRENT_OS == "Windows":
        if 1 < NUM_CUDA_DEVICES:
            torch.cuda.set_device(3)
            device = torch.device("cuda:3")
        elif NUM_CUDA_DEVICES == 1:
            device = torch.device("cuda")

    elif CURRENT_OS == "Darwin":
        if torch.backends.mps.is_available():
            device = torch.device("mps")

    else:
        print(f"Unsupported operating system: {CURRENT_OS}")

    return device