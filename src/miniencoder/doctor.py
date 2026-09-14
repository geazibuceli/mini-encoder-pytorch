import argparse
import platform

import torch


def main():
    argparse.ArgumentParser(description="Check the local MiniEncoder environment.").parse_args()
    print(
        f"Python platform: {platform.platform()}\nPyTorch: {torch.__version__}\nCUDA available: {torch.cuda.is_available()}"
    )
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
        values = torch.ones(8, device="cuda", requires_grad=True)
        values.square().sum().backward()
        torch.cuda.synchronize()
        print("CUDA forward/backward check: passed")
    elif torch.version.cuda is None:
        print("This PyTorch installation is CPU-only. Install a CUDA build to use an NVIDIA GPU.")


if __name__ == "__main__":
    main()
