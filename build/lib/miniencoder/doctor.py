import argparse
import platform

import torch


def main():
    argparse.ArgumentParser(description="Check the local MiniEncoder environment.").parse_args()
    print(f"Python platform: {platform.platform()}\nPyTorch: {torch.__version__}\nCUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__": main()
