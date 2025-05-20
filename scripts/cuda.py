import torch
import platform
import subprocess
from ultralytics import YOLO


def print_section(title):
    print("\n" + "="*30)
    print(title)
    print("="*30)

# 1. OS and Python Info
print_section("System Info")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python version: {platform.python_version()}")

# 2. PyTorch and CUDA
print_section("PyTorch & CUDA Info")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version (from PyTorch): {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("❌ CUDA not available. Check your PyTorch install or NVIDIA drivers.")

# 3. Check nvidia-smi (if available)
print_section("nvidia-smi Output")
try:
    output = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    print(output)
except FileNotFoundError:
    print("❌ 'nvidia-smi' not found. Make sure NVIDIA drivers are installed and in PATH.")
except subprocess.CalledProcessError as e:
    print(f"Error running nvidia-smi: {e}")