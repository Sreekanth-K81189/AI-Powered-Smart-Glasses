# Enabling CUDA (GPU) for this project

If `python check_cuda.py` shows **CUDA available: False**, PyTorch is using CPU only. The project still runs on CPU.

## Python 3.14 and CUDA

**Python 3.14:** Use the **cu126** index (see Enable CUDA below). Do not use cu124 for 3.14.

**Python 3.11/3.12:** For CUDA, use cu124 or cu126 (see Enable CUDA). PyTorch does not yet provide CUDA wheels for Python 3.14 on the official index, so `pip install ... --index-url .../cu124` can fail with “No matching distribution found”. To use **CUDA**, use **Python 3.11 or 3.12** and install PyTorch with CUDA from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/). With Python 3.14, keep using **CPU** PyTorch (install steps below).

## Restore PyTorch (CPU) if it was uninstalled

If torch is missing (e.g. after a failed CUDA install), reinstall CPU PyTorch:

```powershell
& "C:\Users\chekk\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip install torch torchvision
```

## Enable CUDA

1. **NVIDIA GPU** and **drivers**: [NVIDIA Drivers](https://www.nvidia.com/drivers).
2. Install PyTorch with CUDA:

**Python 3.14 (use cu126):**

```powershell
& "C:\Users\chekk\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip uninstall torch torchvision -y
& "C:\Users\chekk\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Python 3.11/3.12** can use `cu124` or `cu126`. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

## Verify

```powershell
python check_cuda.py
```

With CUDA installed, you should see `CUDA available: True`. Object_detection, OCR, Whisper, Skin_cancer_detection, etc. will then use the GPU automatically.
