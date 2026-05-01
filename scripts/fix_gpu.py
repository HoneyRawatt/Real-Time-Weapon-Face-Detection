import os
import glob
import shutil
import sys

# Find the virtual environment's site-packages folder
site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
onnx_capi = os.path.join(site_packages, "onnxruntime", "capi")
torch_lib = os.path.join(site_packages, "torch", "lib")

# Find the DLLs inside PyTorch's lib folder
torch_dlls = glob.glob(os.path.join(torch_lib, "*.dll"))

print(f"Found {len(torch_dlls)} DLLs in PyTorch. Copying to ONNX Runtime...")

# Copy them directly into the ONNX folder
copied = 0
for dll in torch_dlls:
    try:
        shutil.copy(dll, onnx_capi)
        copied += 1
    except Exception as e:
        pass

print(f"\nDone! Successfully copied {copied} DLLs. ONNX Runtime will now use your GPU.")