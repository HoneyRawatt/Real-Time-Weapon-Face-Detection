# Weapon Detection — Setup & Run

Quick steps to create a compatible environment and run the project on Windows.

Prerequisites
- Python 3.11 installed and available via the `py` launcher (recommended).
- Git (optional) and a camera connected for realtime detection.

1) Create Python 3.11 venv

PowerShell (recommended):
```
Set-Location 'd:\coding\.vscode\Weapon_Detection-main'
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

2) Install CPU dependencies

This repository intentionally installs CPU builds by default to avoid CUDA mismatches. If you have CUDA and want GPU acceleration, see the "GPU Wheels" section below.

```
# Install CPU PyTorch + torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --upgrade

# Install the rest from requirements.txt
pip install -r requirements.txt
```

3) Configure email credentials securely

- Copy `.env.example` to `.env` and fill values, or set the environment variables in PowerShell:

```
$env:ALERT_SENDER_EMAIL = 'youremail@example.com'
$env:ALERT_EMAIL_PASSWORD = 'your_app_password'
$env:ALERT_TO_EMAIL = 'recipient@example.com'
```

Important: For Gmail, create an "app password" or enable a secure sending method — do not store real account passwords in code.

4) Run the web UI (Flask)

```
python app.py
# The app runs on http://127.0.0.1:5000 by default
```

5) Run the realtime detector (CLI)

This opens a camera window and runs detection in a loop.

```
python main.py
# Press 'q' in the camera window to quit
```

GPU Wheels (optional)
- If you have an NVIDIA GPU and a specific CUDA toolkit installed, install matching PyTorch wheels from https://pytorch.org/get-started/locally/ (select your CUDA version) instead of the CPU wheels above.
- For TensorFlow GPU, install the appropriate `tensorflow` package that matches your GPU + drivers (careful: TensorFlow GPU support can be version-sensitive).

Notes & Security
- `email_sender.py` now reads `ALERT_SENDER_EMAIL` and `ALERT_EMAIL_PASSWORD` from the environment. Do not commit `.env` (it's ignored by `.gitignore`).
- You may need to download or point to model files included in the repo (e.g., `best100.pt`, `yolov5su.pt`). Keep them in the project root or update model paths in code.
- If you want me to switch the environment to GPU wheels, tell me your CUDA version and I'll provide the exact `pip` command (or I can switch the venv for you).

Troubleshooting
- If `pip install -r requirements.txt` fails due to Python version incompatibilities, verify you're using Python 3.11.
- If DeepFace/retinaface complains about `tf-keras`, install `tf-keras` in the same environment.

License / Disclaimer
- This repo is provided as-is. Secure credentials and verify legal/privacy considerations before using in production.
# Weapon_Detection