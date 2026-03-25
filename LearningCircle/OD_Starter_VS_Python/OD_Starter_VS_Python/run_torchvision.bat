
@echo off
setlocal
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision pillow tqdm
python src\train_torchvision.py
pause
