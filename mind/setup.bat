@echo off
python -m venv env
call env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo.
echo 🔧 MagicBox environment is set. Run with: python main.py
pause

