@echo off
echo Starting Pineapple Detection Backend...
echo.
echo Make sure you are in the PineappleServer directory
echo.
python -m uvicorn app:app --host 192.168.0.128 --port 8000 --reload  #change the host based on your ipv4
pause
