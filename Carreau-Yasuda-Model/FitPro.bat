set column=99
set /a lines=(%column%/5)*2
mode con: cols=%column% lines=%lines%
@ECHO OFF
SETLOCAL
set b=%~dp0
set bd=%b%ModelFitting.py

python "%bd%" %*

REMpause >nul
::PAUSE
timeout 1

