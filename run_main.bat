@echo off
setlocal

set "input_folders=input/z_Others input/CodeAesthetic"
set "output_folders=output/z_Others output/CodeAesthetic"

python main.py --input_folders %input_folders% --output_folders %output_folders% --diarize

endlocal
pause
