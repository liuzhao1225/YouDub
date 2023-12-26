@echo off
setlocal enabledelayedexpansion

set "_folders="

FOR /D %%G IN ("F:\YouDub\input\*") DO (
    SET "_folders=!_folders! %%~nG"
)

@REM set "_folders=SamuelAlbanie1"

set "input_folders="
set "output_folders="

for %%i in (%_folders%) do (
    set "input_folders=!input_folders! input/%%i"
    set "output_folders=!output_folders! output/%%i"
)

echo !input_folders!
echo !output_folders!

python main.py --input_folders !input_folders! --output_folders !output_folders! --diarize

endlocal
