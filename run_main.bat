@echo off
setlocal enabledelayedexpansion

set "_folders=3b1b ariseffai Be_Smart braintruffle CodeAesthetic domainofscience Fireship Koranos Kurzgsaget PrimerBlobs QuantaScienceChannel SamuelAlbanie1 ScienceClicEN TED_Ed TwoMinutePapers yoshtm z_Others"

@REM set "_folders=z_Others"
set "_vocal_only_folders=ariseffai domainofscience SamuelAlbanie1 TwoMinutePapers"
set "input_folders="
set "output_folders="
set "vocal_only_folders="

for %%i in (%_folders%) do (
    set "input_folders=!input_folders! input/%%i"
    set "output_folders=!output_folders! output/%%i"
)

for %%i in (%_vocal_only_folders%) do (
    set "vocal_only_folders=!vocal_only_folders! input/%%i"
)

echo !input_folders!
echo !output_folders!
echo !vocal_only_folders!

python main.py --input_folders !input_folders! --output_folders !output_folders! --vocal_only_folders !vocal_only_folders! --diarize

endlocal
