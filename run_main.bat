@echo off
setlocal enabledelayedexpansion

set "_folders=HistoryOfTheUniverse AI_Explained 3b1b ariseffai Be_Smart braintruffle CodeAesthetic domainofscience Fireship Koranos Kurzgsaget PrimerBlobs QuantaScienceChannel SamuelAlbanie1 ScienceClicEN TED_Ed TwoMinutePapers yoshtm z_Others"

set "_folders=z_Others"
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
