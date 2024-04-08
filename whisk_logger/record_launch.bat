set PYTHON_SCRIPT_PATH=C:\Users\blake\Documents\GitHub\ebakery\src\whisk_logger.py

REM Get current date and time and format as YYYY-MM-DD_hhmm
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set datetime=%%a
set datetime=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%_%datetime:~8,2%%datetime:~10,2%
set OUTPUT_VIDEO_PATH=C:\Users\blake\Videos\Captures\%datetime%.mp4

echo Starting video recording and Python script...
start /b cmd /c ffmpeg -f dshow -rtbufsize 100M -i video="Opal Tadpole" -r 25 -s 640x480 -vcodec libx264 "%OUTPUT_VIDEO_PATH%"
start /b python "%PYTHON_SCRIPT_PATH%"

echo Recording and Python script have started.
echo Press Ctrl+C in the recording command window to stop recording.
