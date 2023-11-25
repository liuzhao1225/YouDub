import json
from moviepy.editor import VideoFileClip, AudioFileClip
import os
import subprocess
def format_timestamp(seconds):
    """Converts seconds to the SRT time format."""
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"


def convert_json_to_srt(json_file, srt_file, max_line_char=30):
    print(f'Converting {json_file} to {srt_file}...')
    with open(json_file, 'r', encoding='utf-8') as f:
        subtitles = json.load(f)

    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, subtitle in enumerate(subtitles, 1):
            start = format_timestamp(subtitle['start'])
            end = format_timestamp(subtitle['end'])
            text = subtitle['text']
            line = len(text)//(max_line_char+1) + 1
            avg = min(round(len(text)/line), max_line_char)
            text = '\n'.join([text[i*avg:(i+1)*avg]
                             for i in range(line)])

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")


def replace_audio_ffmpeg(input_video: str, input_audio: str, input_subtitles: str, output_path: str, fps=30) -> None:
    # Extract the video name from the input video path
    video_name = os.path.basename(input_video)

    # Replace video file extension with '.srt' for subtitles
    srt_name = video_name.replace('.mp4', '.srt').replace(
        '.mkv', '.srt').replace('.avi', '.srt').replace('.flv', '.srt')

    # Construct the path for the subtitles file
    srt_path = os.path.join(os.path.dirname(input_audio), srt_name)

    # Convert subtitles from JSON to SRT format
    convert_json_to_srt(input_subtitles, srt_path)

    # Determine the output folder and define a temporary file path
    output_folder = os.path.dirname(output_path)
    tmp = os.path.join(output_folder, 'tmp.mp4')

    # Prepare a list to hold FFmpeg commands
    commands = []

    # FFmpeg command to replace audio, add subtitles, and set output frame rate
    commands.append(f'ffmpeg -i "{input_video}" -i "{input_audio}" -vf "subtitles={srt_path}:force_style=\'FontName=Arial,FontSize=20,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,WrapStyle=2\'" -c:v libx264 -r {fps} -c:a aac -map 0:v:0 -map 1:a:0 "{tmp}" -y'.replace('\\', '/'))

    # FFmpeg command to speed up the video by 1.05 times
    commands.append(
        f'ffmpeg -i "{tmp}" -vf "setpts=0.94999999999*PTS" -af "atempo=1.05263157895" -c:v libx264 -c:a aac "{output_path}" -y'.replace('\\', '/'))

    # Command to delete the temporary file
    commands.append(f'del "{tmp}"')

    # Add an 'exit' command to close the command prompt window after execution
    commands.append('exit')

    # Join the commands with '&&' to ensure sequential execution
    command = ' && '.join(commands)

    # Execute the combined FFmpeg command
    print(command)
    subprocess.Popen(command, shell=True)
    
def replace_audio(video_path: str, audio_path: str, subtitle_path: str, output_path: str, fontsize=64, font='SimHei', color='white') -> None:
    """Replace the audio of the video file with the provided audio file.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to the audio file to replace the original audio.
        output_path (str): Path to save the output video file.
    """
    video_name = os.path.basename(video_path)
    srt_name = video_name.replace('.mp4', '.srt').replace(
        '.mkv', '.srt').replace('.avi', '.srt').replace('.flv', '.srt')
    srt_path = os.path.join(os.path.dirname(audio_path), srt_name)
    convert_json_to_srt(subtitle_path, srt_path)

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    new_video = video.set_audio(audio)

    new_video.write_videofile(output_path, codec='libx264', threads=16, fps=30)

if __name__ == '__main__':
    # file_name = r"This Virus Shouldn't Exist (But it Does)"
    file_name = "Kurzgesagt Channel Trailer"
    input_folder = 'test'
    output_folder = os.path.join('output', file_name)
    input_video = os.path.join(input_folder, file_name + '.mp4')
    input_audio = os.path.join(output_folder, 'zh.wav')
    input_subtitles = os.path.join(output_folder, 'zh.json')
    srt_path = os.path.join(output_folder, file_name+'.srt')
    output_path = os.path.join(output_folder, file_name + '.mp4')
    replace_audio_ffmpeg(input_video, input_audio,
                         input_subtitles, output_path)
