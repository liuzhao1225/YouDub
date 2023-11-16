import json
from moviepy.editor import VideoFileClip, AudioFileClip
import os
def format_timestamp(seconds):
    """Converts seconds to the SRT time format."""
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"


def convert_json_to_srt(json_file, srt_file):
    print(f'Converting {json_file} to {srt_file}...')
    with open(json_file, 'r', encoding='utf-8') as f:
        subtitles = json.load(f)

    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, subtitle in enumerate(subtitles, 1):
            start = format_timestamp(subtitle['start'])
            end = format_timestamp(subtitle['end'])
            text = subtitle['text']

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
 
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

    # size = video.size
    # size = (size[0], size[1] * 0.8)
    # def generator(txt): return TextClip(
    #     txt, fontsize=size[0]//20, font=font, color=color, method='caption', align='south', size=size, stroke_color='black', stroke_width=max(fontsize//20, 2))
    # subs = [((segment['start'], segment['end']), segment['text'])
    #         for segment in transcript]
    # subtitles = SubtitlesClip(subs, generator)
    # new_video = CompositeVideoClip(
    #     [new_video, subtitles.set_position(('center', 'bottom'))])

    new_video.write_videofile(output_path, codec='libx264', threads=16)
