# audio and video helper functions for extracting audio from video files using FFmpeg
from __future__ import annotations

# standard library imports
import subprocess
import uuid
from pathlib import Path

import imageio_ffmpeg

# get a usable FFmpeg executable path, using imageio-ffmpeg to avoid Windows PATH issues
def get_ffmpeg_executable() -> str:

    return imageio_ffmpeg.get_ffmpeg_exe()

# extract mono 16kHz WAV audio from a video file using FFmpeg
def extract_audio_from_video(video_path: Path, output_dir: Path) -> Path:

    # create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"video_audio_{uuid.uuid4().hex}.wav"
    ffmpeg = get_ffmpeg_executable()
    # run the FFmpeg command to extract audio
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]

    # run the command and capture output, raising an error if it fails
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
