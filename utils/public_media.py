"""Bounded media preparation for untrusted public-demo inputs."""

import json
import math
import subprocess
import wave
from pathlib import Path

MAX_UPLOAD_MB = 200
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_SECONDS = 3600
AUDIO_CHUNK_SECONDS = 300
FORMATS = "mov,matroska,webm,avi,mp3,wav,ogg"


class PublicDemoError(ValueError):
    """A deliberately safe message that can be displayed to visitors."""


def _duration(value):
    try:
        duration = float(value)
    except (TypeError, ValueError):
        raise PublicDemoError("The recording's duration could not be determined.") from None
    if not math.isfinite(duration) or duration <= 0 or duration > MAX_SECONDS:
        raise PublicDemoError(f"Choose a non-empty recording no longer than {MAX_SECONDS // 60} minutes.")
    return duration


def prepare_audio(source, folder):
    """Probe first, decode in bounded chunks, and reject over-limit media without truncating."""
    source = Path(source)
    if not 0 < source.stat().st_size <= MAX_BYTES:
        raise PublicDemoError(f"Choose a non-empty file of at most {MAX_UPLOAD_MB} MB.")
    audio_folder = Path(folder) / "audio-chunks"
    audio_folder.mkdir(exist_ok=False)
    output = audio_folder / "audio-%03d.wav"
    restrictions = ["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS]
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", *restrictions, "-show_entries", "format=duration",
             "-of", "json", str(source)], capture_output=True, text=True, timeout=30, check=True)
        _duration(json.loads(probe.stdout).get("format", {}).get("duration"))
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", *restrictions, "-i", str(source),
             "-map", "0:a:0", "-vn", "-t", str(MAX_SECONDS + 1), "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", "-f", "segment", "-segment_time", str(AUDIO_CHUNK_SECONDS),
             "-reset_timestamps", "1", str(output)], capture_output=True, timeout=300, check=True)
        chunks = sorted(audio_folder.glob("audio-*.wav"))
        duration = 0.0
        for chunk in chunks:
            with wave.open(str(chunk), "rb") as audio:
                duration += audio.getnframes() / audio.getframerate()
        _duration(duration)
    except PublicDemoError:
        raise
    except FileNotFoundError:
        raise PublicDemoError("Media conversion is unavailable on this server. Please contact the demo owner.") from None
    except (subprocess.SubprocessError, OSError, ValueError, wave.Error):
        raise PublicDemoError("This file could not be processed within the demo limits. Try a shorter MP4, MP3, or WAV file.") from None
    return [str(chunk) for chunk in chunks]
