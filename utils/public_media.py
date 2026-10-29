"""Bounded media preparation for untrusted public-demo inputs."""

import json
import math
import re
import subprocess
import time
import wave
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_UPLOAD_MB = 50
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_SECONDS = 600
FORMATS = "mov,matroska,webm,avi,mp3,wav,ogg"


class PublicDemoError(ValueError):
    """A deliberately safe message that can be displayed to visitors."""


def youtube_url(source):
    """Accept one video ID and discard other URL components, including playlists."""
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise PublicDemoError("Enter a YouTube video link, not a local file or playlist.")
    if parsed.netloc.lower() not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}:
        raise PublicDemoError("Enter a YouTube video link.")
    parts = parsed.path.strip("/").split("/")
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parts[0] if len(parts) == 1 else ""
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
        video_id = parts[1]
    else:
        video_id = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise PublicDemoError("Enter a link to one YouTube video.")
    return f"https://www.youtube.com/watch?v={video_id}"


def _duration(value):
    try:
        duration = float(value)
    except (TypeError, ValueError):
        raise PublicDemoError("The recording's duration could not be determined.") from None
    if not math.isfinite(duration) or duration <= 0 or duration > MAX_SECONDS:
        raise PublicDemoError("Choose a non-empty recording no longer than 10 minutes.")
    return duration


def _youtube_download_error(error):
    """Classify known downloader failures without exposing URLs or raw responses."""
    message = str(error).lower()
    if any(marker in message for marker in ("confirm you're not a bot", "confirm you’re not a bot", "http error 429")):
        return "YouTube blocked automated requests from this server (YT_ACCESS). Please upload a file instead."
    if any(marker in message for marker in ("private video", "video unavailable", "not available in your country", "confirm your age", "members-only")):
        return "YouTube reports this video is unavailable or requires account access (YT_RESTRICTED). Try an unrestricted video or upload a file."
    if "requested format is not available" in message:
        return "YouTube did not expose a supported audio format (YT_FORMAT). Try uploading a file; the server's downloader support may need updating."
    if "http error 403" in message:
        return "YouTube refused the download request (YT_FORBIDDEN). Please upload a file instead."
    if any(marker in message for marker in ("timed out", "failed to resolve", "temporary failure in name resolution", "unable to connect")):
        return "The server could not connect to YouTube (YT_NETWORK). Try again later or upload a file."
    return "YouTube could not provide this video (YT_DOWNLOAD). Try uploading a supported file instead."


def download_youtube(source, folder):
    import yt_dlp
    started = time.monotonic()

    def check_size(progress):
        if time.monotonic() - started > 180:
            raise PublicDemoError("This download is taking too long. Please upload a file instead.")
        if max(progress.get("downloaded_bytes") or 0, progress.get("total_bytes") or 0) > MAX_BYTES:
            raise PublicDemoError("The downloaded audio exceeds the 50 MB demo limit.")

    class QuietLogger:
        # yt-dlp messages may contain visitor URLs; do not send them to host logs.
        def debug(self, message): pass
        def warning(self, message): pass
        def error(self, message): pass

    options = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]",
        "outtmpl": str(Path(folder) / "source.%(ext)s"),
        "noplaylist": True, "quiet": True, "logger": QuietLogger(),
        "socket_timeout": 15, "retries": 1, "fragment_retries": 1,
        "extractor_retries": 1, "max_filesize": MAX_BYTES,
        "progress_hooks": [check_size],
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(youtube_url(source), download=False)
            if not info or info.get("_type") in {"playlist", "multi_video"} or info.get("is_live"):
                raise PublicDemoError("Live streams and playlists are not supported in this demo.")
            _duration(info.get("duration"))
            downloader.process_info(info)
        candidates = [p for p in Path(folder).glob("source.*") if p.suffix in {".m4a", ".webm"}]
        if len(candidates) != 1 or not 0 < candidates[0].stat().st_size <= MAX_BYTES:
            raise PublicDemoError("The video could not be downloaded within the 50 MB limit. Try a file upload.")
        return candidates[0]
    except PublicDemoError:
        raise
    except yt_dlp.utils.DownloadError as exc:
        raise PublicDemoError(_youtube_download_error(exc)) from None
    except Exception:
        raise PublicDemoError("YouTube could not provide this video (YT_DOWNLOAD). Try uploading a supported file instead.") from None


def prepare_audio(source, folder):
    """Probe first; decode at most 601 seconds and reject oversize media, never truncate silently."""
    source = Path(source)
    if not 0 < source.stat().st_size <= MAX_BYTES:
        raise PublicDemoError("Choose a non-empty file of at most 50 MB.")
    output = Path(folder) / "audio.wav"
    restrictions = ["-protocol_whitelist", "file,pipe", "-format_whitelist", FORMATS]
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", *restrictions, "-show_entries", "format=duration",
             "-of", "json", str(source)], capture_output=True, text=True, timeout=30, check=True)
        _duration(json.loads(probe.stdout).get("format", {}).get("duration"))
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", *restrictions, "-i", str(source),
             "-map", "0:a:0", "-vn", "-t", str(MAX_SECONDS + 1), "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(output)], capture_output=True, timeout=120, check=True)
        with wave.open(str(output), "rb") as audio:
            _duration(audio.getnframes() / audio.getframerate())
    except PublicDemoError:
        raise
    except FileNotFoundError:
        raise PublicDemoError("Media conversion is unavailable on this server. Please contact the demo owner.") from None
    except (subprocess.SubprocessError, OSError, ValueError, wave.Error):
        raise PublicDemoError("This file could not be processed within the demo limits. Try a shorter MP4, MP3, or WAV file.") from None
    return [str(output)]
