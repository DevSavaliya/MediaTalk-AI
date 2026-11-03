"""Bounded media preparation for untrusted public-demo inputs."""

import json
import math
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_UPLOAD_MB = 200
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_SECONDS = 3600
AUDIO_CHUNK_SECONDS = 300
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
        raise PublicDemoError(f"Choose a non-empty recording no longer than {MAX_SECONDS // 60} minutes.")
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
        if time.monotonic() - started > 300:
            raise PublicDemoError("This download is taking too long. Please upload a file instead.")
        if max(progress.get("downloaded_bytes") or 0, progress.get("total_bytes") or 0) > MAX_BYTES:
            raise PublicDemoError(f"The downloaded media exceeds the {MAX_UPLOAD_MB} MB demo limit.")

    class QuietLogger:
        # yt-dlp messages may contain visitor URLs; do not send them to host logs.
        def debug(self, message): pass
        def warning(self, message): pass
        def error(self, message): pass

    options = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]",
        "outtmpl": str(Path(folder) / "source.%(ext)s"),
        "noplaylist": True, "quiet": True, "logger": QuietLogger(),
        "socket_timeout": 15, "retries": 0, "fragment_retries": 0,
        "extractor_retries": 0, "max_filesize": MAX_BYTES,
        "progress_hooks": [check_size],
        "cachedir": False,
    }

    def validate_video(info, *, incomplete=False):
        if not incomplete:
            if info.get("_type") in {"playlist", "multi_video"} or info.get("is_live"):
                raise PublicDemoError("Live streams and playlists are not supported in this demo.")
            _duration(info.get("duration"))
        return None

    options["match_filter"] = validate_video
    canonical = youtube_url(source)
    # A single alternative public playback transport, not an unbounded retry loop.
    # yt-dlp documents Safari HLS as distinct from the token-gated HTTPS formats.
    profiles = [{}, {
        "extractor_args": {"youtube": {"player_client": ["web_safari"]}},
        "format": "bestaudio[protocol^=m3u8]/worst[protocol^=m3u8][acodec!=none]",
    }]
    first_failure = None
    for attempt, profile in enumerate(profiles):
        try:
            # Isolate partial files so a failed attempt cannot be mistaken for success.
            with tempfile.TemporaryDirectory(prefix="youtube-", dir=folder) as attempt_folder:
                settings = {**options, **profile, "outtmpl": str(Path(attempt_folder) / "source.%(ext)s")}
                with yt_dlp.YoutubeDL(settings) as downloader:
                    info = downloader.extract_info(canonical, download=True)
                if not info:
                    raise PublicDemoError("YouTube returned no recording (YT_DOWNLOAD). Try a file upload.")
                candidates = [p for p in Path(attempt_folder).glob("source.*")
                              if p.suffix in {".m4a", ".webm", ".mp4"}]
                if len(candidates) != 1 or not 0 < candidates[0].stat().st_size <= MAX_BYTES:
                    raise PublicDemoError(f"The video could not be downloaded within the {MAX_UPLOAD_MB} MB limit. Try a file upload.")
                target = Path(folder) / candidates[0].name
                candidates[0].replace(target)
                return target
        except PublicDemoError:
            raise
        except yt_dlp.utils.DownloadError as exc:
            message = _youtube_download_error(exc)
            if attempt == 0 and any(code in message for code in ("YT_FORBIDDEN", "YT_FORMAT")):
                first_failure = message
                continue
            if first_failure and "YT_FORMAT" in message:
                message = first_failure + " Alternative playback was also unavailable."
            raise PublicDemoError(message) from None
        except Exception:
            raise PublicDemoError("YouTube could not provide this video (YT_DOWNLOAD). Try uploading a supported file instead.") from None


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
