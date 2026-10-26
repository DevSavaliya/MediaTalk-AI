import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import requests
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_ENGINE = os.getenv("WHISPER_ENGINE", "faster")
CACHE_DIR = Path(__file__).resolve().parents[1] / "transcript_cache"


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_model = None


def load_model():

    global _model  

    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ({WHISPER_ENGINE}) ...")
        if WHISPER_ENGINE == "faster":
            from faster_whisper import WhisperModel
            # CPU INT8 keeps the same model size while reducing inference cost.
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                                  cpu_threads=4, num_workers=1)
        elif WHISPER_ENGINE == "openai":
            import whisper
            _model = whisper.load_model(WHISPER_MODEL)
        else:
            raise ValueError("WHISPER_ENGINE must be 'faster' or 'openai'.")
        print("Whisper model loaded.")
    return _model 


def transcribe_chunk_whisper(chunk_path: str, on_progress=None) -> str:

    model = load_model()  

    if WHISPER_ENGINE == "openai":
        result = model.transcribe(chunk_path, task="transcribe", language="en",
                                  fp16=str(model.device).startswith("cuda"))
        return result["text"]
    segments, info = model.transcribe(chunk_path, task="transcribe", language="en",
                                      beam_size=1, vad_filter=True)
    parts = []
    for segment in segments:
        parts.append(segment.text.strip())
        if on_progress:
            on_progress(min(segment.end / max(info.duration, 0.001), 1.0))
    return " ".join(parts)


def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str, on_progress=None) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav").close()

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
            if on_progress:
                on_progress((i + 1) / total_pieces)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()

   



def transcribe_chunk(chunk_path: str, language: str = "english", on_progress=None) -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path, on_progress)
    return transcribe_chunk_whisper(chunk_path, on_progress)


def _cache_path(chunk, language):
    # Content-based: renaming/re-uploading a file is safe; editing it invalidates it.
    # Bump the version when decoding settings change. Never include credentials.
    settings = ("v1", language.lower(), SARVAM_MODEL if language.lower() == "hinglish"
                else (WHISPER_ENGINE, WHISPER_MODEL, "en-int8-greedy-vad"))
    digest = hashlib.sha256(json.dumps(settings).encode())
    with open(chunk, "rb") as audio:
        for block in iter(lambda: audio.read(1024 * 1024), b""):
            digest.update(block)
    return CACHE_DIR / f"{digest.hexdigest()}.json"


def _read_cache(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        text = data.get("transcript") if isinstance(data, dict) else None
        return text if isinstance(text, str) and text.strip() else None
    except (OSError, ValueError):
        return None


def _write_cache(path, text):
    if not text.strip():
        return
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump({"transcript": text}, output, ensure_ascii=False)
        temporary.replace(path)
    except OSError:
        # A read-only/full cache must never discard a successful transcription.
        print("Transcript cache unavailable; continuing without saving it.")
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def transcribe_all(chunks: list, language: str = "english", on_progress=None) -> str:

    full_transcript = []
    started = time.perf_counter()

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        def progress(fraction, cached=False):
            if on_progress:
                on_progress({"chunk": i + 1, "chunks": len(chunks),
                             "fraction": (i + fraction) / len(chunks),
                             "cached": cached, "elapsed": time.perf_counter() - started})

        progress(0)
        path = _cache_path(chunk, language) if os.getenv("TRANSCRIPTION_CACHE", "1") != "0" else None
        text = _read_cache(path) if path else None
        if text is None:
            text = transcribe_chunk(chunk, language=language, on_progress=progress)
            if path:
                _write_cache(path, text)
            progress(1)
        else:
            print("Reusing cached transcript.")
            progress(1, cached=True)
        full_transcript.append(text)

    print("Transcription complete.")

    return " ".join(full_transcript).strip()
