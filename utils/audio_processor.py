from pathlib import Path
from pydub import AudioSegment
import os


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav").close()
    return output_path


def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    # A short recording already is one chunk. Avoid writing the whole WAV again.
    if len(audio) <= chunk_ms:
        return [wav_path] if len(audio) else []

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav").close()

        chunks.append(chunk_path)
    
    return chunks


def process_input(source: str) -> list:
    """Prepare an existing local recording; network URLs are not supported."""
    source = source.strip().strip('"')
    if "://" in source:
        raise ValueError("Links are not supported. Enter a local audio/video file path.")
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Choose an existing local audio/video file.")
    print("Detected local file. Converting to WAV...")
    wav_path = convert_to_wav(str(path))
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
