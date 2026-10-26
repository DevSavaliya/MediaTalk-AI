"""Compare local transcription on the same clip; never calls the analysis API."""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--engine", choices=["original", "optimized"], required=True)
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    from dotenv import load_dotenv
    from pydub import AudioSegment
    load_dotenv(ROOT / ".env")
    folder = ROOT / "artifacts" / "transcription-benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    clip = folder / "sample.wav"
    audio = AudioSegment.from_file(args.source)[:args.seconds * 1000]
    audio.set_channels(1).set_frame_rate(16000).set_sample_width(2).export(clip, format="wav").close()
    start = time.perf_counter()
    cache_seconds = None
    if args.engine == "original":
        import whisper
        import os
        model = whisper.load_model(os.getenv("WHISPER_MODEL", "small"))
        loaded = time.perf_counter()
        text = model.transcribe(str(clip), task="transcribe")["text"]
        end = time.perf_counter()
    else:
        from core import transcriber
        transcriber.load_model()
        loaded = time.perf_counter()
        with tempfile.TemporaryDirectory() as cache:
            transcriber.CACHE_DIR = Path(cache)
            text = transcriber.transcribe_all([str(clip)])
            end = time.perf_counter()
            repeated = transcriber.transcribe_all([str(clip)])
            cache_seconds = round(time.perf_counter() - end, 4)
            assert repeated == text
    result = {"engine": args.engine, "audio_seconds": len(audio) / 1000,
              "load_seconds": round(loaded - start, 3),
              "transcribe_seconds": round(end - loaded, 3), "text": text}
    if cache_seconds is not None:
        result["cache_hit_seconds"] = cache_seconds
    (folder / f"{args.engine}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "text"}), flush=True)


if __name__ == "__main__":
    main()
