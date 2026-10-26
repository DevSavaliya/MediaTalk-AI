# Transcription performance

Measured on this computer using the first 60 seconds of the existing
`The AI Economy is about to change.wav` recording. Both engines used the
`small` Whisper model and four CPU inference threads. CUDA was unavailable.

| Measurement | Original | Optimized |
| --- | ---: | ---: |
| Imports and model loading (weights already downloaded) | 21.753 s | 4.735 s |
| Transcribing 60 seconds of audio | 69.193 s | 13.324 s |
| Reusing the optimized transcript cache | — | 0.0191 s |

The measured speech-recognition stage was **5.19 times faster**, taking about
81% less time. These are single-run measurements on a short sample, not a
guarantee for every video. Downloading, media conversion, indexing, and Mistral
analysis are separate stages. The one-time model download is excluded.

The optimized transcript covers the same beginning and ending as the original;
the inspected differences were punctuation, number formatting, and small wording
differences. This is not a formal transcription-accuracy evaluation.

## Changes

- faster-whisper uses CPU INT8 with the same model size, explicit English,
  greedy decoding, and conservative silence filtering.
- Content-based chunk caching avoids repeating successful transcription after
  re-upload, restart, or a later-chunk failure. Language/model changes invalidate
  the cache; failures and empty text are not stored.
- Short WAV recordings are no longer copied into a redundant chunk file.
  Newly downloaded YouTube audio uses mono 16 kHz for speech recognition.
- Streamlit displays chunk progress and elapsed time.
- PyAV is constrained to the tested 16.x decoder API because version 19 is
  incompatible with faster-whisper 1.2.1's audio decoder.

Hinglish still uses Sarvam. Its first-run API latency is unchanged; repeated
audio benefits from caching. The English speed measurement does not apply to it.

## Reproduce

```powershell
.\.venv\Scripts\python.exe tools/benchmark_transcription.py "downloades/The AI Economy is about to change.wav" --engine original
.\.venv\Scripts\python.exe tools/benchmark_transcription.py "downloades/The AI Economy is about to change.wav" --engine optimized
```

Results, including transcripts, are in `artifacts/transcription-benchmark/`.
Each optimized benchmark uses a fresh temporary transcript cache for a genuine
cold-transcription measurement followed by a cache-hit check.

Set `WHISPER_ENGINE=openai` in `.env` to use the original engine, or
`TRANSCRIPTION_CACHE=0` to disable transcript reuse. Restart the app after
changing configuration. Cache files contain transcript text and stay locally in
`transcript_cache/`.

Implementation references: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and its [PyAV compatibility issue](https://github.com/SYSTRAN/faster-whisper/issues/1589).

An environment-wide `uv pip check` also reported existing websockets and
OpenTelemetry version conflicts. Installing the new engine added only
faster-whisper, CTranslate2 and PyAV; those unrelated conflicts were not changed.
