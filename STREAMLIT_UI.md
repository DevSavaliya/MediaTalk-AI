# Video Agent workspace

Automatic source-file watching is disabled to avoid conflicts with Transformers' lazy imports. Restart Streamlit after changing Python files.

Run from the VS Code terminal in `D:\Video Agent`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local address printed in the terminal (normally `http://localhost:8501`).
The original CLI still runs with `python main.py` or `python main.py --chat`.

## Workflow

1. Choose a YouTube link, upload an audio/video file, or enter a local file path.
2. Select English or Hinglish and click **Transcribe & explore**.
3. Follow the processing status. The full transcription becomes available before analysis finishes.
4. Read **Transcript**, **Summary**, and **Takeaways** beside **Ask your video**.
5. Download the transcript or all notes. Reopen completed videos from **Your library**.

**Open saved CLI video** loads the database made by earlier CLI runs. Those
runs saved transcript chunks, not summary text, so the imported workspace shows
the recovered chunks and does not invent missing analysis. The older CLI
collection can contain content accumulated across earlier runs.

## Configuration and storage

- Streamlit 1.64 or later is required; the project's environment was built against 1.64.0.
- The existing project `.env` supplies `MISTRAL_API_KEY`. Optional `MISTRAL_MODEL`
  selects a model; the current default is `ministral-3b-latest`.
- Hinglish uses the existing Sarvam transcription backend and requires
  `SARVAM_API_KEY`. English uses local Whisper. FFmpeg must be available for media conversion.
- English now uses faster-whisper with the same `small` model, CPU INT8, known
  English language, and conservative silence filtering. The first run downloads
  the converted model; later runs reuse it. Set `WHISPER_ENGINE=openai` in `.env`
  to use the original engine. `WHISPER_MODEL` still selects the model size.
- Completed audio chunks are cached locally in `transcript_cache/`. Matching
  audio content, language, and model settings reuse the transcription even after
  an application restart. Set `TRANSCRIPTION_CACHE=0` to disable reuse. This
  skips speech recognition only; downloading/conversion and analysis still run.
  Failed or empty transcriptions are never cached. The UI shows chunk progress
  and elapsed transcription time. Hinglish first-run API requests stay sequential
  to avoid increasing API rate-limit pressure.
- Appearance and the 500 MB upload limit are configured in `.streamlit/config.toml`.
- The server binds to the local computer. This interface is intended for local use.
- New UI videos each have an isolated collection in `vector_db/chroma.sqlite3`.
  They do not add their transcript to another UI video's searchable context.
- Transcripts and generated notes persist in `video_library/*.json`; no API keys
  or live client objects are included. Uploaded source media is processed in a
  temporary directory and cleaned up when processing finishes.
- UI results and the chat chain stay in Session State, so changing tabs, asking
  questions, or downloading notes does not rerun transcription. The embedding
  model is cached in memory. The workspace URL remembers the selected video and
  reading tab. Reloading or reconnecting restores its notes and conversation.
- Chats persist separately for each video in `video_chats/*.json` (latest 100
  messages). Clear chat clears that video's saved conversation. Recent successful
  turns provide context for follow-up questions; errors are excluded, and answers
  must still be supported by the retrieved transcript. Restoring a chat does not
  automatically resend a question or incur another generation request.
- The app requires `websockets>=14,<17`, matching Streamlit 1.64's supported range.
- Analysis failures retain the transcript and chat interface. API failures appear
  in the conversation without a traceback or fabricated answer.

The app uses the existing pipeline and Mistral provider. Transcript text is sent
to the configured services for analysis and answers, as in the CLI.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Streamlit AppTest covers input validation, processing, chat, error recovery,
library reopening, and avoiding duplicate API calls on reruns. Workflow tests
use real temporary Chroma databases to check isolation and persistence, with
offline embeddings and simulated API responses.
