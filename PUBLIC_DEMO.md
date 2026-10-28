# MediaTalk AI public demo preparation

Status: implemented and tested locally; not yet verified on Streamlit Community Cloud.
The user confirmed the local processing, two-turn chat, and separate private-window
checks on October 27, 2026.
The intended subdomain is `mediatalk-ai`, subject to availability.

## Preview

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_cloud.py --server.port 8502
```

This uses the existing local `.env` for provider credentials. The original
`streamlit_app.py` command still provides the persistent local library.

## Session isolation

- `streamlit_cloud.py` always enables public mode for local previews.
  Community Cloud should use `cloud/streamlit_app.py`, which runs that same public
  entry point and forces the faster-whisper backend.
- A `PublicWorkspace` object is created inside Streamlit Session State. It is never
  placed in a shared cache. Each visitor gets an independent library and chat store.
- Each processed video has a new in-memory vector store. Retrieval uses the same
  local sentence-transformer embedding model and the existing RAG prompt/history logic.
- Model weights and process-wide resource guards are shared; transcripts, indexes,
  generated notes, and conversations are not.
- Foreign video IDs, including the legacy `cli` ID, cannot restore a public
  workspace. Public mode ignores workspace IDs in the URL.
- Public mode never saves results to `video_library/`, `video_chats/`, Chroma,
  or the shared transcript cache. The local mode continues to use these locations.
- Downloaded/uploaded media and the normalized WAV use a per-job temporary folder,
  removed when processing exits. An abrupt operating-system/process termination
  may prevent that cleanup; this is not a secure-erasure guarantee.
- Session state is not an authenticated account. Refreshes/reconnections can
  create a new empty workspace. Users should download notes before leaving.
- Clear session data drops application references to session videos and chats;
  memory reclamation follows the runtime's normal session/garbage-collection lifecycle.

## Enforced limits

| Limit | Public demo |
| --- | --- |
| Uploaded file or downloaded audio | 50 MiB |
| Recording duration | 10 minutes |
| Recording attempts per session | 3 |
| Question attempts per session | 20 |
| Question length | 2,000 characters |
| Recording attempts across all sessions | 10 per rolling hour |
| Question attempts across all sessions | 100 per rolling hour |
| Concurrent recording pipelines | 1 |
| Concurrent chat generations | 2 |

Validation is performed in backend code as well as in the widgets. Invalid sources
and oversized uploads are rejected before processing. Attempts that reach processing
or generation consume their allowance even if they fail. Busy rejections do not.
Clearing a session's data does not reset its counters. A new session resets only
its per-session counters; shared hourly counters reset when the server process restarts.

Hourly limits count workflows, not individual provider requests. Analysis may make
multiple Mistral calls, with the existing bounded retry policy. These guards reduce
usage but do not enforce an account-level money budget. Account/provider limits
remain the authority on spend and availability. Limits are process-local; this
design is intended for a small single-process portfolio demo, not a scaled service.

Media is probed with a 30-second timeout before decoding. FFmpeg decoding is capped
at 601 seconds of output and 120 seconds of execution; decoded audio exceeding
600 seconds is rejected rather than silently summarized as a complete video.
Only supported media demuxers and local file/pipe protocols are allowed.
YouTube accepts one canonical video URL, rejects live streams/playlists, checks
duration before download, and limits transfer size. Socket waits and retries are
bounded; the download progress hook also checks an elapsed-time allowance. That
hook is not a strict wall-clock timeout for the entire YouTube extraction process.
Cloud-hosted YouTube downloads can be blocked upstream; users can upload a file instead.

## Provider data and errors

Transcript content and questions are sent to Mistral. Hinglish audio is sent to
Sarvam and requires the owner's `SARVAM_API_KEY`. English uses local faster-whisper
with two CPU threads in public mode. Temporary media is processed on the server,
not inside the visitor's browser. The UI communicates this before processing.

The public pipeline does not print transcript contents. Sarvam error bodies are
no longer printed. Known rate-limit errors remain visible in the workspace without
discarding the transcript or chat. Other failures use safe generic messages.

## Community Cloud configuration

- Repository: `DevSavaliya/MediaTalk-AI`; branch: `main`.
- Main file: `cloud/streamlit_app.py`; Python version: **3.12**.
- The script-level `cloud/requirements.txt` is selected before the root local
  requirements. It uses the official PyTorch CPU wheel for CPython 3.12/Linux x86_64.
  Other packages resolve from PyPI. Original Whisper and its Linux Triton dependency
  are excluded from the cloud installation; the local fallback remains available.
- The script-level `cloud/.streamlit/config.toml` overrides the root loopback-only
  address for hosting and caps uploads at 50 MB. The root configuration still keeps
  local previews on loopback. Theme settings continue to come from the root file.
- Root `packages.txt` installs FFmpeg. Configure credentials through Community Cloud
  Secrets, never in GitHub. No deployment or credential entry has been performed yet.

On October 28, 2026, `uv pip compile` resolved the cloud requirements for Python 3.12,
Linux x86_64/glibc 2.31 with `--only-binary :all:`. The 134-package resolution contains
no NVIDIA/CUDA/Triton packages. This checks metadata and compatible wheel availability;
it is **not** a fresh Linux installation, import test, or successful cloud deployment.
The resolver ran from Windows. Diagnostic output is kept under ignored `artifacts/`.

## Deployment work still required

1. Upload the updated application and cloud configuration files manually.
2. Deploy using the settings above and verify the actual dependency installation.
3. Verify upload, transcription, analysis, two-turn chat, error recovery, and two
   independent visitor sessions on the hosted app.
4. Check actual memory/CPU limits with the real model downloads. These have not
   been measured on Community Cloud; do not advertise a verified live demo yet.

## Automated verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite covers local behavior plus independent sessions, cross-session access
rejection, in-memory retrieval isolation, temporary-media cleanup on success and
failure, disabled shared caching, server-side limits, real FFmpeg decoding, and
Streamlit AppTest chat/rerun/error flows. Providers and embeddings are simulated
for deterministic tests; no live API or cloud deployment success is implied.
