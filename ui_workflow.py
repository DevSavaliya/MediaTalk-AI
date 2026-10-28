"""Streamlit's local video library; the processing pipeline stays in main.py."""

import json
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "video_library"
CHAT_DIR = ROOT / "video_chats"
MEDIA_TYPES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".mp3", ".wav", ".m4a", ".ogg"}
PROCESS_LOCK = threading.Lock()


def list_videos():
    videos = []
    if not LIBRARY.exists():
        return videos
    for path in sorted(LIBRARY.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if re.fullmatch(r"[a-f0-9]{32}", data.get("id", "")) and data["id"] == path.stem:
                videos.append({key: data.get(key) for key in ("id", "title", "source_label", "created_at")})
        except (ValueError, OSError, TypeError):
            continue
    return videos


def save_result(result):
    LIBRARY.mkdir(parents=True, exist_ok=True)
    data = {key: value for key, value in result.items() if key != "rag_chain"}
    path = LIBRARY / f"{result['id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_video(source, language, source_label, on_progress=None):
    from main import run_pipeline

    # Local Whisper is shared by the existing CLI backend. Serialize ingestion.
    if not PROCESS_LOCK.acquire(blocking=False):
        raise ValueError("Another video is being processed. Please wait for it to finish.")
    try:
        video_id = uuid4().hex
        collection = f"video_{video_id}"
        result = run_pipeline(source, language, on_progress=on_progress, collection_name=collection)
        result.update(id=video_id, collection_name=collection, source_label=source_label,
                      language=language, created_at=datetime.now(timezone.utc).isoformat())
        try:
            save_result(result)
        except OSError:
            result["save_error"] = "This session is ready, but its library entry could not be saved. Download your notes before closing."
        return result
    finally:
        PROCESS_LOCK.release()


def process_upload(upload, language, on_progress=None):
    suffix = Path(upload.name).suffix.lower()
    if suffix not in MEDIA_TYPES:
        raise ValueError("Choose a supported audio or video file.")
    # Never use an uploaded filename as a filesystem path.
    with tempfile.TemporaryDirectory(prefix="video-agent-upload-", ignore_cleanup_errors=True) as folder:
        path = Path(folder) / f"source{suffix}"
        path.write_bytes(upload.getbuffer())
        return process_video(str(path), language, Path(upload.name).name, on_progress)


def open_video(video_id):
    if not re.fullmatch(r"[a-f0-9]{32}", video_id):
        raise ValueError("Choose a saved video from the library.")
    data = json.loads((LIBRARY / f"{video_id}.json").read_text(encoding="utf-8"))
    if data.get("collection_name") != f"video_{video_id}":
        raise ValueError("This saved video's index does not match its library entry.")
    from core.rag_engine import load_rag_chain
    data["rag_chain"] = load_rag_chain(collection_name=data["collection_name"])
    return data


def open_cli_video():
    from core.vector_store import load_vector_store
    from core.rag_engine import load_rag_chain

    store = load_vector_store()
    records = store.get(include=["documents", "metadatas"])
    chunks = sorted({(meta.get("chunk_index", 0), text)
                     for text, meta in zip(records["documents"], records["metadatas"])
                     if text and meta})
    # Old CLI runs saved chunks, not the original complete transcript or analyses.
    return {
        "id": "cli", "title": "Saved CLI video", "source_label": "Existing transcript index",
        "transcript": "\n\n".join(text for _, text in chunks), "recovered_chunks": True,
        "summary": "No saved summary. Your CLI run stored the transcript index only.",
        "action_items": "Not saved.", "key_decisions": "Not saved.", "open_questions": "Not saved.",
        "analysis_error": None, "rag_chain": load_rag_chain(), "language": "—",
    }


def _chat_path(video_id):
    if video_id != "cli" and not re.fullmatch(r"[a-f0-9]{32}", video_id):
        raise ValueError("Invalid saved video.")
    return CHAT_DIR / f"{video_id}.json"


def load_chat(video_id):
    path = _chat_path(video_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if not isinstance(data, list):
        raise ValueError("Saved conversation is invalid.")
    return [message for message in data[-100:] if isinstance(message, dict)
            and message.get("role") in ("user", "assistant")
            and isinstance(message.get("content"), str)]


def save_chat(video_id, messages):
    path = _chat_path(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(messages[-100:], output, ensure_ascii=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def answer_question(chain, question, history=None):
    from core.rag_engine import ask_question
    return ask_question(chain, question, history=history)


def export_notes(result):
    parts = [f"# {result['title']}", f"Source: {result.get('source_label', '')}"]
    for key, heading in [("summary", "Summary"), ("action_items", "Action items"),
                         ("key_decisions", "Key decisions"), ("open_questions", "Open questions"),
                         ("transcript", "Transcript")]:
        parts.append(f"## {heading}\n\n{result.get(key, 'Not generated.')}")
    return "\n\n".join(parts)
