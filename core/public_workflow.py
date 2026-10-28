"""Session-owned demo data; only resource budgets and model weights are shared."""

import copy
import os
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from core import ui_workflow
from utils.public_media import (
    MAX_BYTES, MAX_SECONDS, MAX_UPLOAD_MB, PublicDemoError,
    download_youtube, prepare_audio, youtube_url,
)

MAX_VIDEOS_PER_SESSION = 3
MAX_QUESTIONS_PER_SESSION = 20
MAX_QUESTION_CHARS = 2000


class HourlyBudget:
    """Process-wide attempt caps; fresh browser sessions cannot reset them."""
    def __init__(self, videos=10, questions=100):
        self.limits = {"videos": videos, "questions": questions}
        self.events = {key: deque() for key in self.limits}
        self.lock = threading.Lock()

    def reserve(self, kind):
        with self.lock:
            now = time.monotonic()
            events = self.events[kind]
            while events and events[0] <= now - 3600:
                events.popleft()
            if len(events) >= self.limits[kind]:
                raise PublicDemoError("The demo's shared hourly limit has been reached. Please try again later.")
            events.append(now)


BUDGET = HourlyBudget()
CHAT_LOCK = threading.BoundedSemaphore(2)


class PublicWorkspace:
    """Create inside Session State, never in a module global or st.cache_resource."""
    MEDIA_TYPES = ui_workflow.MEDIA_TYPES
    export_notes = staticmethod(ui_workflow.export_notes)

    def __init__(self):
        self.videos = {}
        self.chats = {}
        self.video_attempts = 0
        self.question_attempts = 0

    def list_videos(self):
        return [{key: value[key] for key in ("id", "title", "source_label", "created_at")}
                for value in reversed(list(self.videos.values()))]

    def open_video(self, video_id):
        if video_id not in self.videos:
            raise PublicDemoError("This video is not part of your current session. Please upload it again.")
        return self.videos[video_id]

    def load_chat(self, video_id):
        self.open_video(video_id)
        return copy.deepcopy(self.chats.get(video_id, []))

    def save_chat(self, video_id, messages):
        self.open_video(video_id)
        self.chats[video_id] = copy.deepcopy(messages[-100:])

    def clear(self):
        self.videos.clear()
        self.chats.clear()
        # Clearing private content must not reset the session's usage allowance.

    def _check_language(self, language):
        if language not in {"english", "hinglish"}:
            raise PublicDemoError("Choose English or Hinglish.")
        if language == "hinglish" and not os.getenv("SARVAM_API_KEY"):
            raise PublicDemoError("Hinglish is not configured for this demo. Please choose English.")

    def _process(self, language, label, prepare, on_progress):
        self._check_language(language)
        if self.video_attempts >= MAX_VIDEOS_PER_SESSION:
            raise PublicDemoError("This session has used its 3 recording attempts. Download your notes before leaving.")
        if not ui_workflow.PROCESS_LOCK.acquire(blocking=False):
            raise PublicDemoError("Another recording is being processed. Please try again shortly.")
        try:
            BUDGET.reserve("videos")
            self.video_attempts += 1
            from main import run_pipeline

            with tempfile.TemporaryDirectory(prefix="mediatalk-demo-") as folder:
                if on_progress:
                    on_progress("audio", {})
                chunks = prepare(folder)
                result = run_pipeline("", language, on_progress=on_progress,
                                      prepared_chunks=chunks, transient=True)
            result.update(id=uuid4().hex, source_label=label, language=language,
                          created_at=datetime.now(timezone.utc).isoformat())
            self.videos[result["id"]] = result
            return result
        finally:
            ui_workflow.PROCESS_LOCK.release()

    def process_video(self, source, language, source_label, on_progress=None):
        canonical = youtube_url(source)  # No server-local paths, even for forged widgets.
        return self._process(language, canonical,
                             lambda folder: prepare_audio(download_youtube(canonical, folder), folder),
                             on_progress)

    def process_upload(self, upload, language, on_progress=None):
        suffix = Path(upload.name).suffix.lower()
        if suffix not in self.MEDIA_TYPES:
            raise PublicDemoError("Choose a supported audio or video file.")
        with upload.getbuffer() as data:
            if not 0 < len(data) <= MAX_BYTES:
                raise PublicDemoError("Choose a non-empty file of at most 50 MB.")

            def prepare(folder):
                source = Path(folder) / f"source{suffix}"
                source.write_bytes(data)
                return prepare_audio(source, folder)

            return self._process(language, Path(upload.name).name, prepare, on_progress)

    def answer_question(self, chain, question, history=None):
        if not any(result["rag_chain"] is chain for result in self.videos.values()):
            raise PublicDemoError("Open a video from your current session first.")
        if not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_CHARS:
            raise PublicDemoError("Enter a question between 1 and 2,000 characters.")
        if self.question_attempts >= MAX_QUESTIONS_PER_SESSION:
            raise PublicDemoError("This session has used its 20 question attempts. You can still read and download your notes.")
        if not CHAT_LOCK.acquire(blocking=False):
            raise PublicDemoError("The demo is answering other questions. Please try again shortly.")
        try:
            BUDGET.reserve("questions")
            self.question_attempts += 1
            return ui_workflow.answer_question(chain, question, history=history)
        finally:
            CHAT_LOCK.release()
