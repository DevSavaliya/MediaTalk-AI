import contextlib
import io
import os
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

import main
from core import public_workflow, rag_engine, transcriber, ui_workflow, vector_store
from core.mistral_client import MistralAPIError
from utils import public_media

APP = Path(__file__).resolve().parents[1] / "cloud" / "streamlit_app.py"


def recording(seconds=0.1):
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * int(seconds * 16000))
    data.name = "sample.wav"
    return data


def result(chain=None, video_id="a" * 32):
    return {"id": video_id, "title": "Launch review", "transcript": "Ship on Friday.",
            "summary": "Friday launch", "action_items": "Test", "key_decisions": "Ship",
            "open_questions": "None", "analysis_error": None, "rag_chain": chain or object(),
            "source_label": "sample.wav", "created_at": "2026-10-27", "language": "english"}


class PublicMediaTests(unittest.TestCase):
    def test_youtube_url_canonicalizes_one_video(self):
        expected = "https://www.youtube.com/watch?v=abcdefghijk"
        self.assertEqual(public_media.youtube_url("https://youtu.be/abcdefghijk?list=private&t=20"), expected)
        self.assertEqual(public_media.youtube_url("https://www.youtube.com/shorts/abcdefghijk"), expected)
        self.assertEqual(public_media.youtube_url("https://www.youtube.com/watch?v=w9lXtKsucu0&list=LL&index=15"),
                         "https://www.youtube.com/watch?v=w9lXtKsucu0")

    def test_downloader_errors_are_classified_without_exposing_details(self):
        from yt_dlp.utils import DownloadError
        cases = [
            ("Sign in to confirm you're not a bot", "YT_ACCESS"),
            ("HTTP Error 429", "YT_ACCESS"),
            ("Private video", "YT_RESTRICTED"),
            ("Requested format is not available", "YT_FORMAT"),
            ("HTTP Error 403: Forbidden", "YT_FORBIDDEN"),
            ("Failed to resolve host", "YT_NETWORK"),
            ("Unexpected upstream error", "YT_DOWNLOAD"),
        ]
        for detail, code in cases:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as folder:
                with patch("yt_dlp.YoutubeDL") as downloader:
                    downloader.return_value.__enter__.return_value.extract_info.side_effect = DownloadError(
                        detail + " https://private.example/?token=secret")
                    with self.assertRaises(public_media.PublicDemoError) as caught:
                        public_media.download_youtube("https://youtu.be/abcdefghijk", folder)
                    self.assertIn(code, str(caught.exception))
                    self.assertNotIn("private.example", str(caught.exception))
                    self.assertNotIn("secret", str(caught.exception))

    def test_local_urls_userinfo_and_playlists_rejected(self):
        for source in ["D:/secret.mp4", "file:///etc/passwd", "http://127.0.0.1/video", 
                       "https://youtube.com.evil.test/watch?v=abcdefghijk",
                       "https://user:password@youtube.com/watch?v=abcdefghijk",
                       "https://youtube.com:123/watch?v=abcdefghijk",
                       "https://youtube.com/playlist?list=abc", "https://youtu.be/short"]:
            with self.subTest(source=source), self.assertRaises(public_media.PublicDemoError):
                public_media.youtube_url(source)

    def test_real_decoder_accepts_short_wav(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            source.write_bytes(recording().getvalue())
            paths = public_media.prepare_audio(source, folder)
            with wave.open(paths[0]) as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getframerate(), 16000)

    def test_duration_rejected_before_conversion(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            source.write_bytes(recording().getvalue())
            with patch.object(public_media.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, '{"format":{"duration":"601"}}')) as run:
                with self.assertRaisesRegex(public_media.PublicDemoError, "10 minutes"):
                    public_media.prepare_audio(source, folder)
                self.assertEqual(run.call_count, 1)

    def test_invalid_and_unbounded_durations_rejected(self):
        for value in [None, "NaN", "inf", -1, 0, 601]:
            with self.subTest(value=value), self.assertRaises(public_media.PublicDemoError):
                public_media._duration(value)

    def test_decoder_timeout_is_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.wav"
            source.write_bytes(recording().getvalue())
            with patch.object(public_media.subprocess, "run", side_effect=subprocess.TimeoutExpired("private-path", 30)):
                with self.assertRaises(public_media.PublicDemoError) as caught:
                    public_media.prepare_audio(source, folder)
                self.assertNotIn("private-path", str(caught.exception))


class PublicWorkspaceTests(unittest.TestCase):
    def setUp(self):
        patch.object(public_workflow, "BUDGET", public_workflow.HourlyBudget()).start()
        self.addCleanup(patch.stopall)

    def test_sessions_cannot_open_or_modify_each_others_video_or_chat(self):
        one, two = public_workflow.PublicWorkspace(), public_workflow.PublicWorkspace()
        video = result()
        one.videos[video["id"]] = video
        one.save_chat(video["id"], [{"role": "user", "content": "Private question"}])
        self.assertEqual(two.list_videos(), [])
        for operation in [lambda: two.open_video(video["id"]), lambda: two.load_chat(video["id"]),
                          lambda: two.save_chat(video["id"], []),
                          lambda: two.answer_question(video["rag_chain"], "Tell me")]:
            with self.assertRaises(public_media.PublicDemoError):
                operation()
        self.assertEqual(one.load_chat(video["id"])[0]["content"], "Private question")

    def test_clear_drops_data_but_keeps_usage_limits(self):
        workspace = public_workflow.PublicWorkspace()
        video = result()
        workspace.videos[video["id"]] = video
        workspace.save_chat(video["id"], [{"role": "user", "content": "Hello"}])
        workspace.video_attempts = 3
        workspace.question_attempts = 20
        workspace.clear()
        self.assertEqual(workspace.list_videos(), [])
        self.assertEqual(workspace.chats, {})
        self.assertEqual((workspace.video_attempts, workspace.question_attempts), (3, 20))

    def test_forged_local_path_rejected_before_processing(self):
        with patch.object(public_workflow.PublicWorkspace, "_process") as process:
            with self.assertRaises(public_media.PublicDemoError):
                public_workflow.PublicWorkspace().process_video("D:/secret.wav", "english", "source")
            process.assert_not_called()

    def test_oversized_upload_rejected_before_processing(self):
        with patch.object(public_workflow, "MAX_BYTES", 10), patch.object(public_workflow.PublicWorkspace, "_process") as process:
            with self.assertRaisesRegex(public_media.PublicDemoError, "50 MB"):
                public_workflow.PublicWorkspace().process_upload(recording(), "english")
            process.assert_not_called()

    def test_invalid_language_rejected(self):
        with self.assertRaisesRegex(public_media.PublicDemoError, "English or Hinglish"):
            public_workflow.PublicWorkspace().process_upload(recording(), "forged")

    def test_failed_processing_cleans_media_and_counts_attempt(self):
        workspace = public_workflow.PublicWorkspace()
        folders = []
        def fail(source, folder):
            folders.append(folder)
            raise public_media.PublicDemoError("Invalid media")
        with patch.object(public_workflow, "prepare_audio", side_effect=fail):
            with self.assertRaisesRegex(public_media.PublicDemoError, "Invalid media"):
                workspace.process_upload(recording(), "english")
        self.assertFalse(Path(folders[0]).exists())
        self.assertFalse(ui_workflow.PROCESS_LOCK.locked())
        self.assertEqual(workspace.video_attempts, 1)
        workspace.video_attempts = 3
        with self.assertRaisesRegex(public_media.PublicDemoError, "3 recording attempts"):
            workspace.process_upload(recording(), "english")

    def test_transient_transcription_never_reads_or_writes_shared_cache(self):
        with patch.object(transcriber, "_cache_path") as cache_path, \
             patch.object(transcriber, "_read_cache") as read_cache, \
             patch.object(transcriber, "_write_cache") as write_cache, \
             patch.object(transcriber, "transcribe_chunk", return_value="Private content"):
            self.assertEqual(transcriber.transcribe_all(["audio.wav"], use_cache=False), "Private content")
            cache_path.assert_not_called()
            read_cache.assert_not_called()
            write_cache.assert_not_called()

    def test_forged_question_length_rejected_before_api(self):
        workspace = public_workflow.PublicWorkspace()
        video = result()
        workspace.videos[video["id"]] = video
        with patch.object(ui_workflow, "answer_question") as answer:
            with self.assertRaises(public_media.PublicDemoError):
                workspace.answer_question(video["rag_chain"], "x" * 2001)
            answer.assert_not_called()

    def test_hourly_budget_shared_between_sessions_and_expires(self):
        budget = public_workflow.HourlyBudget(questions=1)
        with patch.object(public_workflow, "BUDGET", budget), patch.object(ui_workflow, "answer_question", return_value="Answer"), patch.object(public_workflow.time, "monotonic", return_value=0):
            first, second = public_workflow.PublicWorkspace(), public_workflow.PublicWorkspace()
            a, b = result(), result()
            first.videos[a["id"]] = a
            second.videos[b["id"]] = b
            first.answer_question(a["rag_chain"], "Question")
            with self.assertRaisesRegex(public_media.PublicDemoError, "hourly limit"):
                second.answer_question(b["rag_chain"], "Question")
        with patch.object(public_workflow.time, "monotonic", return_value=3601):
            budget.reserve("questions")

    def test_failed_answers_consume_budget_and_keep_video(self):
        workspace = public_workflow.PublicWorkspace()
        video = result()
        workspace.videos[video["id"]] = video
        with patch.object(ui_workflow, "answer_question", side_effect=MistralAPIError("Rate limited")):
            with self.assertRaises(MistralAPIError):
                workspace.answer_question(video["rag_chain"], "When?")
        self.assertEqual(workspace.question_attempts, 1)
        self.assertIs(workspace.open_video(video["id"]), video)
        workspace.question_attempts = 20
        with self.assertRaisesRegex(public_media.PublicDemoError, "20 question"):
            workspace.answer_question(video["rag_chain"], "Again?")

    def test_busy_processing_rejects_without_spending_session_allowance(self):
        workspace = public_workflow.PublicWorkspace()
        with ui_workflow.PROCESS_LOCK:
            with self.assertRaisesRegex(public_media.PublicDemoError, "Another recording"):
                workspace.process_upload(recording(), "english")
        self.assertEqual(workspace.video_attempts, 0)

    def test_pipeline_is_transient_private_and_cleans_media(self):
        prepared = []
        def transcribe(chunks, language, **options):
            prepared.extend(chunks)
            self.assertTrue(Path(chunks[0]).is_file())
            self.assertIs(options["use_cache"], False)
            return "PRIVATE_TRANSCRIPT: launch Friday."

        workspace = public_workflow.PublicWorkspace()
        with patch.object(main, "transcribe_all", side_effect=transcribe), \
             patch.object(main, "generate_title", side_effect=MistralAPIError("Rate limited")), \
             patch.object(vector_store, "get_embeddings", return_value=DeterministicFakeEmbedding(size=16)), \
             patch.object(rag_engine, "get_llm", return_value=RunnableLambda(lambda _: AIMessage(content="Friday"))), \
             patch.object(vector_store, "build_vector_store") as disk_store, \
             patch.object(ui_workflow, "save_result") as disk_save, \
             contextlib.redirect_stdout(io.StringIO()) as logs:
            video = workspace.process_upload(recording(), "english")
            self.assertEqual(video["analysis_error"], "Rate limited")
            self.assertEqual(video["rag_chain"].invoke("When?"), "Friday")
            self.assertEqual(len(workspace.list_videos()), 1)
            disk_store.assert_not_called()
            disk_save.assert_not_called()
        self.assertNotIn("PRIVATE_TRANSCRIPT", logs.getvalue())
        self.assertFalse(Path(prepared[0]).exists())

    def test_index_retrieval_is_isolated_even_for_identical_questions(self):
        with patch.object(vector_store, "get_embeddings", return_value=DeterministicFakeEmbedding(size=16)):
            a = vector_store.build_session_vector_store("Alice private Friday launch")
            b = vector_store.build_session_vector_store("Bob private Monday launch")
            self.assertEqual(a.similarity_search("launch")[0].page_content, "Alice private Friday launch")
            self.assertEqual(b.similarity_search("launch")[0].page_content, "Bob private Monday launch")


class PublicUITests(unittest.TestCase):
    def setUp(self):
        patch.dict(os.environ, {"MEDIATALK_PUBLIC_DEMO": "1"}).start()
        patch.object(public_workflow, "BUDGET", public_workflow.HourlyBudget()).start()
        self.addCleanup(patch.stopall)

    def app(self):
        return AppTest.from_file(str(APP), default_timeout=30)

    def test_public_entrypoint_hides_local_library_and_ignores_url_ids(self):
        with patch.object(ui_workflow, "list_videos") as listing, patch.object(ui_workflow, "open_cli_video") as cli, patch.object(ui_workflow, "open_video") as opening:
            at = self.app()
            at.query_params["video"] = "cli"
            at.run()
            self.assertFalse(at.exception)
            self.assertNotIn("video", at.query_params)
            self.assertNotIn("Local file", at.get("button_group")[0].options)
            self.assertFalse(any(button.key == "open_cli" for button in at.button))
            listing.assert_not_called()
            opening.assert_not_called()
            cli.assert_not_called()

    def test_conversation_survives_reruns_but_not_another_session(self):
        at = self.app().run()
        workspace = at.session_state["_public_workspace"]
        video = result()
        workspace.videos[video["id"]] = video
        at.session_state["video_result"] = video
        with patch.object(ui_workflow, "answer_question", return_value="Friday.") as answer:
            at.run()
            at.chat_input[0].set_value("When?").run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.session_state["chat_messages"]), 2)
            at.run()
            self.assertEqual(answer.call_count, 1)
            other = self.app()
            other.query_params["video"] = video["id"]
            other.run()
            self.assertFalse(other.exception)
            self.assertIsNone(other.session_state["video_result"])
            self.assertEqual(other.session_state["_public_workspace"].list_videos(), [])
            self.assertEqual(other.session_state["chat_messages"], [])
            at.button(key="clear_session").click().run()
            self.assertFalse(at.exception)
            self.assertIsNone(at.session_state["video_result"])
            self.assertEqual(workspace.list_videos(), [])

    def test_api_error_preserves_workspace_and_conversation(self):
        at = self.app().run()
        video = result()
        at.session_state["_public_workspace"].videos[video["id"]] = video
        at.session_state["video_result"] = video
        with patch.object(ui_workflow, "answer_question", side_effect=MistralAPIError("Rate limited")):
            at.run()
            at.chat_input[0].set_value("When?").run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertIsNotNone(at.session_state["video_result"])
            self.assertEqual(len(at.session_state["chat_messages"]), 2)


if __name__ == "__main__":
    unittest.main()
