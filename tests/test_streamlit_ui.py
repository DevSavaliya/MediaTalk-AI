import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from core import ui_workflow
from core.mistral_client import MistralAPIError

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def sample_result():
    return {"id": "a" * 32, "title": "Friday launch planning", "source_label": "sample.mp4",
            "transcript": "Alex will test on Wednesday. We launch on Friday.",
            "summary": "Launch on Friday.", "action_items": "1. Alex: testing by Wednesday.",
            "key_decisions": "1. Launch Friday.", "open_questions": "No open questions found.",
            "analysis_error": None, "rag_chain": "test-chain", "language": "english"}


class StreamlitUITests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patch.object(ui_workflow, "CHAT_DIR", Path(folder.name)).start()
        patch.object(ui_workflow, "list_videos", return_value=[]).start()
        self.addCleanup(patch.stopall)

    def app(self):
        return AppTest.from_file(str(APP), default_timeout=30)

    def test_empty_state_and_validation_do_not_process(self):
        with patch.object(ui_workflow, "process_video") as process:
            at = self.app().run()
            self.assertFalse(at.exception)
            self.assertTrue(any("Turn a video" in title.value for title in at.title))
            at.button(key="process_video").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            process.assert_not_called()

    def test_processing_chat_and_reruns_do_not_repeat_pipeline(self):
        with patch.object(ui_workflow, "process_video", return_value=sample_result()) as process, \
             patch.object(ui_workflow, "answer_question", return_value="Friday.") as answer:
            at = self.app().run()
            at.text_input(key="youtube_source").set_value("https://www.youtube.com/watch?v=example")
            at.button(key="process_video").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["video_result"]["title"], "Friday launch planning")
            self.assertIn("Alex will test", at.text[0].value)
            at.chat_input[0].set_value("When is the launch?").run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.session_state["chat_messages"]), 2)
            answer.assert_called_once_with("test-chain", "When is the launch?", history=[])
            at.run()
            process.assert_called_once()
            answer.assert_called_once()
            at.button(key="clear_chat").click().run()
            self.assertEqual(at.session_state["chat_messages"], [])
            self.assertIsNotNone(at.session_state["video_result"])

    def test_transcription_progress_callback_renders(self):
        def process(source, language, label, callback):
            callback("transcribing_progress", {"chunk": 1, "chunks": 1, "fraction": 0.5,
                                                "cached": False, "elapsed": 2.0})
            return sample_result()

        with patch.object(ui_workflow, "process_video", side_effect=process):
            at = self.app().run()
            at.text_input(key="youtube_source").set_value("https://youtu.be/example")
            at.button(key="process_video").click().run()
            self.assertFalse(at.exception)
            self.assertIsNotNone(at.session_state["video_result"])

    def test_api_failure_retains_transcript_and_chat(self):
        result = sample_result()
        result["analysis_error"] = "Mistral rate limit reached."
        at = self.app()
        at.session_state["video_result"] = result
        with patch.object(ui_workflow, "answer_question", side_effect=MistralAPIError("Try again later.")):
            at.run()
            self.assertTrue(at.warning)
            at.chat_input[0].set_value("What happened?").run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertTrue(at.session_state["chat_messages"][-1]["error"])
            self.assertEqual(at.session_state["video_result"]["transcript"], result["transcript"])

    def test_reopen_saved_video_and_new_video_reset_conversation(self):
        record = {"id": "a" * 32, "title": "Friday launch planning"}
        with patch.object(ui_workflow, "list_videos", return_value=[record]), \
             patch.object(ui_workflow, "open_video", return_value=sample_result()) as reopen:
            at = self.app().run()
            at.button(key="open_video").click().run()
            self.assertFalse(at.exception)
            reopen.assert_called_once_with("a" * 32)
            at.button(key="new_video").click().run()
            self.assertFalse(at.exception)
            self.assertIsNone(at.session_state["video_result"])
            self.assertEqual(at.session_state["chat_messages"], [])

    def test_load_failure_does_not_remove_current_video(self):
        with patch.object(ui_workflow, "list_videos", return_value=[{"id": "a" * 32, "title": "Missing"}]), \
             patch.object(ui_workflow, "open_video", side_effect=FileNotFoundError()):
            at = self.app()
            at.session_state["video_result"] = sample_result()
            at.run()
            at.button(key="open_video").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertIsNotNone(at.session_state["video_result"])

    def test_multiple_turns_restore_after_a_fresh_browser_session(self):
        result = sample_result()
        with patch.object(ui_workflow, "open_video", return_value=result), \
             patch.object(ui_workflow, "answer_question", side_effect=["Friday.", "Alex will test on Wednesday."]) as answer:
            at = self.app()
            at.query_params["video"] = result["id"]
            at.query_params["view"] = "Summary"
            at.run()
            at.chat_input[0].set_value("When is the launch?").run()
            at.chat_input[0].set_value("Who is testing before it?").run()
            self.assertFalse(at.exception)
            self.assertEqual(len(at.session_state["chat_messages"]), 4)
            self.assertEqual(answer.call_args.kwargs["history"][1]["content"], "Friday.")
            self.assertEqual(at.session_state[f"video_view_{result['id']}"], "Summary")
            reopened = self.app()
            reopened.query_params.update(at.query_params)
            reopened.run()
            self.assertFalse(reopened.exception)
            self.assertEqual(reopened.session_state["video_result"]["id"], result["id"])
            self.assertEqual(len(reopened.session_state["chat_messages"]), 4)
            self.assertEqual(answer.call_count, 2)  # Restoring never resends a message.
            self.assertFalse(any("Turn a video" in title.value for title in reopened.title))
            reopened.button(key="clear_chat").click().run()
            self.assertEqual(ui_workflow.load_chat(result["id"]), [])
            reopened.button(key="new_video").click().run()
            self.assertNotIn("video", reopened.query_params)
            self.assertIsNone(reopened.session_state["video_result"])

    def test_reload_failure_stays_on_recovery_view(self):
        with patch.object(ui_workflow, "open_video", side_effect=FileNotFoundError()):
            at = self.app()
            at.query_params["video"] = "a" * 32
            at.run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertFalse(any("Turn a video" in title.value for title in at.title))
            at.button(key="new_video").click().run()
            self.assertTrue(any("Turn a video" in title.value for title in at.title))

    def test_empty_answer_and_save_error_keep_video_open(self):
        at = self.app()
        at.session_state["video_result"] = sample_result()
        with patch.object(ui_workflow, "answer_question", return_value=""), \
             patch.object(ui_workflow, "save_chat", side_effect=OSError()):
            at.run()
            at.chat_input[0].set_value("What happened?").run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertTrue(at.warning)
            self.assertIsNotNone(at.session_state["video_result"])


if __name__ == "__main__":
    unittest.main()
