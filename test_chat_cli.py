import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import main
from core.mistral_client import MistralAPIError


class ChatCLITests(unittest.TestCase):
    def test_saved_video_chat_skips_transcription_and_analysis(self):
        chain = Mock()
        output = io.StringIO()
        with patch.object(main, "load_rag_chain", return_value=chain), \
             patch.object(main, "run_pipeline") as pipeline, \
             patch.object(main, "ask_question", return_value="Testing is Wednesday.") as ask, \
             patch("builtins.input", side_effect=["When is testing?", "exit"]), \
             contextlib.redirect_stdout(output):
            self.assertEqual(main.main(["--chat"]), 0)
        pipeline.assert_not_called()
        ask.assert_called_once_with(chain, "When is testing?")
        self.assertIn("Chat with your meeting", output.getvalue())
        self.assertEqual(output.getvalue().count("Testing is Wednesday."), 1)

    def test_chat_keyword_reopens_database(self):
        chain = Mock()
        with patch.object(main, "load_rag_chain", return_value=chain) as load, \
             patch.object(main, "run_pipeline") as pipeline, \
             patch("builtins.input", side_effect=["chat", "exit"]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.main([]), 0)
        load.assert_called_once()
        pipeline.assert_not_called()

    def test_chat_failure_returns_to_prompt_and_can_recover(self):
        output = io.StringIO()
        with patch.object(main, "ask_question", side_effect=[
            MistralAPIError("Mistral limit of 0"), "The launch is Friday."
        ]) as ask, \
             patch("builtins.input", side_effect=["When?", "When is the launch?", "exit"]), \
             contextlib.redirect_stdout(output):
            main.chat_loop(Mock())
        self.assertEqual(ask.call_count, 2)
        self.assertIn("Unable to generate an answer:", output.getvalue())
        self.assertIn("The launch is Friday.", output.getvalue())

    def test_missing_saved_database_has_actionable_error(self):
        output = io.StringIO()
        with patch.object(main, "load_rag_chain", side_effect=FileNotFoundError(
            "No saved video database found. Process a video first.")), \
             patch("builtins.input") as user_input, \
             contextlib.redirect_stdout(output):
            self.assertEqual(main.main(["--chat"]), 1)
        user_input.assert_not_called()
        self.assertIn("Process a video first", output.getvalue())


if __name__ == "__main__":
    unittest.main()
