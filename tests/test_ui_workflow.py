import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import main
from core import rag_engine, ui_workflow, vector_store


class UIWorkflowTests(unittest.TestCase):
    def test_distinct_videos_are_saved_and_reopened_without_mixing(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            root = Path(folder)
            with patch.object(ui_workflow, "LIBRARY", root / "library"), \
                 patch.object(vector_store, "CHROMA_DIR", str(root / "vectors")), \
                 patch.object(vector_store, "get_embeddings", return_value=DeterministicFakeEmbedding(size=16)), \
                 patch.object(rag_engine, "get_llm", return_value=RunnableLambda(lambda _: AIMessage(content="Answer"))), \
                 patch.object(main, "process_input", return_value=["audio"]), \
                 patch.object(main, "transcribe_all", side_effect=["First video's unique content.", "Second video's unique content."]), \
                 patch.object(main, "generate_title", return_value="Test video"), \
                 patch.object(main, "summarize", return_value="Summary"), \
                 patch.object(main, "extract_meeting_details", return_value={"action_items": "None", "key_decisions": "None", "open_questions": "None"}), \
                 contextlib.redirect_stdout(io.StringIO()):
                events = []
                one = ui_workflow.process_video("first.mp4", "english", "first", lambda stage, data: events.append(stage))
                two = ui_workflow.process_video("second.mp4", "english", "second")
                self.assertNotEqual(one["collection_name"], two["collection_name"])
                self.assertEqual(len(ui_workflow.list_videos()), 2)
                reloaded = ui_workflow.open_video(one["id"])
                self.assertEqual(reloaded["transcript"], one["transcript"])
                documents = vector_store.load_vector_store(one["collection_name"]).get()["documents"]
                self.assertEqual(documents, ["First video's unique content."])
                self.assertLess(events.index("transcript"), events.index("indexing"))
                self.assertEqual(events[-1], "complete")
                snapshot = (root / "library" / f"{one['id']}.json").read_text()
                self.assertNotIn("rag_chain", snapshot)

    def test_library_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            ui_workflow.open_video("../.env")


if __name__ == "__main__":
    unittest.main()
