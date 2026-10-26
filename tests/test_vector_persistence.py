import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import main
from core import rag_engine, vector_store
from core.mistral_client import MistralAPIError


class VectorPersistenceTests(unittest.TestCase):
    def setUp(self):
        # Chroma/SQLite are real; offline test embeddings avoid model downloads.
        self.folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.folder.cleanup)
        self.db_path = Path(self.folder.name) / "vector_db"
        patch.object(vector_store, "CHROMA_DIR", str(self.db_path)).start()
        patch.object(vector_store, "get_embeddings",
                     return_value=DeterministicFakeEmbedding(size=16)).start()
        self.addCleanup(patch.stopall)

    def test_database_is_persisted_and_retrievable(self):
        transcript = "Alex will test the video assistant on Wednesday."
        with contextlib.redirect_stdout(io.StringIO()):
            vector_store.build_vector_store(transcript)
        sqlite_file = self.db_path / "chroma.sqlite3"
        self.assertTrue(sqlite_file.is_file())
        with contextlib.closing(sqlite3.connect(str(sqlite_file))) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertGreater(connection.execute("SELECT count(*) FROM embeddings").fetchone()[0], 0)
        restored = vector_store.load_vector_store()
        results = vector_store.get_retriever(restored).invoke(transcript)
        self.assertEqual(results[0].page_content, transcript)

    def test_mistral_failure_does_not_prevent_persistence(self):
        llm = RunnableLambda(lambda prompt: AIMessage(content="Test response"))
        with patch.object(main, "process_input", return_value=["test.wav"]), \
             patch.object(main, "transcribe_all", return_value="Launch the video assistant Friday."), \
             patch.object(rag_engine, "get_llm", return_value=llm), \
             patch.object(main, "generate_title", side_effect=MistralAPIError("HTTP 429")), \
             patch.object(main, "summarize") as summarize, \
             contextlib.redirect_stdout(io.StringIO()):
            result = main.run_pipeline("test.mp4")
        self.assertEqual(result["analysis_error"], "HTTP 429")
        self.assertIsNotNone(result["rag_chain"])
        self.assertTrue((self.db_path / "chroma.sqlite3").is_file())
        self.assertEqual(vector_store.load_vector_store()._collection.count(), 1)
        summarize.assert_not_called()

    def test_load_rag_chain_uses_saved_store(self):
        transcript = "Testing is scheduled for Wednesday."
        with contextlib.redirect_stdout(io.StringIO()):
            vector_store.build_vector_store(transcript)
        prompts = []

        def answer(prompt):
            prompts.append(prompt.to_string())
            return AIMessage(content="Wednesday")

        with patch.object(rag_engine, "get_llm", return_value=RunnableLambda(answer)):
            chain = rag_engine.load_rag_chain()
            self.assertEqual(chain.invoke("When is testing?"), "Wednesday")
        self.assertIn(transcript, prompts[0])

    def test_empty_transcript_does_not_create_database(self):
        with self.assertRaisesRegex(ValueError, "empty transcript"):
            vector_store.build_vector_store("  ")
        self.assertFalse(self.db_path.exists())

    def test_load_missing_database_does_not_create_empty_one(self):
        with self.assertRaisesRegex(FileNotFoundError, "No saved video"):
            vector_store.load_vector_store()
        self.assertFalse(self.db_path.exists())


class DatabaseLocationTests(unittest.TestCase):
    def test_default_path_is_under_project_root(self):
        expected = Path(main.__file__).resolve().parent / "vector_db"
        self.assertEqual(Path(vector_store.CHROMA_DIR), expected)
        self.assertTrue(Path(vector_store.CHROMA_DIR).is_absolute())


if __name__ == "__main__":
    unittest.main()
