import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from core import rag_engine, ui_workflow


class ChatMemoryTests(unittest.TestCase):
    def test_followup_retrieval_and_prompt_include_previous_turn(self):
        queries, prompts = [], []
        def retrieve(query):
            queries.append(query)
            return [Document(page_content="Alex tests on Wednesday before Friday's launch.")]
        def respond(prompt):
            prompts.append(prompt.to_messages())
            return AIMessage(content="Wednesday.")
        history = [{"role": "user", "content": "Who tests the launch?"},
                   {"role": "assistant", "content": "Alex."}]
        with patch.object(rag_engine, "get_retriever", return_value=RunnableLambda(retrieve)), \
             patch.object(rag_engine, "get_llm", return_value=RunnableLambda(respond)):
            chain = rag_engine._create_rag_chain(object())
            self.assertEqual(chain.invoke({"question": "When does he do it?", "history": history}), "Wednesday.")
        self.assertIn("Alex", queries[0])
        self.assertEqual(prompts[0][-3].content, "Who tests the launch?")
        self.assertEqual(prompts[0][-2].content, "Alex.")
        self.assertEqual(prompts[0][-1].content, "When does he do it?")
        self.assertIn("Alex tests on Wednesday", prompts[0][0].content)
        self.assertEqual(len(prompts), 1)

    def test_failed_turns_are_excluded_and_history_is_bounded(self):
        messages = [{"role": "user", "content": "Unanswered"},
                    {"role": "assistant", "content": "HTTP error", "error": True}]
        for number in range(6):
            messages.extend([{"role": "user", "content": f"Question {number}"},
                             {"role": "assistant", "content": "A" * 4000}])
        messages.append({"role": "user", "content": "Still pending"})
        history = rag_engine.conversation_history(messages)
        self.assertEqual(len(history), 8)
        self.assertEqual(history[0], ("human", "Question 2"))
        self.assertEqual(len(history[1][1]), 2000)

    def test_saved_chats_are_isolated_and_validate_paths(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ui_workflow, "CHAT_DIR", Path(folder)):
            first = [{"role": "user", "content": "First video"}]
            second = [{"role": "assistant", "content": "Second video"}]
            ui_workflow.save_chat("a" * 32, first)
            ui_workflow.save_chat("b" * 32, second)
            self.assertEqual(ui_workflow.load_chat("a" * 32), first)
            self.assertEqual(ui_workflow.load_chat("b" * 32), second)
            with self.assertRaises(ValueError):
                ui_workflow.load_chat("../.env")
            ui_workflow.save_chat("a" * 32, [])
            self.assertEqual(ui_workflow.load_chat("a" * 32), [])
            self.assertEqual(ui_workflow.load_chat("b" * 32), second)
