import contextlib
import io
import json
import os
import runpy
import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

import main
from core import extractor, mistral_client, rag_engine, summarizer


def http_error(status=429, headers=None, body=None):
    request = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    response = httpx.Response(
        status, request=request, headers=headers,
        json=body if body is not None else {"message": "Rate limit exceeded", "code": "1300"},
    )
    return httpx.HTTPStatusError("Test API error", request=request, response=response)


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.pacing = patch.object(mistral_client, "_pace_request").start()
        self.sleep = patch.object(mistral_client.time, "sleep").start()
        patch.object(mistral_client.random, "uniform", return_value=0.5).start()
        self.addCleanup(patch.stopall)

    def test_temporary_429_recovers_with_exponential_backoff(self):
        chain = Mock()
        chain.invoke.side_effect = [http_error(), http_error(), "success"]
        with self.assertLogs(mistral_client.logger, level="WARNING"):
            self.assertEqual(mistral_client.invoke_with_retry(chain, "input"), "success")
        self.assertEqual(chain.invoke.call_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.call_args_list], [2.5, 4.5])

    def test_persistent_429_stops_after_three_attempts(self):
        chain = Mock()
        chain.invoke.side_effect = http_error()
        with self.assertLogs(mistral_client.logger, level="WARNING"):
            with self.assertRaisesRegex(mistral_client.MistralAPIError, "after 3 attempts"):
                mistral_client.invoke_with_retry(chain, "input")
        self.assertEqual(chain.invoke.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)

    def test_retry_after_seconds_is_respected(self):
        chain = Mock()
        chain.invoke.side_effect = [http_error(headers={"Retry-After": "10"}), "ok"]
        with self.assertLogs(mistral_client.logger, level="WARNING"):
            self.assertEqual(mistral_client.invoke_with_retry(chain, "input"), "ok")
        self.sleep.assert_called_once_with(10.0)

    def test_retry_after_http_date(self):
        date = datetime.now(timezone.utc) + timedelta(seconds=20)
        delay = mistral_client._retry_after_seconds(format_datetime(date, usegmt=True))
        self.assertGreater(delay, 18)
        self.assertLessEqual(delay, 20)

    def test_invalid_retry_after_headers(self):
        for value in [None, "", "invalid", "nan", "inf"]:
            with self.subTest(value=value):
                self.assertIsNone(mistral_client._retry_after_seconds(value))
        self.assertEqual(mistral_client._retry_after_seconds("-1"), 0)

    def test_long_retry_after_stops_without_retrying_early(self):
        chain = Mock()
        chain.invoke.side_effect = http_error(headers={"Retry-After": "120"})
        with self.assertRaisesRegex(mistral_client.MistralAPIError, "wait budget"):
            mistral_client.invoke_with_retry(chain, "input")
        chain.invoke.assert_called_once()
        self.sleep.assert_not_called()

    def test_cumulative_retry_wait_is_bounded(self):
        chain = Mock()
        chain.invoke.side_effect = http_error(headers={"Retry-After": "40"})
        with self.assertLogs(mistral_client.logger, level="WARNING"):
            with self.assertRaisesRegex(mistral_client.MistralAPIError, "wait budget"):
                mistral_client.invoke_with_retry(chain, "input")
        self.assertEqual(chain.invoke.call_count, 2)
        self.sleep.assert_called_once_with(40.0)

    def test_explicit_quota_exhaustion_is_not_retried(self):
        chain = Mock()
        chain.invoke.side_effect = http_error(body={"error": {"code": "insufficient_quota"}})
        with self.assertRaisesRegex(mistral_client.MistralAPIError, "exhausted quota"):
            mistral_client.invoke_with_retry(chain, "input")
        chain.invoke.assert_called_once()
        self.sleep.assert_not_called()

    def test_zero_request_capacity_is_not_retried(self):
        chain = Mock()
        chain.invoke.side_effect = http_error(headers={
            "x-ratelimit-limit-req-minute": "0",
            "x-ratelimit-remaining-req-minute": "0",
        })
        with self.assertRaisesRegex(mistral_client.MistralAPIError, "limit of 0"):
            mistral_client.invoke_with_retry(chain, "input")
        chain.invoke.assert_called_once()
        self.sleep.assert_not_called()

    def test_exhausted_positive_request_limit_can_retry(self):
        chain = Mock()
        chain.invoke.side_effect = [http_error(headers={
            "x-ratelimit-limit-req-minute": "10",
            "x-ratelimit-remaining-req-minute": "0",
        }), "ok"]
        with self.assertLogs(mistral_client.logger, level="WARNING"):
            self.assertEqual(mistral_client.invoke_with_retry(chain, "input"), "ok")
        self.assertEqual(chain.invoke.call_count, 2)

    def test_other_http_errors_are_not_retried_or_exposed(self):
        for status in [400, 401, 402, 403, 500]:
            with self.subTest(status=status):
                chain = Mock()
                chain.invoke.side_effect = http_error(status, body={"message": "private-response"})
                with self.assertRaises(mistral_client.MistralAPIError) as caught:
                    mistral_client.invoke_with_retry(chain, "input")
                self.assertIn(str(status), str(caught.exception))
                self.assertNotIn("private-response", str(caught.exception))
                chain.invoke.assert_called_once()
        self.sleep.assert_not_called()

    def test_network_error_is_safe_and_not_retried(self):
        chain = Mock()
        chain.invoke.side_effect = httpx.ConnectError("private-connection-detail")
        with self.assertRaisesRegex(mistral_client.MistralAPIError, "Could not reach"):
            mistral_client.invoke_with_retry(chain, "input")
        chain.invoke.assert_called_once()


class IntegrationTests(unittest.TestCase):
    """Real LangChain chains and HTTP serialization; no live API requests."""

    def setUp(self):
        self.requests = []
        self.responses = []
        self.clients = []
        patch.dict(os.environ, {"MISTRAL_API_KEY": "test-only-key"}).start()
        patch.object(mistral_client, "_pace_request").start()
        patch.object(mistral_client.time, "sleep").start()
        real_class = mistral_client.ChatMistralAI

        def factory(**kwargs):
            self.assertEqual(kwargs["max_retries"], 0)
            llm = real_class(**kwargs)
            llm.client.close()
            llm.client = httpx.Client(
                base_url=llm.endpoint,
                headers={"Authorization": "Bearer " + kwargs["mistral_api_key"]},
                transport=httpx.MockTransport(self.handle_request),
            )
            self.clients.append(llm.client)
            return llm

        patch.object(mistral_client, "ChatMistralAI", side_effect=factory).start()
        self.addCleanup(patch.stopall)
        self.addCleanup(lambda: [client.close() for client in self.clients])

    def handle_request(self, request):
        self.assertEqual(request.headers["Authorization"], "Bearer test-only-key")
        body = json.loads(request.content)
        self.requests.append(body)
        if self.responses:
            response = self.responses.pop(0)
            if response is not None:
                return response
        content = "Video Assistant Launch Planning"
        if body.get("response_format", {}).get("type") == "json_schema":
            content = json.dumps({
                "action_items": "1. Alex: finish testing by Wednesday.",
                "key_decisions": "1. Launch next Friday.",
                "open_questions": "No open questions found.",
            })
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "mistral-small-latest",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    def test_title_recovery_and_output_budget(self):
        self.responses = [httpx.Response(429, json={"code": "1300"}), None]
        with self.assertLogs(mistral_client.logger, level="WARNING"):
            title = summarizer.generate_title("A" * 2500)
        self.assertEqual(title, "Video Assistant Launch Planning")
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0], self.requests[1])
        self.assertEqual(self.requests[0]["max_tokens"], 64)
        self.assertEqual(len(self.requests[0]["messages"][-1]["content"]), 2000)

    def test_retry_does_not_replay_successful_summary_chunks(self):
        self.responses = [None, httpx.Response(429, json={"code": "1300"}), None, None]
        with patch.object(summarizer, "split_transcript", return_value=["first", "second"]):
            with self.assertLogs(mistral_client.logger, level="WARNING"):
                summarizer.summarize("transcript")
        self.assertEqual(len(self.requests), 4)
        inputs = [r["messages"][-1]["content"] for r in self.requests]
        self.assertEqual(inputs[:3], ["first", "second", "second"])

    def test_single_chunk_summary_uses_one_request(self):
        self.assertTrue(summarizer.summarize("A short transcript"))
        self.assertEqual(len(self.requests), 1)

    def test_empty_summary_uses_no_requests(self):
        self.assertEqual(summarizer.summarize(""), "No transcript content to summarize.")
        self.assertEqual(len(self.requests), 0)

    def test_combined_extraction_preserves_sections_in_one_request(self):
        result = extractor.extract_meeting_details("Alex will finish testing by Wednesday.")
        self.assertEqual(set(result), {"action_items", "key_decisions", "open_questions"})
        self.assertIn("Alex", result["action_items"])
        self.assertEqual(len(self.requests), 1)

    def test_invalid_extraction_is_not_retried(self):
        self.responses = [httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "{}"},
                         "finish_reason": "stop"}],
        })]
        with self.assertRaisesRegex(mistral_client.MistralAPIError, "invalid meeting details"):
            extractor.extract_meeting_details("transcript")
        self.assertEqual(len(self.requests), 1)

    def test_pipeline_uses_three_requests_for_short_transcript(self):
        with patch.object(main, "process_input", return_value=["audio.wav"]), \
             patch.object(main, "transcribe_all", return_value="Launch next Friday."), \
             patch.object(main, "build_rag_chain", return_value="retriever"), \
             contextlib.redirect_stdout(io.StringIO()):
            result = main.run_pipeline("video.mp4")
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(set(result), {
            "title", "transcript", "summary", "action_items",
            "key_decisions", "open_questions", "rag_chain", "analysis_error",
        })

    def test_chat_uses_same_retry_policy(self):
        chain = Mock()
        chain.invoke.side_effect = [http_error(), "Answer"]
        with self.assertLogs(mistral_client.logger, level="WARNING"), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rag_engine.ask_question(chain, "Question"), "Answer")
        self.assertEqual(chain.invoke.call_count, 2)

    def test_cli_analysis_rate_limit_still_opens_chat(self):
        self.responses = [httpx.Response(429, json={"code": "1300"}) for _ in range(3)]
        output = io.StringIO()
        with patch("builtins.input", side_effect=["video.mp4", "english", "exit"]) as user_input, \
             patch.object(sys, "argv", ["main.py"]), \
             patch("utils.audio_processor.process_input", return_value=["audio.wav"]), \
             patch("core.transcriber.transcribe_all", return_value="A short transcript"), \
             patch("core.rag_engine.build_rag_chain", return_value="mock-rag-chain"), \
             contextlib.redirect_stdout(output), \
             self.assertLogs(mistral_client.logger, level="WARNING"):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(Path(main.__file__)), run_name="__main__")
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("Analysis unavailable:", output.getvalue())
        self.assertIn("Chat with your meeting", output.getvalue())
        self.assertEqual(user_input.call_args.args[0], "You: ")
        self.assertNotIn("Traceback", output.getvalue())
        self.assertEqual(len(self.requests), 3)


class ConfigurationTests(unittest.TestCase):
    def test_missing_key_fails_locally(self):
        with patch.dict(os.environ, {"MISTRAL_API_KEY": ""}):
            with self.assertRaisesRegex(mistral_client.MistralAPIError, "MISTRAL_API_KEY"):
                mistral_client.create_llm()

    def test_pacing_applies_between_calls(self):
        with patch.object(mistral_client, "_last_request_at", 100.0), \
             patch.object(mistral_client.time, "monotonic", side_effect=[100.1, 101.1]), \
             patch.object(mistral_client.time, "sleep") as sleep:
            mistral_client._pace_request()
        self.assertAlmostEqual(sleep.call_args.args[0], 1.0)


if __name__ == "__main__":
    unittest.main()
