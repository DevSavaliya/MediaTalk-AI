"""Shared, bounded handling for the application's synchronous Mistral calls."""

import logging
import math
import os
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from langchain_mistralai import ChatMistralAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 2  # Three attempts total, including the original request.
MAX_RETRY_WAIT = 60.0
MIN_REQUEST_INTERVAL = 1.1  # Local pacing only; account limits may be lower.
_request_lock = threading.Lock()
_last_request_at = None


class MistralAPIError(RuntimeError):
    """An API failure with a safe, user-facing message."""


def create_llm(temperature=0.3, max_tokens=1024):
    key = os.environ.get("MISTRAL_API_KEY")
    if not key or not key.strip():
        raise MistralAPIError("Set MISTRAL_API_KEY in the project's .env file.")
    return ChatMistralAI(
        model=os.environ.get("MISTRAL_MODEL", "ministral-3b-latest"),
        mistral_api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=0,  # Only invoke_with_retry owns retries.
        timeout=30,
    )


def _pace_request():
    global _last_request_at
    with _request_lock:
        if _last_request_at is not None:
            delay = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
            if delay > 0:
                time.sleep(delay)
        _last_request_at = time.monotonic()


def _retry_after_seconds(value):
    """Accept delta seconds or an HTTP date; ignore malformed headers."""
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _explicit_quota_exhausted(response):
    # Code 1300 / "rate_limited" alone cannot distinguish a burst from a quota.
    try:
        body = response.json()
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    details = body.get("error", body)
    if not isinstance(details, dict):
        return False
    text = " ".join(str(details.get(k, "")) for k in ("code", "type", "message")).lower()
    return any(marker in text for marker in (
        "insufficient_quota", "quota_exceeded", "quota exhausted", "quota exceeded",
        "insufficient credits", "credit balance", "monthly token limit",
        "monthly quota", "spend limit", "billing limit",
    ))


def invoke_with_retry(chain, value):
    """Retry only the failed invocation, never replay a whole pipeline."""
    waited = 0.0
    for attempt in range(MAX_RETRIES + 1):
        _pace_request()
        try:
            return chain.invoke(value)
        except httpx.HTTPStatusError as exc:
            response = exc.response
            status = response.status_code
            if status != 429:
                if status == 401:
                    message = "Mistral rejected authentication (HTTP 401). Check the configured API key."
                elif status == 402:
                    message = "Mistral requires payment (HTTP 402). Check account billing."
                else:
                    message = f"Mistral request failed (HTTP {status}). No automatic retry was made."
                raise MistralAPIError(message) from None
            request_limit = response.headers.get("x-ratelimit-limit-req-minute", "").strip()
            if request_limit == "0":
                raise MistralAPIError(
                    "Mistral reports a request-per-minute limit of 0 (HTTP 429) for this model. "
                    "No retry was made. Check the model's access and organization limits "
                    "in Mistral Admin, or contact Mistral support."
                ) from None
            if _explicit_quota_exhausted(response):
                raise MistralAPIError(
                    "Mistral reports an exhausted quota or account spending limit (HTTP 429). "
                    "No retry was made. Check usage and limits in Mistral Admin."
                ) from None
            if attempt == MAX_RETRIES:
                raise MistralAPIError(
                    f"Mistral rate limit reached (HTTP 429) after {attempt + 1} attempts. "
                    "Stopped retrying. Wait before trying again and check your model's "
                    "request, token, and monthly limits in Mistral Admin. "
                    "This response alone does not identify which limit was reached."
                ) from None
            delay = 2.0 ** (attempt + 1) + random.uniform(0, 1)
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            if retry_after is not None:
                delay = max(delay, retry_after)
            if waited + delay > MAX_RETRY_WAIT:
                raise MistralAPIError(
                    "Mistral rate limit reached (HTTP 429). The required retry delay exceeds "
                    "the automatic wait budget of 60 seconds. Stopped without retrying early; "
                    "try again later and check account limits."
                ) from None
            logger.warning(
                "Mistral rate limit reached (HTTP 429); retry %d/%d in %.1f seconds.",
                attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            waited += delay
        except httpx.RequestError:
            raise MistralAPIError(
                "Could not reach Mistral. Check your connection and try again."
            ) from None
