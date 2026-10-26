# Mistral rate limits in Video Agent

The application defaults to `ministral-3b-latest` for all text generation.
It returned a successful live completion with the existing API key, while
`mistral-small-latest` reported zero request capacity. To choose a different
model that your account can use, set `MISTRAL_MODEL` in `.env`.
`main.py` loads the project-local `.env`; existing process environment values
retain precedence. No API credentials belong in source code or diagnostics.

## Requests per pipeline run

Let N be the number of nonempty summary chunks produced by the text splitter
(3,000 characters, with 200 characters of overlap; these are not token counts).

| Operation | Previously | Now |
| --- | --- | --- |
| Title (first 2,000 transcript characters) | 1 | 1 |
| Summary of each chunk | N | N |
| Combine chunk summaries | 1 | 1 only when N > 1 |
| Action items, decisions, questions | 3 | 1 combined JSON response |
| Create local vector store / RAG chain | 0 Mistral calls | 0 Mistral calls |
| Each subsequent chat question | 1 | 1 |

For a nonempty, single-chunk transcript, this reduces six requests to three.
For multiple chunks, it reduces N + 5 requests to N + 3, before retries.
The individual extraction functions remain available for callers that need
only one section.

Extraction still receives the full transcript, now once instead of three
times. No transcript text is silently discarded. Very long transcripts can
still use significant input tokens or exceed a model's context limit.
Completion budgets are 64 tokens for titles, 1,024 for summaries/chat,
and 2,048 for extraction. These are output ceilings, not guaranteed usage;
especially dense transcripts may need a larger extraction budget.

## Retry behavior

`core/mistral_client.py` owns retries for application calls:

- At most two retries after the initial request (three attempts total).
- Only HTTP 429 is retried. Explicit exhausted-quota / spending-limit errors
  and a reported request-per-minute limit of zero stop immediately. Generic
  code 1300 does not identify the specific limit.
- Exponential delays of 2 and 4 seconds, plus 0–1 second of random jitter.
- `Retry-After` supports seconds and HTTP dates, and acts as a minimum delay.
  If the requested delay would exceed the cumulative 60-second retry-sleep
  budget, execution stops instead of retrying before the server permits it.
- Requests within this process are spaced at least 1.1 seconds apart.
  This is conservative local pacing, not a guarantee about organization limits
  and not coordination with other running processes.
- LangChain's built-in retries are disabled to avoid layered retry loops.
- Only the failed operation is retried; completed summary chunks are not replayed.
- Each HTTP call has a 30-second timeout. The retry-sleep budget is not a
  wall-clock deadline for the entire operation.

If remote analysis fails after indexing, the CLI displays any completed results,
marks the remaining sections "Not generated.", and still opens the chat prompt.
A failed chat question reports the error and returns to the question prompt.
This keeps the interface accessible; it does not bypass a provider's quota.
Startup failures (such as a missing database in saved-video mode) exit with
a concise error and status 1.
Generated text stages are not checkpointed across separate program runs.
The local transcript index is saved before remote text generation, at
`vector_db/chroma.sqlite3` under the project directory. A Mistral error therefore
does not prevent local database creation after a successful transcription and
embedding step.

## Open chat with the saved video

```powershell
.\.venv\Scripts\python.exe main.py --chat
```

This loads the existing transcript index, skips audio processing and automatic
analysis, and opens `You:` immediately after loading the embedding model.
Alternatively, run normally and enter `chat` at the source prompt.
Type a question about the video, or `exit` to quit.
The database provides retrieved context; a working Mistral connection is still
required to generate an answer. Unavailable summaries are not fabricated.

## When an account action is needed

A tiny standalone request to the original `mistral-small-latest` model returned HTTP 429/code 1300 during investigation,
without running the pipeline. Reducing pipeline calls cannot by itself resolve
that rejection. A generic 429 does not establish that the key is invalid or
which quota is exhausted.

On October 20, 2026, three live title attempts also returned
`x-ratelimit-limit-req-minute: 0` and
`x-ratelimit-remaining-req-minute: 0`, without `Retry-After`.
This is evidence that Mistral was reporting zero request capacity for that model.
It does not explain why the limit is zero; access, plan, or server-side
configuration needs checking in the account. The application now stops on that
response without retrying.

A later test with the same key and `ministral-3b-latest` returned HTTP 200.
The project now defaults to this working model. Restart an already running
Python/chat session to pick up this change; its existing client retains the
previous model. Meeting-detail extraction uses an enforced JSON schema so the
three result sections have the types expected by the application.

Check the model's request, token, and monthly limits and current organization
usage in Mistral Admin. Other applications/workspaces may share those limits.
Wait for the relevant limit to reset or contact Mistral support if a minimal
request remains blocked. Changing the model or buying credits is not an
automatic remedy.

References:
- [Mistral's rate-limit explanation](https://help.mistral.ai/en/articles/698531-why-am-i-hitting-api-rate-limits-and-how-do-i-increase-them)
- [Mistral's error and retry guidance](https://docs.mistral.ai/resources/error-glossary)

## Offline verification

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite uses mock HTTP transport and does not send API requests. It covers
transient and persistent 429s, quota detection, retry headers, retry budgets,
request counts, combined extraction, and CLI failure handling. Passing these
tests does not prove live account access.
