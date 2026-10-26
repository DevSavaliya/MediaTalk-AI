import argparse
from pathlib import Path
from dotenv import load_dotenv

# Resolve configuration relative to this file, regardless of the working directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_meeting_details
from core.rag_engine import build_rag_chain, load_rag_chain, ask_question
from core.mistral_client import MistralAPIError


def run_pipeline(source: str, language: str = "english", *, on_progress=None, collection_name=None) -> dict:
    def report(stage, **data):
        if on_progress is not None:
            on_progress(stage, data)

    print("starting AI Video Assistant")

    report("audio")
    chunks = process_input(source)

    report("transcribing")
    transcript = transcribe_all(chunks, language, on_progress=lambda data: report("transcribing_progress", **data))
    report("transcript", transcript=transcript)
    print("\n" + "=" * 60)
    print("TRANSCRIPTION")
    print("=" * 60)
    print(transcript)
    print("=" * 60 + "\n")

    # Persist the local transcript index before any remote generation can fail.
    report("indexing")
    rag_chain = (build_rag_chain(transcript, collection_name=collection_name)
                 if collection_name else build_rag_chain(transcript))

    result = {
        "title": "Video transcript (title not generated)",
        "transcript": transcript,
        "summary": "Not generated.",
        "action_items": "Not generated.",
        "key_decisions": "Not generated.",
        "open_questions": "Not generated.",
        "rag_chain": rag_chain,
        "analysis_error": None,
    }

    try:
        report("title")
        print("Generating title...")
        result["title"] = generate_title(transcript)
        report("summary")
        print("Generating summary...")
        result["summary"] = summarize(transcript)
        report("details")
        print("Extracting action items, decisions, and questions...")
        result.update(extract_meeting_details(transcript))
    except MistralAPIError as exc:
        # Keep the saved index and successful outputs; analysis is not a chat prerequisite.
        result["analysis_error"] = str(exc)
    report("complete", analysis_error=result["analysis_error"])
    return result


def display_results(result):
    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)
    if result.get("analysis_error"):
        print(f"\nAnalysis unavailable: {result['analysis_error']}")
        print("Your video is saved. Chat is open, but generated answers require working model access.")


def chat_loop(rag_chain):
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    print("You can reopen this saved video later with: python main.py --chat\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        try:
            answer = ask_question(rag_chain, question)
        except MistralAPIError as exc:
            print(f"\nUnable to generate an answer: {exc}")
            print("The transcript is saved, but the model did not answer. "
                  "Resolve model access before asking again, or type 'exit'.")
            continue
        print(f"\n🤖 Assistant: {answer}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a video and chat with its saved transcript.")
    parser.add_argument("--chat", action="store_true", help="Chat with the saved video without retranscribing it.")
    args = parser.parse_args(argv)
    try:
        if args.chat:
            print("Opening saved video...")
            rag_chain = load_rag_chain()
        else:
            source = input("Enter YouTube URL or local file path (or 'chat' for the saved video): ").strip()
            if source.lower() == "chat":
                rag_chain = load_rag_chain()
            else:
                language = input("Language (english/hinglish): ").strip() or "english"
                result = run_pipeline(source, language)
                display_results(result)
                rag_chain = result["rag_chain"]
    except (MistralAPIError, FileNotFoundError, ValueError) as exc:
        print(f"\nVideo Agent could not start: {exc}")
        return 1
    chat_loop(rag_chain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
