"""Run with the project Python: python -m streamlit run streamlit_app.py."""

import os
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="MediaTalk AI · Your video workspace", page_icon=":material/video_library:", layout="wide")

from core import ui_workflow as workflow
from core.public_workflow import PublicWorkspace, MAX_UPLOAD_MB, MAX_QUESTION_CHARS, PublicDemoError

PUBLIC_DEMO = os.getenv("MEDIATALK_PUBLIC_DEMO") == "1"
if PUBLIC_DEMO:
    if "_public_workspace" not in st.session_state:
        st.session_state._public_workspace = PublicWorkspace()
        # Never import local results into a newly created public session.
        st.session_state.video_result = None
        st.session_state.chat_messages = []
        st.session_state.partial_transcript = ""
        st.session_state.workspace_error = None
    workflow = st.session_state._public_workspace
    st.query_params.pop("video", None)
    st.query_params.pop("view", None)

for key, value in {"video_result": None, "chat_messages": [], "partial_transcript": "", "workspace_error": None}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def activate(result):
    st.session_state.video_result = result
    st.session_state.chat_save_error = None
    if result is None:
        st.session_state.chat_messages = []
        st.query_params.pop("video", None)
        st.query_params.pop("view", None)
    else:
        if not PUBLIC_DEMO:
            st.query_params["video"] = result["id"]
        try:
            st.session_state.chat_messages = workflow.load_chat(result["id"])
        except (OSError, ValueError):
            st.session_state.chat_messages = []
            st.session_state.chat_save_error = "Your video is open, but the saved conversation could not be read."
    st.session_state.partial_transcript = ""
    st.session_state.workspace_error = None


def safe_error(exc):
    # Only our known user-facing errors may be displayed; never dump API responses.
    from core.mistral_client import MistralAPIError
    if isinstance(exc, (MistralAPIError, PublicDemoError)):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return "The file or saved index could not be found. Check the path or process the video again."
    if isinstance(exc, ValueError):
        return "The input or saved video could not be opened. Check your source and try again."
    return "Processing could not finish. Check that the media is accessible, FFmpeg is installed, and the transcription service is configured."


def persist_chat():
    try:
        workflow.save_chat(st.session_state.video_result["id"], st.session_state.chat_messages)
        st.session_state.chat_save_error = None
    except (OSError, ValueError):
        st.session_state.chat_save_error = "Your conversation is available in this session, but could not be saved to disk."


def remember_view(key):
    if not PUBLIC_DEMO:
        st.query_params["view"] = st.session_state[key]


# A browser reconnect/reload starts a fresh Session State. Restore the workspace
# from its URL rather than treating the next chat submission as a new video.
if not PUBLIC_DEMO and st.session_state.video_result is None and st.query_params.get("video"):
    with st.spinner("Restoring your video and conversation…"):
        try:
            video_id = st.query_params["video"]
            activate(workflow.open_cli_video() if video_id == "cli" else workflow.open_video(video_id))
        except Exception as exc:
            st.session_state.workspace_error = safe_error(exc)
elif not PUBLIC_DEMO and st.session_state.video_result is not None:
    st.query_params["video"] = st.session_state.video_result["id"]


with st.sidebar:
    st.title("MediaTalk AI", icon=":material/video_library:")
    st.caption("Watch less. Understand more.")
    st.space("small")
    if st.button("New video", icon=":material/add:", type="primary", width="stretch", key="new_video"):
        activate(None)
        st.rerun()
    st.subheader("This session" if PUBLIC_DEMO else "Your library")
    saved = workflow.list_videos()
    if saved:
        labels = {item["id"]: (item.get("title") or item.get("source_label") or "Untitled video") for item in saved}
        selected = st.selectbox("Saved videos", list(labels), format_func=labels.get, key="saved_video")
        if st.button("Open video", icon=":material/folder_open:", width="stretch", key="open_video"):
            with st.spinner("Opening your video…"):
                try:
                    activate(workflow.open_video(selected))
                    st.rerun()
                except Exception as exc:
                    st.session_state.workspace_error = safe_error(exc)
    else:
        st.caption("Processed videos will appear here, ready to revisit.")
    if not PUBLIC_DEMO and (workflow.ROOT / "vector_db" / "chroma.sqlite3").is_file():
        if st.button("Open saved CLI video", icon=":material/history:", width="stretch", key="open_cli"):
            with st.spinner("Loading your existing video index…"):
                try:
                    activate(workflow.open_cli_video())
                    st.rerun()
                except Exception as exc:
                    st.session_state.workspace_error = safe_error(exc)
    if PUBLIC_DEMO:
        st.caption(f"Recording attempts: {workflow.video_attempts}/3 · Question attempts: {workflow.question_attempts}/20")
        if st.button("Clear session data", icon=":material/delete:", key="clear_session"):
            workflow.clear()
            activate(None)
            st.session_state.workspace_error = None
            st.session_state._upload_generation = st.session_state.get("_upload_generation", 0) + 1
            st.session_state.pop("youtube_source", None)
            st.rerun()
    st.space("large")
    with st.expander("How it works", icon=":material/help_outline:"):
        sources = "Paste a YouTube link or upload media." if PUBLIC_DEMO else "Paste a YouTube link, upload media, or choose a local file."
        st.markdown(f"**1. Add a video**\n\n{sources}\n\n**2. Read the transcript**\n\nFollow transcription and analysis as they run.\n\n**3. Ask your video**\n\nGet answers grounded in its transcript.")
    if PUBLIC_DEMO:
        st.caption("Your workspace is separate from other visitors. Notes and chats last only for this browser session; refreshing or reconnecting may start a new session. Download anything you want to keep.")
        st.caption("Media is processed on the demo server. Transcript excerpts and questions go to Mistral; Hinglish audio goes to Sarvam. Use non-sensitive recordings. Temporary media is removed after processing.")
    else:
        st.caption("Files and notes stay on this computer. Transcript text is sent to your configured model for analysis and answers.")

with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    st.caption("YOUR VIDEO WORKSPACE")
    st.badge("Transcript-grounded answers", icon=":material/check_circle:", color="green")

if st.session_state.workspace_error:
    st.error(st.session_state.workspace_error)

if st.session_state.video_result is None:
    if st.query_params.get("video"):
        st.title("Your saved video could not be reopened")
        st.info("Choose a saved video in the sidebar, or select New video. Your existing files have not been changed.")
        st.stop()
    st.title("Turn a video into a conversation.")
    st.markdown("A readable transcript, the important takeaways, and a place to ask your next question.")
    st.space("small")
    # Wrap at the available content width, including when the sidebar is open.
    with st.container(horizontal=True, gap="medium"):
        entry = st.container(width=560)
        guide = st.container(width=320)
    with entry, st.container(border=True):
        st.subheader("Start with your video", icon=":material/add_circle_outline:")
        source_modes = ["YouTube link", "Upload a file"] if PUBLIC_DEMO else ["YouTube link", "Upload a file", "Local file"]
        mode = st.segmented_control("Video source", source_modes, default="YouTube link", key="source_mode", width="stretch")
        if mode not in source_modes:
            st.error("Choose one of the available video sources.")
            st.stop()
        with st.form("video_form", border=False):
            upload = None
            source = ""
            if mode == "Upload a file":
                upload_key = f"media_upload_{st.session_state.get('_upload_generation', 0)}" if PUBLIC_DEMO else "media_upload"
                upload = st.file_uploader("Video or audio file", type=[ext[1:] for ext in sorted(workflow.MEDIA_TYPES)], max_upload_size=MAX_UPLOAD_MB if PUBLIC_DEMO else 500, key=upload_key)
            elif mode == "Local file":
                source = st.text_input("Full file path", placeholder=r"D:\Videos\my-recording.mp4", key="local_source")
            else:
                source = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…", key="youtube_source")
            language = st.selectbox("Spoken language", ["English", "Hinglish"], key="spoken_language", help="English uses local Whisper. Hinglish uses your configured Sarvam transcription service.")
            submitted = st.form_submit_button("Transcribe & explore", icon=":material/arrow_forward:", type="primary", width="stretch", key="process_video")
        st.caption("MP4, WEBM, MOV, MKV, AVI, MP3, WAV, M4A or OGG · " + ("up to 50 MB and 10 minutes" if PUBLIC_DEMO else "up to 500 MB per upload"))
        if PUBLIC_DEMO:
            st.caption("Demo limits: 3 recording attempts and 20 questions per session. Shared hourly limits also apply. YouTube may block cloud downloads; file uploads remain available.")
    with guide:
        st.subheader("From playback to clarity")
        for icon, title, caption in [
            ("description", "Every word, captured", "Read and download the full transcript."),
            ("auto_awesome", "The takeaways that matter", "Explore a summary, action items, decisions, and open questions."),
            ("forum", "Go beyond the summary", "Ask specific questions using the video's own context."),
        ]:
            with st.container(border=True):
                st.markdown(f":material/{icon}: **{title}**")
                st.caption(caption)
    if submitted:
        source = source.strip().strip('"')
        error = None
        if mode == "Upload a file" and upload is None:
            error = "Choose a video or audio file first."
        elif mode == "Local file" and (not source or not Path(source).is_file()):
            error = "Enter the full path to an existing video or audio file."
        elif mode not in ("Upload a file", "Local file"):
            parsed = urlparse(source)
            if parsed.scheme not in ("http", "https") or (parsed.hostname or "").lower() not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}:
                error = "Enter a valid YouTube video URL."
        if error:
            st.error(error)
        else:
            st.session_state.partial_transcript = ""
            labels = {"audio": "Preparing audio", "transcribing": "Transcribing your video", "indexing": "Indexing the transcript for this session" if PUBLIC_DEMO else "Saving the searchable transcript", "title": "Finding a title", "summary": "Writing the summary", "details": "Extracting takeaways"}
            with st.status("Preparing your video…", expanded=True) as status:
                transcription_progress = st.empty()
                def progress(stage, data):
                    if stage in labels:
                        status.update(label=labels[stage] + "…")
                        st.caption(labels[stage])
                    if stage == "transcribing_progress":
                        label = "Reused transcript" if data["cached"] else "Transcribing"
                        transcription_progress.progress(data["fraction"], text=f"{label} · chunk {data['chunk']}/{data['chunks']} · {data['elapsed']:.0f}s elapsed")
                    if stage == "transcript":
                        st.session_state.partial_transcript = data["transcript"]
                        with st.expander("Your transcription is ready", expanded=False):
                            st.text(data["transcript"])
                try:
                    if upload is not None:
                        result = workflow.process_upload(upload, language.lower(), progress)
                    else:
                        label = Path(source).name if mode == "Local file" else source
                        result = workflow.process_video(source, language.lower(), label, progress)
                    activate(result)
                    status.update(label="Your video workspace is ready", state="complete", expanded=False)
                    st.rerun()
                except Exception as exc:
                    status.update(label="Processing paused", state="error")
                    st.error(safe_error(exc))
    if st.session_state.partial_transcript:
        with st.expander("Recovered transcript", expanded=True):
            st.text(st.session_state.partial_transcript)
            st.download_button("Download transcript", st.session_state.partial_transcript, "transcript.txt", mime="text/plain", on_click="ignore")
else:
    result = st.session_state.video_result
    st.title(result["title"])
    st.caption(result.get("source_label", "Your video"))
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge("Transcript ready", color="green", icon=":material/task_alt:")
        st.caption(f"{len(result['transcript'].split()):,} words · {result.get('language', 'English').capitalize()}")
        st.download_button("Download notes", workflow.export_notes(result), "video-notes.md", mime="text/markdown", icon=":material/download:", on_click="ignore", key="download_notes")
    if result.get("analysis_error"):
        st.warning("The transcript is ready, but some analysis could not be generated. " + result["analysis_error"])
    if result.get("save_error"):
        st.warning(result["save_error"])
    with st.container(horizontal=True, gap="medium"):
        reading = st.container(width=560)
        conversation = st.container(width=440)
    with reading:
        tab_key = f"video_view_{result['id']}"
        view = st.session_state.get(tab_key, "Transcript") if PUBLIC_DEMO else st.query_params.get("view", "Transcript")
        if view not in ("Transcript", "Summary", "Takeaways"):
            view = "Transcript"
        transcript_tab, summary_tab, takeaways_tab = st.tabs(
            ["Transcript", "Summary", "Takeaways"], key=tab_key, default=view,
            on_change=remember_view, args=(tab_key,))
        with transcript_tab:
            if result.get("recovered_chunks"):
                st.caption("Recovered from CLI transcript chunks. Overlapping passages may repeat; the CLI index can include earlier runs.")
            with st.container(height=520, border=True):
                st.text(result["transcript"])
            st.download_button("Download transcript", result["transcript"], "transcript.txt", mime="text/plain", icon=":material/download:", on_click="ignore", key="download_transcript")
        with summary_tab, st.container(border=True):
            st.subheader("The big picture", icon=":material/short_text:")
            st.markdown(result["summary"])
        with takeaways_tab:
            for field, title, icon in [("action_items", "Action items", "checklist"), ("key_decisions", "Key decisions", "flag"), ("open_questions", "Open questions", "help_outline")]:
                with st.container(border=True):
                    st.subheader(title, icon=f":material/{icon}:")
                    st.markdown(result[field])
    with conversation:
        with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="distribute"):
            st.subheader("Ask your video", icon=":material/forum:")
            if st.button("Clear chat", icon=":material/restart_alt:", key="clear_chat", disabled=not st.session_state.chat_messages):
                st.session_state.chat_messages = []
                persist_chat()
                st.rerun()
        st.caption("Ask about this video, then follow up. " + ("Your conversation stays in this session." if PUBLIC_DEMO else "Your conversation is saved with the video."))
        if st.session_state.get("chat_save_error"):
            st.warning(st.session_state.chat_save_error)
        messages = st.container(height=420, border=True, autoscroll=True)
        with messages:
            if not st.session_state.chat_messages:
                st.markdown("### A conversation with context")
                st.markdown("Ask about a claim, revisit a detail, or find what the speaker said.")
                st.caption("Try: “What is the main argument?” or “Which decisions were made?”")
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    if message.get("error"):
                        st.error(message["content"])
                    else:
                        st.markdown(message["content"])
        question = st.chat_input("Ask a question about this video…", key=f"question_{result['id']}", max_chars=MAX_QUESTION_CHARS if PUBLIC_DEMO else 4000, submit_mode="disable")
        if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user" and not question:
            st.info("The previous reply was interrupted. You can send the question again.")
        if question:
            history = list(st.session_state.chat_messages)
            st.session_state.chat_messages.append({"role": "user", "content": question})
            persist_chat()
            with messages:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Finding the relevant context…"):
                        try:
                            answer = workflow.answer_question(result["rag_chain"], question, history=history)
                            if not isinstance(answer, str) or not answer.strip():
                                raise ValueError("Empty model response")
                            st.markdown(answer)
                            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                        except Exception as exc:
                            from core.mistral_client import MistralAPIError
                            message = str(exc) if isinstance(exc, (MistralAPIError, PublicDemoError)) else "The answer could not be generated. Your video is still open; please try again."
                            st.error(message)
                            st.session_state.chat_messages.append({"role": "assistant", "content": message, "error": True})
            persist_chat()
            st.rerun()
        st.caption("The transcript records what was said; it does not independently verify the speaker's claims.")
