import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import transcriber
from utils.audio_processor import chunk_audio
from pydub import AudioSegment


class TranscriptionOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "audio.wav"
        self.audio.write_bytes(b"first audio")
        patch.object(transcriber, "CACHE_DIR", self.root / "cache").start()
        patch.dict("os.environ", {"TRANSCRIPTION_CACHE": "1"}).start()
        self.addCleanup(patch.stopall)

    def test_repeated_audio_reuses_transcript_across_paths(self):
        duplicate = self.root / "uploaded-again.wav"
        duplicate.write_bytes(self.audio.read_bytes())
        events = []
        with patch.object(transcriber, "transcribe_chunk", return_value="A real transcript.") as transcribe:
            first = transcriber.transcribe_all([str(self.audio)], on_progress=events.append)
            again = transcriber.transcribe_all([str(duplicate)], on_progress=events.append)
        self.assertEqual(first, again)
        transcribe.assert_called_once()
        self.assertTrue(events[-1]["cached"])
        self.assertEqual(events[-1]["fraction"], 1)

    def test_content_language_model_and_engine_invalidate_cache(self):
        with patch.object(transcriber, "transcribe_chunk", return_value="Transcript") as transcribe:
            transcriber.transcribe_all([str(self.audio)])
            self.audio.write_bytes(b"different audio")
            transcriber.transcribe_all([str(self.audio)])
            transcriber.transcribe_all([str(self.audio)], language="hinglish")
            with patch.object(transcriber, "WHISPER_MODEL", "base"):
                transcriber.transcribe_all([str(self.audio)])
            with patch.object(transcriber, "WHISPER_ENGINE", "openai"):
                transcriber.transcribe_all([str(self.audio)])
        self.assertEqual(transcribe.call_count, 5)

    def test_partial_failure_resumes_completed_chunks(self):
        second = self.root / "second.wav"
        second.write_bytes(b"second audio")
        with patch.object(transcriber, "transcribe_chunk", side_effect=["First.", RuntimeError("unavailable")]):
            with self.assertRaises(RuntimeError):
                transcriber.transcribe_all([str(self.audio), str(second)])
        with patch.object(transcriber, "transcribe_chunk", return_value="Second.") as transcribe:
            result = transcriber.transcribe_all([str(self.audio), str(second)])
        self.assertEqual(result, "First. Second.")
        self.assertEqual(transcribe.call_count, 1)
        self.assertEqual(transcribe.call_args.args[0], str(second))

    def test_empty_and_corrupt_cache_are_not_reused(self):
        with patch.object(transcriber, "transcribe_chunk", side_effect=["", "Recovered", "New text"]) as transcribe:
            transcriber.transcribe_all([str(self.audio)])
            transcriber.transcribe_all([str(self.audio)])
            transcriber._cache_path(self.audio, "english").write_text("broken json")
            self.assertEqual(transcriber.transcribe_all([str(self.audio)]), "New text")
        self.assertEqual(transcribe.call_count, 3)

    def test_cache_write_failure_preserves_result(self):
        with patch.object(transcriber, "transcribe_chunk", return_value="Keep this"), \
             patch.object(Path, "mkdir", side_effect=PermissionError):
            self.assertEqual(transcriber.transcribe_all([str(self.audio)]), "Keep this")

    def test_cache_can_be_disabled(self):
        with patch.dict("os.environ", {"TRANSCRIPTION_CACHE": "0"}), \
             patch.object(transcriber, "transcribe_chunk", return_value="Text") as transcribe:
            transcriber.transcribe_all([str(self.audio)])
            transcriber.transcribe_all([str(self.audio)])
        self.assertEqual(transcribe.call_count, 2)
        self.assertFalse((self.root / "cache").exists())

    def test_faster_generator_is_consumed_and_progress_reported(self):
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(text=" First. ", end=10),
                                              SimpleNamespace(text=" Second. ", end=20)]),
                                         SimpleNamespace(duration=20))
        progress = []
        with patch.object(transcriber, "load_model", return_value=model), \
             patch.object(transcriber, "WHISPER_ENGINE", "faster"):
            text = transcriber.transcribe_chunk_whisper("audio.wav", progress.append)
        self.assertEqual(text, "First. Second.")
        self.assertEqual(progress, [0.5, 1.0])
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "en")
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])

    def test_short_audio_is_not_copied_and_long_audio_remains_complete(self):
        path = self.root / "valid.wav"
        AudioSegment.silent(duration=1250).export(path, format="wav").close()
        self.assertEqual(chunk_audio(str(path)), [str(path)])
        AudioSegment.silent(duration=65000).export(path, format="wav").close()
        chunks = chunk_audio(str(path), chunk_minutes=1)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(sum(len(AudioSegment.from_wav(p)) for p in chunks), 65000)

    def test_installed_audio_decoder_accepts_real_wav(self):
        # Catch incompatible PyAV releases before a full model is downloaded.
        from faster_whisper.audio import decode_audio
        path = self.root / "decoder.wav"
        AudioSegment.silent(duration=500, frame_rate=16000).export(path, format="wav").close()
        decoded = decode_audio(str(path), sampling_rate=16000)
        self.assertEqual(len(decoded), 8000)


if __name__ == "__main__":
    unittest.main()
