import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import audio_processor


class LocalMediaInputTests(unittest.TestCase):
    def test_links_and_missing_files_do_not_reach_decoder(self):
        with patch.object(audio_processor, "convert_to_wav") as convert:
            for source in ["https://youtu.be/abcdefghijk", "http://example.com/video.mp4", "file:///secret.wav"]:
                with self.subTest(source=source), self.assertRaisesRegex(ValueError, "Links are not supported"):
                    audio_processor.process_input(source)
            with self.assertRaises(FileNotFoundError):
                audio_processor.process_input("missing-recording.wav")
            convert.assert_not_called()

    def test_quoted_local_path_prepares_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "my recording.mp4"
            source.write_bytes(b"mock media")
            with patch.object(audio_processor, "convert_to_wav", return_value="audio.wav") as convert, \
                 patch.object(audio_processor, "chunk_audio", return_value=["chunk.wav"]) as chunks:
                self.assertEqual(audio_processor.process_input(f'"{source}"'), ["chunk.wav"])
                convert.assert_called_once_with(str(source))
                chunks.assert_called_once_with("audio.wav")
