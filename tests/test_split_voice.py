import json
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from src.main import app
from src.split_voices.synthesize import (
    load_segments,
    load_clone_map,
    load_voice_map,
    normalize_audio,
    resolve_voice_map_path,
)
from src.story_paths import get_story_name, get_story_output_root


class SplitVoiceCLITests(unittest.TestCase):
    def test_story_output_paths_are_grouped_by_story(self):
        story = Path("/tmp/input/hai_dua_tre")
        self.assertEqual(get_story_name(story), "hai_dua_tre")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "hai_dua_tre" / "media" / "hai_dua_tre.mp3"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"audio")
            self.assertEqual(get_story_name(audio), "hai_dua_tre")
            self.assertTrue(get_story_output_root(audio).name == "hai_dua_tre")

    def test_character_synth_helpers_load_segment_and_voice_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segments = root / "segments.json"
            segments.write_text(json.dumps({
                "segments": [
                    {"speaker": "narrator", "text": "Lời dẫn.", "pause_after_ms": 400},
                    {"speaker": "Liên", "text": "Xin chào.", "pause_after_ms": 700},
                ]
            }), encoding="utf-8")
            voices = root / "voices.json"
            voices.write_text(json.dumps({
                "narrator": "Hải Đăng",
                "Liên": "Mai Anh",
            }), encoding="utf-8")

            loaded_segments = load_segments(segments)
            loaded_voices = load_voice_map(voices)

            self.assertEqual([item["speaker"] for item in loaded_segments], ["narrator", "Liên"])
            self.assertEqual(loaded_voices["narrator"], "Hải Đăng")
            self.assertEqual(loaded_voices["Liên"], "Mai Anh")

    def test_character_voice_map_supports_clone_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voices = root / "voices.json"
            (root / "lien.wav").write_bytes(b"RIFF" + b"\0" * 2000)
            voices.write_text(json.dumps({
                "narrator": "Hải Đăng",
                "Liên": {
                    "voice": "clone-lien",
                    "ref_audio": "lien.wav",
                    "ref_text": "Xin chào",
                },
            }), encoding="utf-8")

            loaded_voices = load_voice_map(voices)
            loaded_clones = load_clone_map(voices)

            self.assertEqual(loaded_voices["narrator"], "Hải Đăng")
            self.assertEqual(loaded_voices["Liên"], "clone-lien")
            self.assertEqual(loaded_clones["Liên"]["name"], "clone-lien")
            self.assertEqual(
                loaded_clones["Liên"]["ref_audio"],
                str((root / "lien.wav").resolve()),
            )
            self.assertEqual(loaded_clones["Liên"]["ref_text"], "Xin chào")

    def test_character_voice_map_defaults_to_story_config_folder(self):
        segment_file = Path("/tmp/story/story_segments.json")
        expected = Path("/tmp/story/config/character_voices.json")

        self.assertEqual(resolve_voice_map_path(segment_file), expected)

        custom = Path("/tmp/custom/voices.json")
        self.assertEqual(resolve_voice_map_path(segment_file, custom), custom)

    def test_clone_map_resolves_project_relative_asset_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "input" / "story" / "config"
            config_dir.mkdir(parents=True)
            asset_dir = root / "assets" / "voices"
            asset_dir.mkdir(parents=True)
            ref_audio = asset_dir / "thuy.wav"
            ref_audio.write_bytes(b"RIFF" + b"\0" * 2000)
            voices = config_dir / "character_voices.json"
            voices.write_text(json.dumps({
                "narrator": {
                    "voice": "clone-thuy",
                    "ref_audio": "../../../assets/voices/thuy.wav",
                    "ref_text": "Nội dung trong file mẫu",
                },
            }), encoding="utf-8")

            loaded = load_clone_map(voices)

            self.assertEqual(
                loaded["narrator"]["ref_audio"],
                str(ref_audio.resolve()),
            )

    def test_character_synth_helpers_reject_empty_segment_and_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segments = root / "segments.json"
            segments.write_text(json.dumps({
                "segments": [{"speaker": "Liên", "text": "", "pause_after_ms": 400}]
            }), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_segments(segments)

        with self.assertRaises(RuntimeError):
            normalize_audio([])

    def test_split_voice_runs_from_config_without_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "chapter.txt"
            output = root / "segments.json"
            source.write_text("Lien buoc vao phong.\n- Hom nay dep qua.", encoding="utf-8")
            config = root / "split.json"
            config.write_text(json.dumps({
                "input": str(source),
                "output": str(output),
                "characters": ["Lien", "An"],
                "use_llm": False,
                "annotate": "none",
                "max_chars": 400,
                "unknown_as": "narrator",
                "keep_meta": True,
            }), encoding="utf-8")

            result = CliRunner().invoke(app, ["split-voice", "--config", str(config)])

            self.assertEqual(result.exit_code, 0, result.output)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertGreater(len(data["segments"]), 0)
            self.assertIn("type", data["segments"][0])

    def test_extract_characters_runs_from_cli_without_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "chapter.txt"
            output = root / "chapter_config.json"
            source.write_text(
                "Lien buoc vao phong.\n"
                "- Hom nay dep qua.\n"
                "Lien quay lai va noi lai.",
                encoding="utf-8",
            )

            result = CliRunner().invoke(app, [
                "extract-characters",
                str(source),
                "--min-count",
                "1",
                "-o",
                str(output),
            ])

            self.assertEqual(result.exit_code, 0, result.output)
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(data["use_llm"])
            self.assertFalse(data["input"].startswith("/"))


if __name__ == "__main__":
    unittest.main()
