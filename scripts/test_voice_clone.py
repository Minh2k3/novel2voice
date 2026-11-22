#!/usr/bin/env python3
"""Sinh audio thử cho các giọng clone VieNeu-TTS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_VOICES_DIR = PROJECT_ROOT / "assets" / "voices"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "tests" / "voices"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

DEFAULT_TEST_TEXTS = [
    "Em yêu anh nhiều lắm, anh có biết không?",
    "Hôm nay trời đẹp quá, chúng mình đi dạo nhé.",
    "Cuộc sống thật tuyệt vời khi có anh bên cạnh.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test giọng clone VieNeu-TTS từ các file trong assets/voices.",
    )
    parser.add_argument(
        "-v",
        "--voice",
        action="append",
        help="File giọng clone. Có thể truyền nhiều lần; mặc định quét cả thư mục giọng.",
    )
    parser.add_argument(
        "--voices-dir",
        type=Path,
        default=DEFAULT_VOICES_DIR,
        help=f"Thư mục quét giọng mẫu (mặc định: {DEFAULT_VOICES_DIR.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thư mục lưu audio thử (mặc định: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)}).",
    )
    parser.add_argument(
        "-t",
        "--text",
        action="append",
        help="Câu thử. Có thể truyền nhiều lần; mặc định dùng 3 câu có sẵn.",
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        help="File UTF-8, mỗi dòng không trống là một câu thử.",
    )
    parser.add_argument(
        "--ref-text",
        help="Nội dung đã đọc trong file giọng mẫu. Dùng chung cho mọi giọng.",
    )
    parser.add_argument(
        "--mode",
        default="v3turbo",
        choices=("v3turbo", "standard", "fast", "remote"),
        help="Mode VieNeu-TTS (mặc định: v3turbo).",
    )
    parser.add_argument(
        "--format",
        choices=("wav", "mp3"),
        default="wav",
        help="Định dạng audio đầu ra (mặc định: wav).",
    )
    return parser.parse_args()


def resolve_voice_files(args: argparse.Namespace) -> list[Path]:
    if args.voice:
        files: list[Path] = []
        for item in args.voice:
            path = Path(item).expanduser()
            if not path.is_absolute():
                candidate = PROJECT_ROOT / path
                path = candidate if candidate.exists() else args.voices_dir / path
            if not path.is_file():
                raise FileNotFoundError(f"Không tìm thấy file giọng clone: {path}")
            if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                raise ValueError(f"Định dạng giọng không được hỗ trợ: {path}")
            files.append(path.resolve())
        return files

    voices_dir = args.voices_dir.expanduser()
    if not voices_dir.is_absolute():
        voices_dir = PROJECT_ROOT / voices_dir
    if not voices_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục giọng: {voices_dir}"
        )
    return sorted(
        (
            path.resolve()
            for path in voices_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )


def load_test_texts(args: argparse.Namespace) -> list[str]:
    texts: list[str] = []
    if args.text_file:
        path = args.text_file.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file câu thử: {path}")
        texts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines())

    if args.text:
        texts.extend(args.text)
    if not texts:
        texts = list(DEFAULT_TEST_TEXTS)

    return [text for text in texts if text.strip()]


def main() -> int:
    args = parse_args()
    try:
        voice_files = resolve_voice_files(args)
        test_texts = load_test_texts(args)
        if not voice_files:
            raise FileNotFoundError(
                f"Không có file giọng trong thư mục: "
                f"{args.voices_dir if args.voices_dir.is_absolute() else PROJECT_ROOT / args.voices_dir}"
            )

        from src.tts import get_tts_provider

        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Tìm thấy {len(voice_files)} giọng clone.")
        print(f"Mỗi giọng sẽ đọc {len(test_texts)} câu.")
        print(f"Đầu ra: {output_dir}")

        for voice_index, voice_path in enumerate(voice_files, start=1):
            print(
                f"\n[{voice_index}/{len(voice_files)}] "
                f"Đang test: {voice_path.relative_to(PROJECT_ROOT) if voice_path.is_relative_to(PROJECT_ROOT) else voice_path}"
            )
            provider = get_tts_provider(
                provider_name="vieneu",
                vieneu_mode=args.mode,
                vieneu_ref_audio=str(voice_path),
                vieneu_ref_text=args.ref_text,
            )
            try:
                tasks = [
                    (
                        text,
                        output_dir / f"{voice_path.stem}_{index:02d}.{args.format}",
                    )
                    for index, text in enumerate(test_texts, start=1)
                ]
                provider.generate_batch(tasks, max_concurrency=1, show_progress=True)
                for _, output_path in tasks:
                    print(f"  - {output_path.relative_to(PROJECT_ROOT)}")
            finally:
                close = getattr(provider, "close", None)
                if close:
                    close()

        print("\nHoàn tất test giọng clone.")
        return 0
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
