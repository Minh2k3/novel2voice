#!/usr/bin/env python3
"""synthesize.py - Bước 3: đọc segments.json (từ split_voices.py) -> 1 file audio bằng VieNeu-TTS v3 Turbo.

  - Mỗi speaker -> 1 giọng theo voices.json, ví dụ:
        {"narrator": "Hải Đăng", "Liên": "Mai Anh", "An": "Quang Sơn"}
  - Gom các segment cùng giọng vào infer_batch() (nhanh trên GPU), rồi ghép lại đúng thứ tự.
  - Chèn khoảng lặng pause_after_ms giữa các segment.
  - --cues: chuyển emotion thành thẻ [cười] / [thở dài] của VieNeu (tính năng thử nghiệm).

Dùng:
  python synthesize.py chuong1.json -o chuong1.wav
  python synthesize.py chuong1.json --voices config/character_voices.json -o chuong1.wav --cues --parts parts/
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.story_paths import get_story_output_root

# VieNeu chỉ hỗ trợ 3 thẻ: [cười] [thở dài] [hắng giọng]. Sửa bảng này tùy ý.
EMOTION_TO_CUE = {"happy": "[cười]", "sad": "[thở dài]", "tired": "[thở dài]"}


def load_segments(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("JSON không có trường segments dạng danh sách.")
    if not segments:
        raise ValueError("Không có segment nào để tổng hợp.")
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment {index} không phải object JSON.")
        if not segment.get("speaker"):
            raise ValueError(f"Segment {index} thiếu trường speaker.")
        if not isinstance(segment.get("text"), str) or not segment["text"].strip():
            raise ValueError(f"Segment {index} thiếu text.")
    return segments


def load_voice_map(path):
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if "characters" in data:
        data = data["characters"]
    voices = {}
    for speaker, value in data.items():
        if isinstance(value, dict):
            nested = value.get("clone")
            voice_name = (
                value.get("voice")
                or value.get("clone_name")
                or value.get("name")
                or (nested.get("name") if isinstance(nested, dict) else nested)
            )
            ref_audio = (
                value.get("ref_audio")
                or value.get("clone_audio")
                or value.get("audio")
            )
            if isinstance(nested, dict):
                ref_audio = ref_audio or nested.get("ref_audio") or nested.get("audio")
            value = voice_name or (f"{speaker} clone" if ref_audio else None)
        if isinstance(value, str) and value.strip():
            voices[speaker] = value.strip()
    if "narrator" not in voices:
        raise ValueError('Voice map phải có khóa "narrator".')
    return voices


def load_clone_map(path):
    """Đọc cấu hình giọng clone theo từng nhân vật.

    Ví dụ:
      "Liên": {
        "voice": "clone-lien",
        "ref_audio": "assets/voices/lien.wav",
        "ref_text": "Nội dung chính xác trong file ghi âm"
      }

    Trường ref_text chỉ mang tính tài liệu; VieNeu-TTS đăng ký clone qua ref_audio.
    Đường dẫn tương đối được tính từ thư mục chứa file voice map.
    """
    voice_path = Path(path)
    with voice_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if "characters" in data:
        data = data["characters"]

    clones = {}
    for speaker, value in data.items():
        if not isinstance(value, dict):
            continue

        ref_audio = value.get("ref_audio") or value.get("clone_audio") or value.get("audio")
        nested = value.get("clone")
        if isinstance(nested, dict):
            ref_audio = ref_audio or nested.get("ref_audio") or nested.get("audio")

        if not ref_audio:
            continue
        if not isinstance(ref_audio, str) or not ref_audio.strip():
            raise ValueError(f"Cấu hình clone của nhân vật '{speaker}' thiếu ref_audio hợp lệ.")

        clone_name = (
            value.get("voice")
            or value.get("clone_name")
            or value.get("name")
            or (nested.get("name") if isinstance(nested, dict) else None)
        )
        if isinstance(nested, str):
            clone_name = clone_name or nested
        if not isinstance(clone_name, str) or not clone_name.strip():
            clone_name = f"{speaker} clone"

        ref_text = value.get("ref_text") or value.get("clone_text")
        if isinstance(nested, dict):
            ref_text = ref_text or nested.get("ref_text") or nested.get("text")

        audio_path = Path(ref_audio.strip()).expanduser()
        if not audio_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            story_root = voice_path.parent.parent
            candidates = (
                voice_path.parent / audio_path,
                story_root / audio_path,
                project_root / audio_path,
            )
            audio_path = next(
                (candidate.resolve() for candidate in candidates if candidate.is_file()),
                (voice_path.parent / audio_path).resolve(),
            )
        else:
            audio_path = audio_path.resolve()

        clones[speaker] = {
            "name": clone_name.strip(),
            "ref_audio": str(audio_path),
            "ref_text": ref_text.strip() if isinstance(ref_text, str) else None,
        }
    return clones


def resolve_voice_map_path(input_path, voices_path=None):
    if voices_path:
        return Path(voices_path)
    input_path = Path(input_path)
    return input_path.parent / "config" / "character_voices.json"


def normalize_audio(audio):
    result = np.asarray(audio, dtype=np.float32).reshape(-1)
    if result.size == 0:
        raise RuntimeError("VieNeu-TTS trả về audio rỗng.")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="File JSON từ split_voices.py")
    ap.add_argument("--voices", help="File JSON ánh xạ speaker -> tên giọng")
    ap.add_argument("-o", "--output")
    ap.add_argument("--cues", action="store_true",
                    help="Thêm thẻ cảm xúc [cười]/[thở dài] cho thoại (thử nghiệm)")
    ap.add_argument("--parts", metavar="DIR", help="Lưu thêm từng segment ra file wav để nghe thử")
    args = ap.parse_args()

    story_root = get_story_output_root(Path(args.input))
    if not args.output:
        args.output = str(story_root / "media" / "character.wav")

    try:
        segs = load_segments(args.input)
        voice_map_path = resolve_voice_map_path(args.input, args.voices)
        voices = load_voice_map(voice_map_path)
        clones = load_clone_map(voice_map_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        sys.exit(str(exc))

    from vieneu import Vieneu  # import muộn để lỗi cài đặt báo rõ ràng

    tts = Vieneu()  # mặc định v3 Turbo; có GPU CUDA thì tự dùng PyTorch
    sample_rate = int(getattr(tts, "sample_rate", 48000))
    known = {x for pair in tts.list_preset_voices() for x in pair}

    for speaker, clone in clones.items():
        try:
            ref_path = Path(clone["ref_audio"])
            if not ref_path.is_file():
                raise ValueError(
                    f'Không tìm thấy file ghi âm clone của "{speaker}": {ref_path}'
                )
            if ref_path.stat().st_size < 1000:
                raise ValueError(
                    f'File ghi âm clone của "{speaker}" quá nhỏ hoặc rỗng: {ref_path}'
                )
            tts.add_voice(clone["name"], str(ref_path))
            voices[speaker] = clone["name"]
        except (OSError, ValueError, RuntimeError) as exc:
            sys.exit(str(exc))

    for name in set(voices.values()):
        if name not in known and name not in {clone["name"] for clone in clones.values()}:
            print(f"[Cảnh báo] giọng '{name}' không nằm trong danh sách preset "
                  f"(nếu là giọng clone đã add_voice thì bỏ qua).", file=sys.stderr)

    # 1) Chuẩn bị text + gom theo giọng
    groups, texts = {}, []
    for i, s in enumerate(segs):
        voice = voices.get(s["speaker"])
        if voice is None:
            print(f'[Cảnh báo] speaker "{s["speaker"]}" chưa có trong voices.json -> dùng narrator',
                  file=sys.stderr)
            voice = voices["narrator"]
        text = s["text"]
        cue = EMOTION_TO_CUE.get(s.get("emotion"))
        if args.cues and cue and s["speaker"] != "narrator":
            text = f"{cue} {text}"
        texts.append(text)
        groups.setdefault(voice, []).append(i)

    # 2) Tổng hợp theo từng giọng (batch)
    audios = [None] * len(segs)
    t0 = time.time()
    try:
        for voice, idx in groups.items():
            out = tts.infer_batch([texts[i] for i in idx], voice=voice)
            if len(out) != len(idx):
                raise RuntimeError(
                    f"VieNeu trả về {len(out)} audio cho {len(idx)} segment giọng '{voice}'."
                )
            for i, a in zip(idx, out):
                audios[i] = normalize_audio(a)
    finally:
        elapsed = time.time() - t0

    # 3) Ghép theo thứ tự + chèn khoảng lặng
    pieces = []
    if args.parts:
        os.makedirs(args.parts, exist_ok=True)
    try:
        for i, (s, a) in enumerate(zip(segs, audios)):
            pieces.append(a)
            pause_ms = s.get("pause_after_ms", 400)
            if not isinstance(pause_ms, (int, float)) or pause_ms < 0:
                raise ValueError(
                    f"Segment {i + 1} có pause_after_ms không hợp lệ: {pause_ms!r}"
                )
            pieces.append(
                np.zeros(int(sample_rate * pause_ms / 1000), dtype=np.float32)
            )
            if args.parts:
                tts.save(a, os.path.join(args.parts, f'{i:04d}_{s["speaker"]}.wav'))
        full = np.concatenate(pieces)
        tts.save(full, args.output)

        dur = len(full) / sample_rate
        print(f"Đã ghi {args.output}: {dur:.1f}s audio từ {len(segs)} segment, "
              f"tổng hợp mất {elapsed:.1f}s "
              f"(RTF ≈ {elapsed / max(dur, 1e-9):.3f})")
    finally:
        close = getattr(tts, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
