#!/usr/bin/env python3
"""split_voices.py (v2) - Bước 2: tách truyện thành JSON segments cho TTS đa giọng.

Đầu ra:
  {"segments": [{"speaker", "text", "emotion", "delivery", "pause_after_ms"}, ...]}

Quy trình:
  1) Rule: tách narration / dialogue (dòng bắt đầu bằng gạch đầu dòng - – —).
  2) Rule: gán người nói (dòng dẫn kết thúc bằng ":", luân phiên giữa 2 người).
  3) --llm: hỏi LLM local (Ollama) cho các câu thoại còn "unknown".
  4) Cắt segment quá dài theo ranh giới câu (--max-chars).
  5) --llm: LLM gán emotion / delivery (chỉ trả NHÃN theo id, không đụng vào text).
  6) Tính pause_after_ms bằng rule theo dấu câu, cuối đoạn, đổi người nói.
  7) Kiểm tra toàn vẹn: ghép text lại phải khớp văn bản gốc.

Ví dụ:
  python split_voices.py chuong1.txt --characters Liên An -o chuong1.json
  python split_voices.py chuong1.txt --characters Liên An --llm qwen3:14b
"""
import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

# ---- Từ vựng nhãn: sửa cho khớp với engine TTS của bạn -----------------------
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised",
            "tender", "worried", "tired", "serious"]
DELIVERIES = ["normal", "soft", "whisper", "shout", "fast", "slow",
              "hesitant", "trembling"]
# ---- Thời gian nghỉ (ms) theo dấu câu cuối segment ---------------------------
PAUSE_BY_END = {".": 400, "!": 450, "?": 450, "…": 700, ";": 300, ":": 300, ",": 250}
PAUSE_DEFAULT, PAUSE_PARA_END, PAUSE_SPEAKER_CHANGE, PAUSE_MAX = 400, 200, 100, 1200
TRAIL_CHARS = " \"”“’')]»*_"

DASH_LINE = re.compile(r"^\s*[-–—]\s*(.+)$")
INLINE_TAIL = re.compile(r"\s[-–—]\s")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
VERBS = ["trả lời", "đáp lại", "đáp", "hỏi", "nói", "bảo", "thưa", "reo",
         "gắt", "kêu", "hét", "cười", "thì thầm", "nhắc", "gọi"]
VERB_RE = re.compile(r"(?<!\w)(" + "|".join(VERBS) + r")(?!\w)")


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def new_seg(type_, speaker, text, source, para_end):
    return {"type": type_, "speaker": speaker, "text": text, "source": source,
            "para_end": para_end, "emotion": "neutral", "delivery": "normal"}


# ------------------------------- Rule: tách & gán -----------------------------
def names_in(text, characters):
    found = []
    for name in characters:
        for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
            found.append((m.start(), name))
    return sorted(found)


def speaker_from_lead(line, characters):
    """Người nói = tên gần nhất đứng TRƯỚC động từ nói; không có động từ thì tên đầu tiên."""
    names = names_in(line, characters)
    if not names:
        return None
    v = VERB_RE.search(line)
    if v:
        before = [n for pos, n in names if pos < v.start()]
        if before:
            return before[-1]
    return names[0][1]


def parse(text, characters):
    segs = []
    for raw in text.splitlines():
        line = norm(raw)
        if not line:
            continue
        m = DASH_LINE.match(line)
        if not m:
            segs.append(new_seg("narration", "narrator", line, "rule", True))
            continue
        body = m.group(1)
        parts = INLINE_TAIL.split(body, maxsplit=1)
        # Dạng:  - Chị đợi lâu chưa? - Minh hỏi.
        if (len(parts) == 2 and VERB_RE.search(parts[1]) and len(parts[1]) < 80
                and not INLINE_TAIL.search(parts[1])):
            who = speaker_from_lead(parts[1], characters)
            segs.append(new_seg("dialogue", who, parts[0].strip(),
                                "rule" if who else None, False))
            segs.append(new_seg("narration", "narrator", parts[1].strip(), "rule", True))
        else:
            segs.append(new_seg("dialogue", None, body.strip(), None, True))
    return segs


def pick_partner(known, gap_text, history, characters):
    for h in history:
        if h not in (None, known, "unknown"):
            return h
    for _, n in names_in(gap_text, characters):
        if n != known:
            return n
    return None


def resolve(segs, characters, max_gap=1):
    idx = [i for i, s in enumerate(segs) if s["type"] == "dialogue"]

    for i in idx:  # dòng dẫn kết thúc bằng ":"
        if segs[i]["speaker"]:
            continue
        prev = segs[i - 1] if i > 0 else None
        if prev and prev["type"] == "narration" and prev["text"].endswith(":"):
            who = speaker_from_lead(prev["text"], characters)
            if who:
                segs[i]["speaker"], segs[i]["source"] = who, "rule"

    def gap_text(a, b):
        return " ".join(segs[j]["text"] for j in range(a + 1, b)
                        if segs[j]["type"] == "narration")

    def gap_len(a, b):
        return sum(1 for j in range(a + 1, b) if segs[j]["type"] == "narration")

    for _ in range(3):  # luân phiên xuôi / ngược
        changed = False
        for k, i in enumerate(idx):
            if segs[i]["speaker"]:
                continue
            if k > 0:
                p = idx[k - 1]
                P = segs[p]["speaker"]
                if P and gap_len(p, i) <= max_gap:
                    hist = [segs[j]["speaker"] for j in reversed(idx[:k - 1])]
                    who = pick_partner(P, gap_text(p, i), hist, characters)
                    if who:
                        segs[i]["speaker"], segs[i]["source"] = who, "alternation"
                        changed = True
                        continue
            if k < len(idx) - 1:
                n = idx[k + 1]
                N = segs[n]["speaker"]
                if N and gap_len(i, n) <= max_gap:
                    hist = [segs[j]["speaker"] for j in idx[k + 2:]]
                    who = pick_partner(N, gap_text(i, n), hist, characters)
                    if who:
                        segs[i]["speaker"], segs[i]["source"] = who, "alternation"
                        changed = True
        if not changed:
            break

    for i in idx:
        if not segs[i]["speaker"]:
            segs[i]["speaker"], segs[i]["source"] = "unknown", "none"
    return segs


# --------------------------------- LLM (Ollama) -------------------------------
def ollama_json(prompt, model, url, num_ctx=8192):
    body = json.dumps({
        "model": model, "stream": False, "format": "json",
        "think": False,  # tắt chế độ suy luận của Qwen3 (cần Ollama bản mới)
        "options": {"temperature": 0, "num_ctx": num_ctx},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(json.load(r)["message"]["content"])


def llm_fill_speakers(segs, characters, model, url):
    for i, s in enumerate(segs):
        if s["type"] != "dialogue" or s["speaker"] != "unknown":
            continue
        ctx = "\n".join(x["text"] for x in segs[max(0, i - 3): i + 2])
        prompt = (f"Nhân vật: {', '.join(characters)}.\nĐoạn truyện:\n{ctx}\n\n"
                  f'Ai nói câu: "{s["text"]}"?\n'
                  'Chỉ trả về JSON dạng {"speaker": "<một nhân vật trong danh sách hoặc unknown>"}')
        try:
            who = ollama_json(prompt, model, url).get("speaker")
            if who in characters:
                s["speaker"], s["source"] = who, "llm"
        except Exception as e:  # noqa: BLE001
            print(f"[LLM speaker lỗi ở segment {i}] {e}", file=sys.stderr)


def llm_annotate(segs, model, url, which="dialogue", batch=8):
    """LLM chỉ trả nhãn emotion/delivery theo id; text không bao giờ đi qua LLM."""
    targets = [i for i, s in enumerate(segs) if which == "all" or s["type"] == "dialogue"]
    for b in range(0, len(targets), batch):
        ids = targets[b: b + batch]
        lo, hi = max(0, ids[0] - 2), min(len(segs), ids[-1] + 2)
        ctx = "\n".join(f'[{j}] {segs[j]["speaker"]}: {segs[j]["text"]}' for j in range(lo, hi))
        prompt = (
            "Bạn là đạo diễn lồng tiếng cho truyện. Dựa vào nội dung và ngữ cảnh, "
            "chọn cảm xúc và cách đọc cho từng câu.\n"
            f"Cảm xúc hợp lệ: {', '.join(EMOTIONS)}\n"
            f"Cách đọc hợp lệ: {', '.join(DELIVERIES)}\n\n"
            f"Đoạn truyện:\n{ctx}\n\n"
            f"Cần gán nhãn cho id: {', '.join(map(str, ids))}\n"
            'Chỉ trả về JSON: {"items": [{"id": <số>, "emotion": "...", "delivery": "..."}]}'
        )
        try:
            out = ollama_json(prompt, model, url)
            for it in out.get("items", []):
                try:
                    i = int(it.get("id"))
                except (TypeError, ValueError):
                    continue
                if i not in ids:
                    continue
                if it.get("emotion") in EMOTIONS:
                    segs[i]["emotion"] = it["emotion"]
                if it.get("delivery") in DELIVERIES:
                    segs[i]["delivery"] = it["delivery"]
        except Exception as e:  # noqa: BLE001
            print(f"[LLM annotate lỗi ở batch {ids}] {e}", file=sys.stderr)


# ------------------------------ Hậu xử lý & xuất ------------------------------
def split_long(segs, max_chars):
    """Cắt segment dài theo ranh giới câu để TTS đọc ổn định."""
    out = []
    for s in segs:
        if len(s["text"]) <= max_chars:
            out.append(s)
            continue
        chunks, cur = [], ""
        for sent in SENTENCE_SPLIT.split(s["text"]):
            if cur and len(cur) + 1 + len(sent) > max_chars:
                chunks.append(cur)
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            chunks.append(cur)
        for j, c in enumerate(chunks):
            d = dict(s)
            d["text"] = c
            d["para_end"] = s["para_end"] and j == len(chunks) - 1
            out.append(d)
    return out


def pause_for(text, para_end, speaker_change):
    t = text.rstrip(TRAIL_CHARS)
    base = 700 if t.endswith("...") else PAUSE_BY_END.get(t[-1:], PAUSE_DEFAULT)
    if para_end:
        base += PAUSE_PARA_END
    if speaker_change:
        base += PAUSE_SPEAKER_CHANGE
    return min(base, PAUSE_MAX)


def check_integrity(original, segs):
    strip = lambda t: re.sub(r"[\s\-–—]+", "", unicodedata.normalize("NFC", t))
    return strip(original) == strip("".join(s["text"] for s in segs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?")
    ap.add_argument("--config", metavar="FILE", help="Đọc tham số chạy từ file JSON")
    ap.add_argument("--characters", nargs="*", default=None, help="Tên các nhân vật")
    ap.add_argument("--llm", metavar="MODEL", help="Model Ollama (vd: qwen3:14b)")
    ap.add_argument("--url", default=None)
    ap.add_argument("--annotate", choices=["none", "dialogue", "all"], default=None,
                    help="Gán emotion/delivery cho: thoại (mặc định), tất cả, hoặc không (cần --llm)")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="Cắt segment dài hơn số ký tự này theo ranh giới câu")
    ap.add_argument("--unknown-as", default=None,
                    help="Giọng dùng cho câu thoại không xác định được người nói")
    ap.add_argument("--keep-meta", action="store_true",
                    help="Thêm type/source/needs_review vào từng segment để debug")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    config = {}
    if args.config:
        config_path = Path(args.config)
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)

        base_dir = config_path.resolve().parent.parent
        args.input = args.input or config.get("input")
        args.characters = args.characters or config.get("characters")
        args.output = args.output or config.get("output", "segments.json")
        args.url = args.url or config.get("ollama_url", "http://localhost:11434/api/chat")
        args.annotate = args.annotate or config.get("annotate", "dialogue")
        args.max_chars = args.max_chars if args.max_chars is not None else config.get("max_chars", 400)
        args.unknown_as = args.unknown_as or config.get("unknown_as", "narrator")
        args.keep_meta = args.keep_meta or config.get("keep_meta", False)

        if args.llm is None:
            if config.get("use_llm", False):
                args.llm = config.get("llm_model") or config.get("llm")

        if args.input and not os.path.isabs(args.input):
            args.input = str(base_dir / args.input)
        if args.output and not os.path.isabs(args.output):
            args.output = str(base_dir / args.output)
    else:
        args.url = args.url or "http://localhost:11434/api/chat"
        args.annotate = args.annotate or "dialogue"
        args.max_chars = args.max_chars if args.max_chars is not None else 400
        args.unknown_as = args.unknown_as or "narrator"
        args.output = args.output or "segments.json"

    if not args.input:
        ap.error("thieu file input: truyen vao vi tri cach hoac dat input trong --config")
    if not args.characters:
        ap.error("thieu danh sach nhan vat: dung --characters hoac dat characters trong --config")

    text = open(args.input, encoding="utf-8").read()
    segs = resolve(parse(text, args.characters), args.characters)
    if args.llm:
        llm_fill_speakers(segs, args.characters, args.llm, args.url)
    segs = split_long(segs, args.max_chars)
    if args.llm and args.annotate != "none":
        llm_annotate(segs, args.llm, args.url, args.annotate)

    if not check_integrity(text, segs):
        sys.exit("LỖI: nội dung ghép lại không khớp văn bản gốc!")

    final, warnings = [], []
    for k, s in enumerate(segs):
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        change = nxt is not None and nxt["speaker"] != s["speaker"]
        review = s["type"] == "dialogue" and s["source"] in ("none", "llm")
        speaker = args.unknown_as if s["speaker"] == "unknown" else s["speaker"]
        item = {"speaker": speaker, "text": s["text"], "emotion": s["emotion"],
                "delivery": s["delivery"],
                "pause_after_ms": pause_for(s["text"], s["para_end"], change)}
        if args.keep_meta:
            item.update({"type": s["type"], "source": s["source"], "needs_review": review})
        if review:
            warnings.append((k, s["speaker"], s["source"], s["text"][:60]))
        final.append(item)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"segments": final}, f, ensure_ascii=False, indent=2)

    for k, it in enumerate(final):
        print(f'[{k:>2}] {it["speaker"]:<9} {it["emotion"]:<9} {it["delivery"]:<8} '
              f'{it["pause_after_ms"]:>4}ms  {it["text"][:55]}')
    print(f"\nĐã ghi {len(final)} segment -> {args.output}")
    if warnings:
        print("\nCần duyệt lại (speaker không chắc chắn):")
        for k, who, src, t in warnings:
            print(f'  [{k}] speaker={who} nguồn={src}: {t}')


if __name__ == "__main__":
    main()
