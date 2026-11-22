#!/usr/bin/env python3
"""extract_characters.py - Lọc tên nhân vật từ cả tệp truyện và tạo file config cho split_voices.py.

Quy trình:
  1) Rule (không cần LLM): gom các cụm viết hoa giữa câu và dạng "chức danh + Tên"
     (chị Tí, cụ Thi, bác Siêu...) cùng số lần xuất hiện và đoạn trích.
  2) --scan (tùy chọn): LLM đọc truyện theo từng khúc để bổ sung tên mà rule bỏ sót.
  3) --llm MODEL: LLM lọc danh sách ứng viên, bỏ địa danh / từ thường, và gợi ý bí danh
     (vd "cụ", "bà cụ"). Tên và bí danh LLM trả về đều bị kiểm tra lại với văn bản gốc.
  4) Ghi file config JSON (input, output, characters, aliases, use_llm, ...).

Ví dụ:
  python extract_characters.py input/hai_dua_tre/hai_dua_tre.txt --llm qwen3:14b
  python extract_characters.py truyen.txt --llm qwen3:14b --scan -o truyen_config.json
  python extract_characters.py truyen.txt            # chỉ rule, không LLM (cần tự duyệt lại)
"""
import argparse
import json
import os
import re
import sys
import unicodedata
import urllib.request
from collections import Counter, defaultdict

TITLES = ["chị", "anh", "cụ", "bác", "ông", "bà", "cô", "chú", "cậu", "dì", "thím", "mợ", "thầy"]
# Không bao giờ nhận làm bí danh: đại từ / từ xưng hô chung
ALIAS_BLOCK = {"chị", "anh", "em", "mẹ", "bố", "ba", "má", "cha", "con", "ông", "bà", "cô",
               "chú", "cậu", "dì", "hắn", "nó", "y", "nàng", "chàng", "họ", "mình", "tôi",
               "ta", "người", "bác", "thím", "mợ", "thầy", "bé", "cháu"}

TOK = re.compile(r"\w+")
SKIP_BACK = set(" \t\"“”'‘’-–—(")


def norm(t):
    return unicodedata.normalize("NFC", t)


# --------------------------- Bước 1: rule khai thác ứng viên ------------------
def tokenize(text):
    toks = []
    for m in TOK.finditer(text):
        w = m.group()
        i = m.start() - 1
        while i >= 0 and text[i] in SKIP_BACK:
            i -= 1
        initial = i < 0 or text[i] in ".!?…\n:"
        cap = len(w) >= 2 and w[0].isupper() and w[1:].islower()
        toks.append((w, m.start(), m.end(), cap, initial))
    return toks


def mine(text, min_count):
    """Trả về {tên_chuẩn: {bare, count, titled, snippets}}."""
    toks = tokenize(text)
    n = len(toks)
    lower_count = Counter(w for w, *_ in toks if w == w.lower())
    bare = defaultdict(lambda: [0, 0])  # tên -> [số lần giữa câu, số lần đầu câu]
    titled = Counter()                   # (Chức danh, tên) -> số lần
    pos = defaultdict(list)

    def adj(a, b):
        return toks[b][1] == toks[a][2] + 1 and text[toks[a][2]] == " "

    i = 0
    while i < n:
        w, s, _, cap, init = toks[i]
        if w.lower() in TITLES and i + 1 < n and toks[i + 1][3] and adj(i, i + 1):
            j, parts = i + 1, [toks[i + 1][0]]
            while j + 1 < n and len(parts) < 3 and toks[j + 1][3] and adj(j, j + 1):
                j += 1
                parts.append(toks[j][0])
            name = " ".join(parts)
            titled[(w.capitalize(), name)] += 1
            pos[name].append(s)
            i = j + 1
            continue
        if cap:
            j, parts = i, [w]
            while j + 1 < n and len(parts) < 3 and toks[j + 1][3] and adj(j, j + 1):
                j += 1
                parts.append(toks[j][0])
            start_init = init
            # "Nghe Liên ..." đầu câu: bỏ từ đầu câu nếu nó cũng xuất hiện viết thường trong truyện
            while len(parts) >= 2 and start_init and lower_count[parts[0].lower()] >= 1:
                parts = parts[1:]
                start_init = False
            name = " ".join(parts)
            bare[name][1 if start_init else 0] += 1
            pos[name].append(s)
            i = j + 1
            continue
        i += 1

    cands = {}
    for nm in set(bare) | {n2 for (_, n2) in titled}:
        mid, init = bare.get(nm, (0, 0))
        tc = sum(c for (_, n2), c in titled.items() if n2 == nm)
        total = mid + init + tc
        lc = lower_count[nm.lower()] if " " not in nm else 0
        if lc > total:
            continue  # từ này thường xuất hiện ở dạng viết thường -> từ thường
        if not (tc >= 1 or mid >= min_count or (init >= 3 and lc == 0)):
            continue
        titles = Counter({t: c for (t, n2), c in titled.items() if n2 == nm})
        canonical = f"{titles.most_common(1)[0][0]} {nm}" if titles and tc >= mid + init else nm
        snippets = []
        for p in pos[nm][:2]:
            a, b = max(0, p - 70), min(len(text), p + 90)
            snippets.append("…" + re.sub(r"\s+", " ", text[a:b]).strip() + "…")
        cands[canonical] = {"bare": nm, "count": total, "titled": tc, "snippets": snippets}
    return cands


def count_in(text, phrase):
    return len(re.findall(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


# --------------------------------- LLM (Ollama) --------------------------------
def ollama_json(prompt, model, url, num_ctx=8192):
    body = json.dumps({
        "model": model, "stream": False, "format": "json", "think": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.load(r)["message"]["content"])


def llm_scan(text, model, url, chunk_chars=3000):
    """LLM đọc truyện theo khúc và liệt kê tên nhân vật (để bổ sung ứng viên)."""
    chunks, cur = [], ""
    for p in (x for x in text.split("\n") if x.strip()):
        if cur and len(cur) + len(p) > chunk_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    found = Counter()
    for k, ch in enumerate(chunks, 1):
        print(f"  scan {k}/{len(chunks)}", file=sys.stderr)
        prompt = ("Liệt kê tên riêng của các nhân vật (người) xuất hiện hoặc được nhắc tới trong "
                  "đoạn truyện tiếng Việt sau. Giữ nguyên cách gọi như trong văn bản (ví dụ "
                  "\"Chị Tí\", \"Cụ Thi\"). Không thêm tên không có trong đoạn. Không liệt kê "
                  "địa danh.\n\nĐoạn truyện:\n" + ch +
                  '\n\nChỉ trả về JSON: {"names": ["..."]}')
        try:
            for nm in ollama_json(prompt, model, url).get("names", []):
                if isinstance(nm, str) and nm.strip():
                    found[norm(nm.strip())] += 1
        except Exception as e:  # noqa: BLE001
            print(f"[scan lỗi ở khúc {k}] {e}", file=sys.stderr)
    return found


def llm_verify(cands, model, url, batch=20):
    """Trả về {tên_chuẩn: [bí danh do LLM đề xuất]} cho những ứng viên được giữ lại."""
    keep = {}
    items = sorted(cands.items(), key=lambda kv: -kv[1]["count"])
    for b in range(0, len(items), batch):
        lines = [f'- {nm} (xuất hiện {info["count"]} lần): ' + " | ".join(info["snippets"])
                 for nm, info in items[b: b + batch]]
        prompt = (
            "Dưới đây là các cụm từ viết hoa tìm thấy trong một truyện tiếng Việt, kèm số lần "
            "xuất hiện và vài đoạn trích.\n"
            "Hãy chọn những cụm là TÊN NHÂN VẬT (người, hoặc nhân vật có hành động hay lời thoại "
            "trong truyện, kể cả nhân vật phụ chỉ được nhắc tên). LOẠI các cụm là địa danh, tổ "
            "chức, thời gian, từ thông thường viết hoa vì đứng đầu câu, hoặc chức danh chung.\n"
            'Với mỗi nhân vật được chọn, thêm "aliases": các cách gọi KHÁC của riêng nhân vật đó '
            'xuất hiện trong đoạn trích (ví dụ "cụ", "bà cụ"). Không đưa đại từ hay từ xưng hô '
            "chung (chị, anh, em, hắn, nó, mẹ...). Không có thì để [].\n"
            'Trường "name" phải chép đúng như trong danh sách.\n\nDanh sách:\n' + "\n".join(lines) +
            '\n\nChỉ trả về JSON: {"characters": [{"name": "...", "aliases": ["..."]}]}')
        try:
            out = ollama_json(prompt, model, url)
        except Exception as e:  # noqa: BLE001
            print(f"[LLM lỗi ở batch {b // batch + 1}] {e}", file=sys.stderr)
            continue
        for c in out.get("characters", []):
            nm = c.get("name")
            if nm in cands:
                al = c.get("aliases") or []
                keep[nm] = [a.strip() for a in al if isinstance(a, str) and a.strip()]
    return keep


def clean_aliases(keep, text):
    """Bỏ bí danh bịa (không có trong truyện), đại từ chung, và bí danh trùng giữa các nhân vật."""
    low = text.lower()
    cleaned = {}
    for nm, al in keep.items():
        bare = nm.split(" ", 1)[1] if nm.split()[0].lower() in TITLES and " " in nm else nm
        ok = []
        for a in al:
            a = norm(a)
            if a.lower() in ALIAS_BLOCK or a == nm or a == bare or a.lower() not in low:
                continue
            if a not in ok:
                ok.append(a)
        cleaned[nm] = ok
    freq = Counter(a.lower() for al in cleaned.values() for a in al)
    return {nm: [a for a in al if freq[a.lower()] == 1] for nm, al in cleaned.items()}


# ------------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Tệp truyện .txt")
    ap.add_argument("--llm", metavar="MODEL", help="Model Ollama để lọc ứng viên (vd: qwen3:14b)")
    ap.add_argument("--scan", action="store_true", help="LLM đọc cả truyện để bổ sung ứng viên (chậm hơn)")
    ap.add_argument("--url", default="http://localhost:11434/api/chat")
    ap.add_argument("--min-count", type=int, default=2,
                    help="Số lần xuất hiện giữa câu tối thiểu để thành ứng viên (mặc định 2)")
    ap.add_argument("-o", "--output", help="File config đầu ra (mặc định: <tên>_config.json cạnh input)")
    args = ap.parse_args()

    input_path = os.path.abspath(args.input)
    input_relative = os.path.relpath(input_path, os.getcwd()).replace("\\", "/")
    text = norm(open(input_path, encoding="utf-8").read())
    stem, _ = os.path.splitext(input_path)
    config_path = os.path.abspath(args.output) if args.output else f"{stem}_config.json"

    cands = mine(text, args.min_count)
    print(f"Rule tìm được {len(cands)} ứng viên.", file=sys.stderr)

    if args.llm and args.scan:
        known_bare = {info["bare"] for info in cands.values()} | set(cands)
        for nm, k in llm_scan(text, args.llm, args.url).items():
            bare = nm.split(" ", 1)[1] if nm.split()[0].lower() in TITLES and " " in nm else nm
            c = count_in(text, bare)
            if nm not in known_bare and bare not in known_bare and c >= 1:
                cands[nm] = {"bare": bare, "count": c, "titled": 0,
                             "snippets": [f"(LLM đề xuất, {k} khúc truyện nhắc tới)"]}

    if args.llm:
        keep = llm_verify(cands, args.llm, args.url)
        aliases = clean_aliases(keep, text)
        chosen = sorted(keep, key=lambda n: -cands[n]["count"])
        dropped = [n for n in cands if n not in keep]
    else:
        print("[Chưa dùng LLM] danh sách chưa được lọc, có thể lẫn địa danh/từ thường. "
              "Hãy duyệt tay trong file config.", file=sys.stderr)
        chosen = sorted(cands, key=lambda n: -cands[n]["count"])
        aliases, dropped = {}, []

    config = {
        "_comment": "Dat use_llm=true de dung Ollama; dat false de khong dung LLM.",
        "input": input_relative,
        "output": os.path.relpath(f"{stem}_segments.json", os.getcwd()).replace("\\", "/"),
        "characters": chosen,
    }
    if any(aliases.get(n) for n in chosen):
        config["aliases"] = {n: aliases[n] for n in chosen if aliases.get(n)}
    config.update({
        "use_llm": bool(args.llm), "llm_model": args.llm or "qwen3:14b", "ollama_url": args.url,
        "annotate": "dialogue", "max_chars": 400, "unknown_as": "narrator", "keep_meta": True,
    })
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\nNhân vật ({len(chosen)}):")
    for n in chosen:
        al = f"   bí danh: {', '.join(aliases[n])}" if aliases.get(n) else ""
        print(f"  {n:<14} ~{cands[n]['count']} lần{al}")
    if dropped:
        print("\nĐã loại: " + ", ".join(dropped))
    print(f"\nĐã ghi config -> {config_path}")


if __name__ == "__main__":
    main()
