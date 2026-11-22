# Hướng dẫn gọi VieNeu-TTS `tts.infer()` theo từng nhân vật

Trong dự án, cấu hình riêng của từng truyện đặt tại:

```text
input/<ten-truyen>/config/split_voices.json
input/<ten-truyen>/config/character_voices.json
```

`split_voices.json` cấu hình tách đoạn; `character_voices.json` ánh xạ
speaker sang tên giọng VieNeu-TTS.

## 1. Ý tưởng

-   Khởi tạo **một instance** `Vieneu()` cho toàn bộ kịch bản.
-   Đọc danh sách đoạn thoại từ JSON.
-   Dùng trường `speaker` để tra giọng tương ứng trong `VOICE_MAP`.
-   Gọi `tts.infer()` cho từng đoạn, truyền `text` và `voice`.
-   Lưu từng đoạn thành file WAV; sau đó ghép các đoạn và chèn khoảng
    nghỉ.

## 2. Ví dụ JSON đầu vào

``` json
{
  "segments": [
    {
      "speaker": "narrator",
      "text": "Buổi sáng hôm ấy, thành phố thức dậy trong màn sương.",
      "pause_after_ms": 500
    },
    {
      "speaker": "minh",
      "text": "Tôi nghĩ chúng ta nên bắt đầu ngay bây giờ.",
      "pause_after_ms": 300
    },
    {
      "speaker": "lan",
      "text": "Bạn chắc chắn chứ?",
      "pause_after_ms": 700
    }
  ]
}
```

`pause_after_ms` là khoảng nghỉ sau đoạn, tính bằng mili giây. Nếu không
có trường này, chương trình có thể dùng mặc định 400 ms.

## 3. Ánh xạ nhân vật với giọng

``` python
VOICE_MAP = {
    "narrator": "Hải Đăng",
    "minh": "Xuân Vĩnh",
    "lan": "Mai Anh",
}
```

Tên giọng phải khớp với tên giọng có trong phiên bản VieNeu-TTS đang
cài. Kiểm tra API và danh sách giọng theo đúng phiên bản thư viện của
bạn; không nên giả định mọi phiên bản đều có cùng phương thức liệt kê
giọng.

## 4. Code tổng hợp từng đoạn

Ví dụ dưới đây minh họa luồng xử lý. Cần xác nhận chữ ký `Vieneu`,
`infer`, `save` và thuộc tính sample rate theo phiên bản VieNeu-TTS đang
cài trước khi dùng nguyên trạng.

``` python
import json
from pathlib import Path

import numpy as np
from vieneu import Vieneu

# Khởi tạo model một lần
tts = Vieneu()

VOICE_MAP = {
    "narrator": "Hải Đăng",
    "minh": "Xuân Vĩnh",
    "lan": "Mai Anh",
}


def generate_dialogue(json_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_parts = []

    for index, segment in enumerate(data["segments"], start=1):
        speaker = segment["speaker"]
        text = segment["text"]

        voice = VOICE_MAP.get(speaker)
        if voice is None:
            raise ValueError(f"Chưa cấu hình giọng cho nhân vật: {speaker}")

        print(f"[{index}] {speaker} -> {voice}")

        # Tổng hợp đoạn hiện tại bằng giọng của nhân vật
        audio = tts.infer(
            text=text,
            voice=voice,
            temperature=0.8,
        )

        audio = np.asarray(audio, dtype=np.float32)

        # Lưu riêng từng đoạn để dễ nghe thử và thay thế
        segment_path = output_dir / f"segment_{index:03d}.wav"
        tts.save(audio, str(segment_path))

        audio_parts.append(audio)

        # Chèn khoảng lặng sau đoạn
        pause_ms = segment.get("pause_after_ms", 400)
        if pause_ms > 0:
            silence = np.zeros(
                int(tts.sample_rate * pause_ms / 1000),
                dtype=np.float32,
            )
            audio_parts.append(silence)

    if audio_parts:
        final_audio = np.concatenate(audio_parts)
        final_path = output_dir / "final_dialogue.wav"
        tts.save(final_audio, str(final_path))
        print(f"Hoàn thành: {final_path}")


if __name__ == "__main__":
    generate_dialogue("script.json", "output")
```

Chạy chương trình:

``` bash
python generate_dialogue.py
```

Kết quả dự kiến:

``` text
output/
├── segment_001.wav
├── segment_002.wav
├── segment_003.wav
└── final_dialogue.wav
```

## 5. Dùng giọng clone cho nhân vật

Khi chạy `synthesize-voices`, có thể khai báo clone trực tiếp trong
`character_voices.json`. Đường dẫn `ref_audio` tương đối được tính từ thư mục
`config` chứa file JSON:

```json
{
  "narrator": "Hải Đăng",
  "Liên": {
    "voice": "clone-lien",
    "ref_audio": "assets/voices/lien.wav",
    "ref_text": "Nội dung chính xác trong đoạn ghi âm"
  },
  "An": "Quang Sơn"
}
```

Chạy bình thường:

```bash
python -m src.main synthesize-voices input/hai_dua_tre/hai_dua_tre_segments.json
```

Nếu không đặt `"voice"` cho clone, hệ thống dùng tên mặc định `<tên nhân vật> clone`.
`ref_text` là tài liệu mô tả giọng mẫu; VieNeu-TTS đăng ký clone bằng `ref_audio`.

Nếu muốn dùng giọng tham chiếu riêng, đăng ký các giọng khi khởi động:

``` python
tts = Vieneu()

tts.add_voice("Narrator", "voices/narrator.wav")
tts.add_voice("Minh", "voices/minh.wav")
tts.add_voice("Lan", "voices/lan.wav")

VOICE_MAP = {
    "narrator": "Narrator",
    "minh": "Minh",
    "lan": "Lan",
}
```

Sau khi đăng ký, tiếp tục gọi `tts.infer(text=..., voice=...)` như bình
thường. Chỉ sử dụng bản ghi giọng mà bạn có quyền sử dụng.

## 6. Lưu ý

1.  **Không khởi tạo model trong vòng lặp.** Khởi tạo một lần rồi dùng
    lại cho tất cả đoạn.
2.  **Mỗi đoạn có thể chọn một voice khác nhau.** Đây là lý do gọi
    `infer()` riêng cho từng đoạn phù hợp với hội thoại nhiều nhân vật.
3.  **Không mặc định `infer_batch()` hỗ trợ nhiều voice trong cùng
    batch.** Hãy kiểm tra tài liệu của đúng phiên bản; cách tuần tự là
    phương án dễ kiểm soát nhất.
4.  **Kiểm tra định dạng audio và sample rate.** Đoạn code ghép bằng
    `numpy.concatenate` giả định các đoạn có cùng sample rate, số chiều
    và kiểu dữ liệu. Nếu khác nhau, cần chuẩn hóa trước khi ghép.
5.  **Xác nhận API theo phiên bản cài đặt.** Có thể kiểm tra chữ ký hàm
    bằng:

``` bash
python - <<'PY'
import inspect
from vieneu import Vieneu

print("Vieneu:", inspect.signature(Vieneu))
print("infer:", inspect.signature(Vieneu.infer))
print("save:", inspect.signature(Vieneu.save))
print("add_voice:", inspect.signature(Vieneu.add_voice))
PY
```

Nếu API hoặc thuộc tính trong ví dụ không khớp, hãy ưu tiên chữ ký hiển
thị từ phiên bản đã cài.

## 7. Cấu hình nhân vật trong JSON (tùy chọn)

Có thể đặt cấu hình nhân vật ngay trong JSON để thay giọng mà không sửa
code:

``` json
{
  "characters": {
    "narrator": {
      "voice": "Hải Đăng"
    },
    "minh": {
      "voice": "Xuân Vĩnh"
    },
    "lan": {
      "voice": "Mai Anh"
    }
  },
  "segments": [
    {
      "speaker": "narrator",
      "text": "Câu dẫn chuyện đầu tiên.",
      "pause_after_ms": 500
    },
    {
      "speaker": "minh",
      "text": "Lời thoại của Minh.",
      "pause_after_ms": 300
    }
  ]
}
```

Khi đó, chương trình lấy voice từ `data["characters"][speaker]["voice"]`
thay vì dùng `VOICE_MAP` viết cố định trong Python.
