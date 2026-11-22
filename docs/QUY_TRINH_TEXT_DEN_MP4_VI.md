# Quy trình từ tệp văn bản đến MP4

| Bước | Input | Output | Cách 1: đầy đủ parameter | Cách 2: dùng config | Lệnh ví dụ |
|---|---|---|---|---|---|
| 1. Chuẩn bị thư mục truyện và config | Tệp TXT gốc; tên truyện; danh sách nhân vật | `input/<ten-truyen>/config/split_voices.json`; `input/<ten-truyen>/config/character_voices.json`; tệp truyện trong `input/<ten-truyen>/` | Tạo thư mục và khai báo từng đường dẫn, danh sách nhân vật, giọng của từng nhân vật bằng tham số hoặc file JSON riêng | Đặt hai file JSON trong `input/<ten-truyen>/config/` và để các đường dẫn input/output tương đối tính từ thư mục config | `mkdir -p input/hai_dua_tre/config` |
| 2. Dịch văn bản nếu tệp gốc là tiếng Anh | Tệp TXT tiếng Anh | Tệp TXT tiếng Việt | Truyền đầy đủ file, provider dịch, model, glossary và output | Thiết lập `translation_provider`, `model`, `glossary_path`, `input_path` và `output_audio` trong file process config | `python -m src.main translate input/hai_dua_tre/hai_dua_tre_en.txt --translation-provider deepseek --model deepseek-chat --glossary glossary.csv -o input/hai_dua_tre/hai_dua_tre.txt` |
| 3. Tách tệp thành các segment theo nhân vật | Tệp TXT tiếng Việt; danh sách nhân vật; cấu hình tách/annotate | JSON segments có speaker, text, emotion, delivery và pause_after_ms | Truyền đầy đủ `--config`; file config phải rõ input, output, characters, use_llm, llm_model, ollama_url, annotate, max_chars, unknown_as và keep_meta | Chạy `split-voice --config input/<ten-truyen>/config/split_voices.json`; các đường dẫn `input` và `output` trong JSON được tính từ thư mục config | `python -m src.main split-voice --config input/hai_dua_tre/config/split_voices.json` |
| 4. Tổng hợp từng segment bằng giọng nhân vật | Segments JSON; voice map; VieNeu-TTS model | File WAV tổng hợp; tùy chọn các file WAV từng segment | Truyền đầy đủ file input, file voice map, file WAV output và thư mục parts qua tham số CLI | Bỏ qua `--voices`; hệ thống tự đọc `input/<ten-truyen>/config/character_voices.json` theo thư mục của segments JSON | `python -m src.main synthesize-voices input/hai_dua_tre/hai_dua_tre_segments.json --voices input/hai_dua_tre/config/character_voices.json -o output/hai_dua_tre/hai_dua_tre_characters.wav --parts output/hai_dua_tre/parts` |
| 5. Chuyển WAV tổng hợp sang MP3 | WAV tổng hợp của Bước 4 | MP3 đầu vào cho render video | Khai báo đầy đủ file WAV đầu vào, codec, bitrate và đường dẫn MP3 đầu ra bằng ffmpeg | Chưa có command config riêng cho bước này; đường dẫn có thể được đặt trong script/config quy trình của phần mềm nếu mở rộng | `ffmpeg -y -i output/hai_dua_tre/hai_dua_tre_characters.wav -codec:a libmp3lame -b:a 128k output/hai_dua_tre/hai_dua_tre_characters.mp3` |
| 6. Kiểm tra kế hoạch video | Một hoặc nhiều MP3 chương | Bảng kế hoạch gom audio chương cho từng MP4 đầu ra | Truyền đầy đủ danh sách MP3 và `--chapters-per-video` | Chưa có file config riêng cho `plan-videos`; dùng tham số hoặc giá trị mặc định `VIDEO_CHAPTERS_PER_VIDEO` | `python -m src.main plan-videos output/hai_dua_tre/hai_dua_tre_characters.mp3 --chapters-per-video 1` |
| 7. Render MP4 | MP3; ảnh nền; số chương mỗi video; file story config tùy chọn; tiêu đề và metadata tùy chọn | MP4; JPG thumbnail; JSON thông tin render | Truyền đầy đủ `audio_files`, `--background`, `--output-dir`, `--chapters-per-video`, `--title`, `--story-config`, `--translated-file`, `--resume/--no-resume` | Cách config hiện có là `tts --config`; cấu hình `vi_path`, `output_audio`, `background`, `story_config`, `video_title`, `video_output_dir`, `chapters_per_video`, `resume` và `dry_run`, sau đó chạy `tts --config`. Cách này tổng hợp TTS chuẩn, không dùng trực tiếp WAV trong Bước 4 | `python -m src.main render-video output/hai_dua_tre/hai_dua_tre_characters.mp3 --background input/hai_dua_tre/HaiDuaTre.jpg --output-dir output/hai_dua_tre/videos --chapters-per-video 1 --title "Hai đứa trẻ" --story-config configs/story.json --resume` |
| 8. Xem và kiểm tra MP4 | File MP4; JSON/thumbnail của Bước 7 | MP4 đã được xác nhận có dữ liệu, đúng thời lượng và codec | Truyền đầy đủ các đường dẫn MP4 vào ffprobe/ffplay hoặc player | Chưa có config riêng cho bước kiểm tra; cấu hình đường dẫn output render trong config ở Bước 7 | `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 output/hai_dua_tre/videos/hai_dua_tre_characters.mp4` |
| 9. Upload YouTube tùy chọn | MP4; metadata JSON; channel/OAuth; quyền riêng tư | Video đã upload và trạng thái YouTube | Truyền đầy đủ MP4, `--privacy`, `--channel-id`, `--metadata-file`, `--dry-run` nếu chỉ kiểm tra | Đặt giá trị trong file tts/process config hoặc youtube config; dùng `upload_youtube=true`, `youtube_metadata_file`, `youtube_privacy` và chạy quy trình tương ứng | `python -m src.main youtube-upload output/hai_dua_tre/videos/hai_dua_tre_characters.mp4 --privacy private --dry-run` |

## Config vi du

### `input/hai_dua_tre/config/split_voices.json`

```json
{
  "input": "../hai_dua_tre.txt",
  "output": "../hai_dua_tre_segments.json",
  "characters": ["Liên", "An", "Chị Tí", "Cụ Thi", "Bác Siêu"],
  "use_llm": true,
  "llm_model": "qwen3:14b",
  "ollama_url": "http://localhost:11434/api/chat",
  "annotate": "dialogue",
  "max_chars": 400,
  "unknown_as": "narrator",
  "keep_meta": true
}
```

### `input/hai_dua_tre/config/character_voices.json`

```json
{
  "narrator": "Hải Đăng",
  "Liên": "Mai Anh",
  "An": "Quang Sơn",
  "Bác Siêu": "Thanh Bình"
}
```

## Cấu trúc output theo truyện

Mặc định, toàn bộ sản phẩm của một truyện nằm trong một thư mục:

```text
output/
└── hai_dua_tre/
    ├── media/
    │   ├── character.wav
    │   └── hai_dua_tre.mp3
    └── videos/
        ├── hai_dua_tre.jpg
        ├── hai_dua_tre.json
        └── hai_dua_tre.mp4
```

Có thể dùng `--output` hoặc `--output-dir` để ghi đè vị trí cụ thể.
