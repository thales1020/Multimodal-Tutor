# Tài liệu Quản lý và Xây dựng Dữ liệu (Data Management Document)

Tài liệu này đặc tả cấu trúc dữ liệu, nguồn thu thập và quy trình tiền xử lý phục vụ cho dự án AI Teaching Assistant dựa trên khóa học CS50x, tuân thủ các yêu cầu về quản lý dữ liệu có cấu trúc và có trách nhiệm.

## 1. Nguồn Dữ liệu (Data Sourcing)

* **Nguồn gốc:** Trích xuất từ 4-5 video bài giảng công khai của khóa học CS50x trên YouTube (ví dụ: Week 1 - C, Week 4 - Memory, Week 6 - Python).
* **Ngôn ngữ:** Tiếng Anh (dựa trên phụ đề gốc của video).
* **Quy mô dự kiến:** Khoảng 2-3 giờ video tổng hợp, tạo ra khoảng 200-300 chunks ngữ cảnh và 100-150 cặp câu hỏi đánh giá (QA pairs).

---

## 2. Cấu trúc Dữ liệu (Data Schema)

Hệ thống duy trì hai tập dữ liệu độc lập: Dữ liệu ngữ cảnh (lưu trong Vector DB) và Dữ liệu đánh giá (dùng để kiểm thử hệ thống).

### 2.1. Cấu trúc Chunk (Vector DB Schema)

Mỗi bản ghi đại diện cho một phân đoạn video dài 60 giây (1 phút), kết hợp thông tin văn bản và thị giác.

```json
{
  "chunk_id": "cs50_week4_001",
  "video_id": "week4_memory",
  "start_time": 120.0,
  "end_time": 180.0,
  "transcript": "Let's look at how swap works with pointers...",
  "ocr_text": "void swap(int *a, int *b) { int tmp = *a; ... }",
  "visual_description": "David Malan is pointing to a memory diagram on the chalkboard showing addresses 0x123 and 0x456.",
  "metadata": {
    "week": "Week 4",
    "language": "C"
  }
}
```

### 2.2. Cấu trúc Tập đánh giá (QA Dataset Schema)

Được sử dụng làm "Ground Truth" để đo lường hiệu suất mô hình.

```json
{
  "question_id": "q_015",
  "question": "Hàm swap trong C hoạt động như thế nào với con trỏ theo giải thích ở tuần 4?",
  "target_answer": "Hàm nhận vào địa chỉ của hai biến (con trỏ), sau đó hoán đổi giá trị tại các địa chỉ đó thông qua một biến tạm.",
  "ground_truth_video_id": "week4_memory",
  "ground_truth_window": [120.0, 150.0],
  "question_type": "code_explanation" 
}
```

---

## 3. Quy trình Xây dựng Dữ liệu Bán tự động (Semi-automated Workflow)

Quy trình này đảm bảo việc xử lý, làm sạch và loại bỏ nhiễu dữ liệu một cách có hệ thống.

### Bước 1: Thu thập và Phân mảnh (Ingestion & Chunking)

1. **Tải phụ đề:** Sử dụng `youtube-transcript-api` tải phụ đề gốc.
2. **Phân mảnh (Chunking):** Nhóm phụ đề thành các đoạn cố định 60 giây (1 phút).
3. **Trích xuất hình ảnh:** Sử dụng `yt-dlp` và `ffmpeg` cắt chính xác 1 keyframe tại giây thứ 30 của mỗi chunk.

### Bước 2: Trích xuất Đa phương thức và Làm sạch (Multimodal Extraction & Cleaning)

1. **Gọi API VLM:** Gửi từng keyframe qua API của Vision-Language Model (GPT-4o-mini/Gemini Flash).
2. **Prompt quy định:** Yêu cầu VLM trích xuất chữ viết (OCR) và mô tả cấu trúc hình ảnh (sơ đồ, luồng dữ liệu).
3. **Làm sạch (Cleaning):** Lọc bỏ các chunk chứa hình ảnh nhiễu (chuyển cảnh đen, mờ) hoặc xử lý dữ liệu bị thiếu nếu VLM không nhận diện được thông tin hữu ích.
4. **Ẩn danh (Anonymization):** Đảm bảo không lưu trữ các thông tin nhận dạng cá nhân (PII) không liên quan vào cơ sở dữ liệu.

### Bước 3: Khởi tạo và Kiểm duyệt QA (QA Generation & Human-in-the-loop)

1. **Tự động tạo:** Đưa dữ liệu ngữ cảnh (Chunk JSON) vào LLM để tự động sinh 3-5 câu hỏi phức tạp đòi hỏi suy luận từ cả hình ảnh và lời nói.
2. **Rà soát thủ công:** Con người kiểm tra từng câu hỏi, điều chỉnh `target_answer` cho chuẩn xác, và loại bỏ các câu hỏi mang tính thiên kiến hoặc sai lệch.

---

## 4. Phân chia Dữ liệu (Data Splits)

Tập QA Dataset gồm 100-150 câu hỏi sẽ được phân chia hợp lý để phục vụ phát triển và kiểm thử.

* **Tập Phát triển (Dev Set):** Gồm 50 câu hỏi. Được sử dụng để kiểm thử prompt, điều chỉnh `similarity_threshold`, và tinh chỉnh tham số `top_k` của RAG.
* **Tập Kiểm thử (Test Set):** Gồm 50-100 câu hỏi (Held-out). Được bảo mật hoàn toàn trong giai đoạn phát triển và chỉ dùng một lần ở cuối dự án để tính toán điểm số F1, BERTScore và Grounding Accuracy phục vụ báo cáo. Việc tách biệt này đảm bảo kết quả đánh giá không bị thiên lệch (overfitting).

5. Cấu trúc Thư mục Dữ liệu (Data Directory Structure)Toàn bộ quá trình thu thập, xử lý và lưu trữ dữ liệu được quản lý tập trung trong thư mục data/ ở thư mục gốc của dự án.Để tuân thủ tiêu chuẩn kỹ thuật, các tệp dữ liệu lớn (video, ảnh) sẽ không được đẩy lên Git repository mà chỉ được tạo ra thông qua các kịch bản tự động (download scripts).
data/
├── raw/                      # Chứa dữ liệu thô tải về (Không commit lên Git)
│   ├── videos/               # Tệp .mp4 tải từ YouTube (CS50x)
│   └── transcripts/          # Tệp phụ đề gốc .json hoặc .vtt
├── interim/                  # Dữ liệu đang xử lý trung gian (Không commit)
│   ├── frames/               # Các keyframe (.jpg) được trích xuất từ video
│   └── text_chunks/          # Phụ đề đã được cắt thành đoạn 60 giây (1 phút)
├── processed/                # Dữ liệu đã xử lý hoàn chỉnh (Có thể commit)
│   ├── vector_chunks.json    # File JSON tổng hợp (Transcript + OCR + Visual Description)
│   └── metadata.json         # File mapping video_id với thông tin (Tuần, Ngôn ngữ)
├── evaluation/               # Tập dữ liệu QA để đánh giá hệ thống (Bắt buộc commit)
│   ├── dev_set.json          # 50 câu hỏi để tinh chỉnh prompt và thông số RAG
│   └── test_set.json         # 50-100 câu hỏi held-out để lấy điểm benchmark cuối cùng
└── chroma_db/                # Thư mục lưu trữ local của Vector Database (Không commit)

Quy định Quản lý Phiên bản (Version Control Rules)Tệp .gitignore: Bắt buộc phải thêm các thư mục data/raw/, data/interim/, và data/chroma_db/ vào .gitignore để tránh phình to kích thước repository.Kịch bản tái tạo (Reproducibility): Thư mục src/data/ sẽ chứa các script Python (ví dụ: download_videos.py, extract_frames.py) để hội đồng đánh giá có thể tự động tải lại toàn bộ dữ liệu thô và xây dựng lại thư mục data/ từ đầu nếu cần.

---

## 6. Hướng dẫn Sử dụng (Usage Guide)

### 6.1 Trích playlist CS50x từ YouTube

```bash
# Bước 1: Extract tất cả video metadata từ official CS50x playlist
python src/retrieval/extract_playlist.py \
  --playlist-url "https://www.youtube.com/playlist?list=PLhQjrBD2T380hlTqAU8HfvVepCcjCqTg6" \
  --output data/playlist_metadata.json
```

### 6.2 Chạy Ingestion & Chunking Pipeline

```bash
# Với metadata (tên chunks theo week):
python src/retrieval/ingestion_chunking.py \
  --metadata data/playlist_metadata.json \
  --url-file cs50_urls.txt \
  --chunk-seconds 60
```Tài liệu của bạn về cơ bản đã đầy đủ và chính xác. Tuy nhiên, có hai điểm cần chỉnh sửa:
1. **Lỗi định dạng Markdown:** Mục 5 và phần "Quy định Quản lý Phiên bản" đang bị mất định dạng heading và list.
2. **Sai vị trí script:** Ở mục 6, các script tải dữ liệu đang trỏ vào `src/retrieval/`. Theo kiến trúc đã thống nhất, thư mục `retrieval` dành cho RAG logic. Các script này nên nằm trong thư mục `data/`.

Dưới đây là bản Markdown đã được chuẩn hóa và sửa lỗi để bạn sao chép trực tiếp:

***

# Tài liệu Quản lý và Xây dựng Dữ liệu (Data Management Document)

Tài liệu này đặc tả cấu trúc dữ liệu, nguồn thu thập và quy trình tiền xử lý phục vụ cho dự án AI Teaching Assistant dựa trên khóa học CS50x, tuân thủ các yêu cầu về quản lý dữ liệu có cấu trúc và có trách nhiệm.

## 1. Nguồn Dữ liệu (Data Sourcing)

* **Nguồn gốc:** Trích xuất từ 4-5 video bài giảng công khai của khóa học CS50x trên YouTube (ví dụ: Week 1 - C, Week 4 - Memory, Week 6 - Python).
* **Ngôn ngữ:** Tiếng Anh (dựa trên phụ đề gốc của video).
* **Quy mô dự kiến:** Khoảng 2-3 giờ video tổng hợp, tạo ra khoảng 200-300 chunks ngữ cảnh và 100-150 cặp câu hỏi đánh giá (QA pairs).

---

## 2. Cấu trúc Dữ liệu (Data Schema)

Hệ thống duy trì hai tập dữ liệu độc lập: Dữ liệu ngữ cảnh (lưu trong Vector DB) và Dữ liệu đánh giá (dùng để kiểm thử hệ thống).

### 2.1. Cấu trúc Chunk (Vector DB Schema)

Mỗi bản ghi đại diện cho một phân đoạn video dài 60 giây (1 phút), kết hợp thông tin văn bản và thị giác.

```json
{
  "chunk_id": "cs50_week4_001",
  "video_id": "week4_memory",
  "start_time": 120.0,
  "end_time": 180.0,
  "transcript": "Let's look at how swap works with pointers...",
  "ocr_text": "void swap(int *a, int *b) { int tmp = *a; ... }",
  "visual_description": "David Malan is pointing to a memory diagram on the chalkboard showing addresses 0x123 and 0x456.",
  "metadata": {
    "week": "Week 4",
    "language": "C"
  }
}
```

### 2.2. Cấu trúc Tập đánh giá (QA Dataset Schema)

Được sử dụng làm "Ground Truth" để đo lường hiệu suất mô hình.

```json
{
  "question_id": "q_015",
  "question": "Hàm swap trong C hoạt động như thế nào với con trỏ theo giải thích ở tuần 4?",
  "target_answer": "Hàm nhận vào địa chỉ của hai biến (con trỏ), sau đó hoán đổi giá trị tại các địa chỉ đó thông qua một biến tạm.",
  "ground_truth_video_id": "week4_memory",
  "ground_truth_window": [120.0, 150.0],
  "question_type": "code_explanation" 
}
```

---

## 3. Quy trình Xây dựng Dữ liệu Bán tự động (Semi-automated Workflow)

Quy trình này đảm bảo việc xử lý, làm sạch và loại bỏ nhiễu dữ liệu một cách có hệ thống.

### Bước 1: Thu thập và Phân mảnh (Ingestion & Chunking)

1. **Tải phụ đề:** Sử dụng `youtube-transcript-api` tải phụ đề gốc.
2. **Phân mảnh (Chunking):** Nhóm phụ đề thành các đoạn cố định 60 giây (1 phút).
3. **Trích xuất hình ảnh:** Sử dụng `yt-dlp` và `ffmpeg` cắt chính xác 1 keyframe tại giây thứ 30 của mỗi chunk.

### Bước 2: Trích xuất Đa phương thức và Làm sạch (Multimodal Extraction & Cleaning)

1. **Gọi API VLM:** Gửi từng keyframe qua API của Vision-Language Model (GPT-4o-mini/Gemini Flash).
2. **Prompt quy định:** Yêu cầu VLM trích xuất chữ viết (OCR) và mô tả cấu trúc hình ảnh (sơ đồ, luồng dữ liệu).
3. **Làm sạch (Cleaning):** Lọc bỏ các chunk chứa hình ảnh nhiễu (chuyển cảnh đen, mờ) hoặc xử lý dữ liệu bị thiếu nếu VLM không nhận diện được thông tin hữu ích.
4. **Ẩn danh (Anonymization):** Đảm bảo không lưu trữ các thông tin nhận dạng cá nhân (PII) không liên quan vào cơ sở dữ liệu.

### Bước 3: Khởi tạo và Kiểm duyệt QA (QA Generation & Human-in-the-loop)

1. **Tự động tạo:** Đưa dữ liệu ngữ cảnh (Chunk JSON) vào LLM để tự động sinh 3-5 câu hỏi phức tạp đòi hỏi suy luận từ cả hình ảnh và lời nói.
2. **Rà soát thủ công:** Con người kiểm tra từng câu hỏi, điều chỉnh `target_answer` cho chuẩn xác, và loại bỏ các câu hỏi mang tính thiên kiến hoặc sai lệch.

---

## 4. Phân chia Dữ liệu (Data Splits)

Tập QA Dataset gồm 100-150 câu hỏi sẽ được phân chia hợp lý để phục vụ phát triển và kiểm thử.

* **Tập Phát triển (Dev Set):** Gồm 50 câu hỏi. Được sử dụng để kiểm thử prompt, điều chỉnh `similarity_threshold`, và tinh chỉnh tham số `top_k` của RAG.
* **Tập Kiểm thử (Test Set):** Gồm 50-100 câu hỏi (Held-out). Được bảo mật hoàn toàn trong giai đoạn phát triển và chỉ dùng một lần ở cuối dự án để tính toán điểm số F1, BERTScore và Grounding Accuracy phục vụ báo cáo. Việc tách biệt này đảm bảo kết quả đánh giá không bị thiên lệch (overfitting).

---

## 5. Cấu trúc Thư mục Dữ liệu (Data Directory Structure)

Toàn bộ quá trình thu thập, xử lý và lưu trữ dữ liệu được quản lý tập trung trong thư mục `data/` ở thư mục gốc của dự án. Để tuân thủ tiêu chuẩn kỹ thuật, các tệp dữ liệu lớn (video, ảnh) sẽ không được đẩy lên Git repository mà chỉ được tạo ra thông qua các kịch bản tự động (download scripts).

```text
data/
├── raw/                      # Chứa dữ liệu thô tải về (Không commit lên Git)
│   ├── videos/               # Tệp .mp4 tải từ YouTube (CS50x)
│   └── transcripts/          # Tệp phụ đề gốc .json hoặc .vtt
├── interim/                  # Dữ liệu đang xử lý trung gian (Không commit)
│   ├── frames/               # Các keyframe (.jpg) được trích xuất từ video
│   └── text_chunks/          # Phụ đề đã được cắt thành đoạn 60 giây (1 phút)
├── processed/                # Dữ liệu đã xử lý hoàn chỉnh (Có thể commit)
│   ├── vector_chunks.json    # File JSON tổng hợp (Transcript + OCR + Visual Description)
│   └── metadata.json         # File mapping video_id với thông tin (Tuần, Ngôn ngữ)
├── evaluation/               # Tập dữ liệu QA để đánh giá hệ thống (Bắt buộc commit)
│   ├── dev_set.json          # 50 câu hỏi để tinh chỉnh prompt và thông số RAG
│   └── test_set.json         # 50-100 câu hỏi held-out để lấy điểm benchmark cuối cùng
└── chroma_db/                # Thư mục lưu trữ local của Vector Database (Không commit)
```

### Quy định Quản lý Phiên bản (Version Control Rules)
* **Tệp `.gitignore`:** Bắt buộc phải thêm các thư mục `data/raw/`, `data/interim/`, và `data/chroma_db/` vào `.gitignore` để tránh phình to kích thước repository.
* **Kịch bản tái tạo (Reproducibility):** Thư mục `data/` sẽ chứa các script Python (ví dụ: `download_videos.py`, `extract_frames.py`) để hội đồng đánh giá có thể tự động tải lại toàn bộ dữ liệu thô và xây dựng lại thư mục `data/` từ đầu nếu cần.

---

## 6. Hướng dẫn Sử dụng (Usage Guide)

### 6.1 Trích playlist CS50x từ YouTube

```bash
# Bước 1: Extract tất cả video metadata từ official CS50x playlist
python data/extract_playlist.py \
  --playlist-url "https://www.youtube.com/playlist?list=PLhQjrBD2T380hlTqAU8HfvVepCcjCqTg6" \
  --output data/playlist_metadata.json
```

### 6.2 Chạy Ingestion & Chunking Pipeline

```bash
# Với metadata (tên chunks theo week):
python data/ingestion_chunking.py \
  --metadata data/playlist_metadata.json \
  --url-file cs50_urls.txt \
  --chunk-seconds 60
```

**Output:**
- `data/interim/text_chunks/chunks.json` - 150+ chunks với format: `{chunk_id: "cs50_week1_000", video_id: "week1_c", ...}`
- `data/interim/frames/` - 150+ `.jpg` keyframes tên: `cs50_week1_000.jpg`, `cs50_week1_001.jpg`, ...

**Output:**
- `data/interim/text_chunks/chunks.json` - 150+ chunks với format: `{chunk_id: "cs50_week1_000", video_id: "week1_c", ...}`
- `data/interim/frames/` - 150+ `.jpg` keyframes tên: `cs50_week1_000.jpg`, `cs50_week1_001.jpg`, ...
