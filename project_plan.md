Dưới đây là bản kế hoạch dự án hoàn chỉnh, súc tích và cập nhật chiến lược dữ liệu tự xây dựng từ khóa học CS50x.

### 1. Định nghĩa bài toán (Business Problem)
* [cite_start]**Bối cảnh:** Các video bài giảng CS50x kéo dài hơn 2 giờ, khiến học viên khó tìm kiếm các đoạn mã, sơ đồ bộ nhớ hoặc hình vẽ minh họa cụ thể [cite: 8-10, 12].
* [cite_start]**Mục tiêu:** Thiết kế hệ thống AI Agentic truy xuất và trả lời câu hỏi dựa trên nội dung đa phương thức (phụ đề và hình ảnh) của khóa học [cite: 4-5].
* **Chỉ số thành công:** Giảm 40% thời gian tra cứu; Độ chính xác (F1/BERTScore) > 80%; Grounding Accuracy > 80%; [cite_start]Độ trễ (Latency) < 8s [cite: 14-16].

### 2. Hạ tầng và Công cụ (Infrastructure)
* [cite_start]**Ngôn ngữ & Quản lý:** Python, Git [cite: 19-20].
* **Framework:** LlamaIndex hoặc LangChain.
* **Mô hình AI:** GPT-4o-mini hoặc Gemini 1.5 Flash (VLM/LLM); [cite_start]Whisper (ASR, nếu cần)[cite: 49].
* **Lưu trữ:** ChromaDB hoặc Qdrant (Vector DB).

### 3. Cấu trúc mã nguồn (Repository Structure)
[cite_start]Tuân thủ chuẩn mô-đun hóa dự án [cite: 21, 150-163].
```text
project-root/
├── src/                
│   ├── agents/         # Router logic & Agent classifier
│   ├── retrieval/      # RAG pipeline & Chunking
│   ├── models/         # VLM/LLM wrappers & Prompts
│   └── api/            # FastAPI endpoints
├── data/               # Script yt-dlp, ffmpeg, transcript API
├── configs/            # retrieval_config.yaml
├── tests/              # Đánh giá hệ thống
├── requirements.txt    # Quản lý thư viện
└── README.md           # Hướng dẫn cài đặt
```

### 4. Quản lý dữ liệu (Data Management)
[cite_start]Chiến lược xây dựng dữ liệu tự động hóa một phần (Semi-automated) từ 4-5 video CS50x tiêu biểu [cite: 36-39].

* **Dữ liệu Ngữ cảnh (Vector DB Schema):** * Phân mảnh video thành các chunk 60 giây.
    * Trích xuất 1 keyframe mỗi chunk, dùng VLM để lấy văn bản OCR và mô tả cấu trúc.
    * Cấu trúc lưu trữ: `chunk_id`, `video_id`, `start_time`, `end_time`, `transcript`, `ocr_text`, `visual_description`.
* **Dữ liệu Đánh giá (QA Dataset):**
    * Số lượng: 100-150 câu hỏi. [cite_start]Phân chia thành Dev set (50 câu) và Test set (50-100 câu)[cite: 39, 45].
    * Quy trình: Đưa các chunk vào LLM để sinh câu hỏi tự động. Rà soát thủ công để loại bỏ lỗi và chốt đáp án chuẩn.
    * Cấu trúc lưu trữ: `question_id`, `question`, `target_answer`, `ground_truth_video_id`, `ground_truth_window`, `question_type`.

### 5. Thành phần Agentic AI và RAG
* [cite_start]**Phân loại ý định (Router Agent):** LLM nhận diện câu hỏi và quyết định luồng xử lý [cite: 81-86].
* **Truy xuất (Retrieval):** Text-only Retrieval trên Vector DB. Truy xuất `top_k = 3-5` chunks với `similarity_threshold = 0.45`.
* [cite_start]**Cơ chế Fallback:** Nếu ngữ cảnh không đủ độ tin cậy, kích hoạt Web Search hoặc phản hồi "Không tìm thấy trong bài giảng"[cite: 83].
* **Tổng hợp (Synthesis):** Trả lời câu hỏi và đính kèm timestamp cụ thể.

### 6. Triển khai và Học liên tục (Deployment & Monitoring)
* [cite_start]**Triển khai:** Hệ thống vận hành qua REST API (FastAPI) [cite: 63-64]. [cite_start]Áp dụng Cache layer để tối ưu latency[cite: 70].
* **Học liên tục:** Lưu trữ lịch sử câu hỏi và phản hồi người dùng. [cite_start]Thiết lập quy trình phát hiện suy giảm hiệu suất (model drift) [cite: 92-101].

### 7. Đạo đức và Độ tin cậy (Ethics & Robustness)
* [cite_start]**Minh bạch (Explainability):** Mọi câu trả lời bắt buộc có trích dẫn timestamp để người dùng kiểm chứng[cite: 130].
* [cite_start]**Robustness:** Hệ thống xử lý an toàn các câu hỏi nằm ngoài phạm vi khóa học (out-of-domain)[cite: 105, 112].
* [cite_start]**Quyền riêng tư:** Ẩn danh các thông tin định danh (PII) trong quá trình xử lý phụ đề và hình ảnh [cite: 108-109].

### 8. Lộ trình thực hiện (Timeline & Deliverables)
* **Tuần 1:** Tải video/phụ đề CS50x. Phân mảnh 60s và trích xuất keyframes.
* **Tuần 2:** Chạy batch VLM tạo `ocr_text`, `visual_description`. Xây dựng tập QA (Dev/Test set).
* **Tuần 3:** Xây dựng luồng Retrieval và thiết lập Vector DB.
* **Tuần 4:** Phát triển logic Router Agent và Fallback.
* **Tuần 5:** Triển khai FastAPI, thiết lập Cache, đo lường Latency.
* **Tuần 6:** Chạy Test set. [cite_start]Nộp mã nguồn (Git), Báo cáo PDF (10-15 trang) và Slide thuyết trình [cite: 119-121, 133-144, 146-197].