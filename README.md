# المساعد الطبي الذكي — Offline Medical AI Assistant

**دليل الاستخدام — Arabic & English Guide**

A 100% offline medical textbook chat + live voice assistant. No internet, no
model setup, no PDF upload — everything is bundled by the GitHub Actions
pipeline at build time.

مساعد طبي ذكي يعمل بالكامل دون اتصال. لا إنترنت، لا تثبيت نماذج، لا رفع ملفات PDF —
كل شيء مدمج عبر خط أنابيب GitHub Actions أثناء البناء.

---

## 🚀 Quick Start / البداية

### Option A — Install the packaged executable (recommended)

```bash
# After running the GitHub Actions workflow and downloading the bundle:
chmod +x MedicalAIAssistant
./MedicalAIAssistant
```

The executable contains `assets/` (knowledge DB + all voice/LLM models) — run it
once and it works forever, offline.

### Option B — Run from source

```bash
python -m pip install -r requirements.txt

# 1) Build the knowledge base from your medical PDFs (see below)
python scripts/build_knowledge_db.py --force

# 2) Run the GUI
python src/gui_app.py
```

---

## 📚 1. Adding Medical PDFs / إضافة الكتب الطبية

Place any **selectable-text** PDF textbooks in `medical_pdfs/` (one folder, one
build):

```text
medical_pdfs/
├── basis-al-batiniyya.pdf
├── pharmacology-guide.pdf
└── neurology-handbook.pdf
```

Supported books must have extractable text (no scanned-only images). Then run:

```bash
python scripts/build_knowledge_db.py --force
```

This reads every PDF with PyMuPDF, splits text into sentences, computes
multi-lingual sentence embeddings (`fastembed`) and writes a single immutable
database:

```
assets/books_knowledge.db     # SQLite + FTS5 + full text index
assets/book_embeddings.npy    # float32 cosine-normalized matrix (loaded in RAM)
assets/medical_glossary.json  # extracted terms for STT priming
assets/refusal_ar.wav         # ~0 ms refusal clip
```

### Optional: seed real PDFs before build / اختيار كتب قبل البناء

Copy your own `.pdf` files into `medical_pdfs/`. The pipeline only packages the
DB + models — **raw PDFs are excluded from the final executable**.

الكتب الواقعية تُوضع في مجلد `medical_pdfs/`. خط الأنابيب يعبئ فقط قاعدة الكتب والنماذج —
ملفات PDF الخام **لا تُدرج** في البرنامج النهائي.

---

## 🤖 2. Running the GitHub Actions Build Pipeline / تشغيل خط أنابيب البناء

Push to `main`/`develop` or trigger manually (workflow_dispatch):

1. Go to the repository → **Actions** tab → **Build & Release Offline Medical AI** → **Run workflow**
2. (Optional) select **Create a GitHub release: yes**
3. Click **Run workflow**

The pipeline runs 5 stages:

| Stage | What it does |
|---|---|
| **Build knowledge base** | `medical_pdfs/*.pdf` → `assets/books_knowledge.db` + embeddings |
| **Download voice/LLM models** | Phi-3-mini GGUF Q4_K_M (~3.8 GB) + Silero VAD + faster-whisper int8 + Piper Arabic ONNX → `assets/voice_models/` |
| **CI — lint & guardrail tests** | pytest `test_strict_guardrail.py`: < 15 ms retrieval, hard out-of-scope rejection |
| **Package executable** | PyInstaller single-file `MedicalAIAssistant` with all assets bundled |
| **Release** | GitHub release with bundle + sha256 checksums |

All four models are **verified by SHA-256** in `scripts/download_models.py` — the
build fails if any model is corrupted.

---

## 🔒 3. How the Strict Guardrail Works / كيف يعمل البوابة الصارمة

1. **Pre-LLM hard gate:** before any LLM call, an in-memory hybrid search
   (NumPy cosine similarity + SQLite FTS5 keyword match) scores the query.
2. If the best score ≤ `STRICT_SIMILARITY_THRESHOLD` (**0.62**), the LLM is never
   invoked — the exact refusal is returned in < 10 ms:

   > هذه المعلومة غير مذكورة في الكتب الطبية المدمجة في التطبيق.

3. If valid, the LLM (Phi-3-mini GGUF, pinned in RAM) generates with
   `temperature=0.0`, `top_p=0.1`, `repeat_penalty=1.1`.

4. Every answer ends with an **exact citation badge**:

   ```
   [المصدر: أساسيات الطب الباطني - صفحة 13]
   ```

   Click the badge → verbatim paragraph opens in a modal.

### Tests prove the contract / الاختبارات تثبت العقد

```bash
cd tests
python -m pytest test_strict_guardrail.py -v
```

T2: retrieval < 15 ms over the full matrix · T4: out-of-scope → hard blocked,
LLM never called · T5: exact refusal string · T6: badge format verified.

---

## 🎙 4. Live Voice Mode / الوضع الصوتي المباشر

Press **🎤 صوت مباشر** and speak:

- **Silero VAD** listens on 30 ms frames; transcribes after 300 ms of silence.
- **faster-whisper** CTranslate2 int8 transcribes the RAM buffer in < 200 ms.
- **Piper Arabic streaming TTS** plays the first clause in < 40 ms (clause splitter)
  while the LLM streams the rest → first word speaks in < 0.6 s.
- **Instant barge-in:** speaking while the AI talks cuts its speech in < 50 ms
  and a new transcription begins immediately.

الوضع الصوتي الكامل:
- **Silero VAD** يستمع في إطارات 30 مل ثانية، ينقل الكلام بعد 300 مل ثانية صمت.
- **faster-whisper** int8 ينقل النص من الذاكرة في < 200 مل ثانية.
- **Piper TTS** العربية يتحدث أول جملة في < 40 مل ثانية.
- **المقاطعة الفورية:** التحدث أثناء كلام الذكاء الاصطناعي يوقفه في < 50 مل ثانية.

---

## 📦 5. Repository Structure / هيكل المستودع

```text
medical-ai-assistant/
├── .github/workflows/build_release.yml   # 5-stage CI/CD pipeline
├── medical_pdfs/                         # place your PDFs here
├── assets/                               # DB + models (built by CI)
├── scripts/
│   ├── build_knowledge_db.py             # PDFs → SQLite + embeddings
│   └── download_models.py                # downloads + verifies all models
├── src/
│   ├── __init__.py
│   ├── config.py                         # PyInstaller-safe paths, thresholds
│   ├── retriever.py                      # <15 ms hybrid search + guardrail
│   ├── llm_engine.py                     # offline GGUF + clause streaming
│   ├── live_voice.py                     # VAD + STT + streaming TTS + barge-in
│   └── gui_app.py                        # PySide6 chat + voice orb
├── tests/test_strict_guardrail.py        # guardrail unit tests
├── requirements.txt
└── README.md
```

---

## 🛠 Requirements

Python ≥ 3.10 with `pip install -r requirements.txt`. No external model
installation — `download_models.py` fetches everything into `assets/voice_models/`.

For Android build (local): the environment provides JDK 17, Android SDK 36,
ARM64 Build Tools 35.0.0, Gradle 8.14.3, and an offline Maven cache.

---

## License

MIT — free for personal and medical-education use.