# PDF Extraction and Multi-Language OCR Engine

Automated, cloud-scale PDF text, layout, and OCR extraction pipeline designed specifically to run all computationally intensive processing on **GitHub Actions cloud runners**.

---

## 🌟 Key Features

- **Cloud Runner Execution:** Heavy OCR processing (rendering, Tesseract OCR, layout parsing) executes entirely inside GitHub Actions Ubuntu runners. The local development environment is used solely for development, testing, and workflow management.
- **Dual-Mode Extraction:**
  - Fast digital layout and text block extraction for digital pages.
  - Tesseract OCR (`hin+eng`) with automatic deskew and image enhancement for scanned/low-density pages.
- **Structured Content Parsing:** Automatically identifies and extracts:
  - Multiple Choice Questions (MCQs), options (A, B, C, D / क, ख, ग, घ), and answers.
  - Practice exercises and sentence drills.
  - Tables with detected headers and rows.
  - Headings and body paragraphs.
- **Large PDF Handling & Resumability:**
  - Streams page-by-page to keep RAM usage minimal (< 500 MB) even at 300 DPI.
  - Divides pages into configurable batches (default 25 pages).
  - Saves intermediate checkpoints (`.checkpoints/`) keyed by input SHA256 checksum to enable resumability across runs.
- **6 Mandatory Output Artifacts:**
  1. `extracted_content.json` - Complete hierarchical structured document JSON.
  2. `extracted_text.txt` - Clean formatted text with page demarcations.
  3. `ocr_review_report.json` - Review report flagging low-confidence pages and suspect words.
  4. `extraction_manifest.json` - Processing manifest with input checksums, environment details, and output file hashes.
  5. `extraction_stats.json` - Statistical summary of page coverage, word counts, and extracted entities.
  6. `extraction.log` - Full timestamped processing logs.
- **Automated Verification:** Standalone validator (`scripts/validate_extraction.py`) verifies 100% page coverage, schema conformance, and generates a GitHub Actions Step Summary.
- **Downloadable Artifacts:** All 6 outputs are automatically zipped and uploaded to GitHub Actions artifacts (configurable retention, default 30 days), preserved even if subsequent Git steps fail.
- **Optional Git Commit:** Generated outputs can optionally be committed to the `output/` directory using standard `GITHUB_TOKEN` permissions without overwriting or force-pushing.

---

## 📁 Repository Structure

```text
.
├── .github/
│   └── workflows/
│       ├── pdf-extraction.yml     # Primary GitHub Actions cloud workflow
│       └── ocr_convert.yml        # Legacy converter workflow
├── input/
│   └── Spoken_English_Guru_Practice_Exercises_Quizzes_eBook_1.pdf  # Target document (18.5 MB, 818 pages)
├── scripts/
│   ├── extract_pdf.py             # Core extraction and OCR engine
│   └── validate_extraction.py     # Independent JSON schema and page-coverage validator
├── requirements.txt               # Python dependencies for runner
├── process_ocr.py                 # Backward-compatible entrypoint
├── .gitignore                     # Excludes temporary images, cache & checkpoints
└── README.md                      # Documentation
```

---

## 🚀 How to Run via GitHub Actions

### Option A: From the GitHub Web UI (Recommended)

1. Open the repository on GitHub: [https://github.com/kiskurahul400-web/PDFOCR](https://github.com/kiskurahul400-web/PDFOCR)
2. Click on the **Actions** tab at the top.
3. In the left sidebar, select **"PDF Extraction and OCR Processing"**.
4. Click the **"Run workflow"** dropdown on the right.
5. Configure the desired execution parameters:
   - **Path to input PDF file:** `input/Spoken_English_Guru_Practice_Exercises_Quizzes_eBook_1.pdf` (or leave default)
   - **Page range to process:** e.g. `1-25`, `1-50`, `51-100`, or `all`
   - **Tesseract OCR languages:** `hin+eng`
   - **Image rendering DPI:** `300`
   - **Batch size:** `25`
   - **Commit results:** Check to commit output to repo, or uncheck to only produce downloadable artifacts.
   - **Artifact retention days:** `30`
6. Click the green **"Run workflow"** button.

### Option B: Triggering via GitHub CLI (`gh`)

From your terminal or local environment:

```bash
# Process first 25 pages
gh workflow run pdf-extraction.yml -f page_range="1-25" -f ocr_languages="hin+eng"

# Process pages 26 to 50
gh workflow run pdf-extraction.yml -f page_range="26-50" -f ocr_languages="hin+eng"

# Process entire document with results committed
gh workflow run pdf-extraction.yml -f page_range="all" -f commit_results="true"
```

To watch workflow progress:
```bash
gh run watch
```

---

## 📥 Retrieving Extraction Output Files

Once the workflow finishes:
1. Navigate to the completed run in the **Actions** tab.
2. Scroll to the bottom to the **Artifacts** section.
3. Click on `pdf-extraction-artifacts-run<RUN_NUMBER>` to download the ZIP package containing:
   - `extracted_content.json`
   - `extracted_text.txt`
   - `ocr_review_report.json`
   - `extraction_manifest.json`
   - `extraction_stats.json`
   - `extraction.log`
4. Alternatively, use the GitHub CLI:
   ```bash
   gh run download <RUN_ID> -n pdf-extraction-artifacts-run<RUN_NUMBER>
   ```

---

## 🛠 Local Development & Testing

Only lightweight development, configuration, and syntax validation are performed locally:

```bash
# Check Python syntax
python3 -m py_compile scripts/extract_pdf.py scripts/validate_extraction.py

# Test CLI help
python3 scripts/extract_pdf.py --help
python3 scripts/validate_extraction.py --help
```
