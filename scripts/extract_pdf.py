#!/usr/bin/env python3
"""
PDF Extraction and OCR Processing Engine
Designed for execution on GitHub Actions runners.

Features:
- Dual extraction mode: Digital embedded text + Tesseract OCR fallback / hybrid.
- Multi-language OCR support (defaults to Hindi + English: hin+eng).
- Structured extraction: MCQs, Practice Exercises, Tables, Headings, and Paragraphs.
- Large document batching, memory safety, and checkpoint-based resumability.
- Generates all 6 required output files:
    1. extracted_content.json
    2. extracted_text.txt
    3. ocr_review_report.json
    4. extraction_manifest.json
    5. extraction_stats.json
    6. extraction.log
"""

import os
import sys
import gc
import re
import json
import time
import hashlib
import logging
import argparse
import platform
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

# Third-party imports with graceful handling
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from pytesseract import Output
except ImportError:
    pytesseract = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def setup_logger(log_file: str) -> logging.Logger:
    """Sets up unified logging to both file and stdout."""
    logger = logging.getLogger("PDFExtraction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    return logger


def calculate_sha256(filepath: str) -> str:
    """Calculates SHA256 checksum of a file efficiently in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_page_range(range_str: str, max_pages: int) -> List[int]:
    """
    Parses a page range string (e.g., 'all', '1-50', '1,3,5-10') into a list of 1-based page numbers.
    """
    if not range_str or range_str.strip().lower() == "all":
        return list(range(1, max_pages + 1))

    pages = set()
    parts = [p.strip() for p in range_str.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = max(1, int(start_str.strip()))
                end = min(max_pages, int(end_str.strip()))
                if start <= end:
                    pages.update(range(start, end + 1))
            except ValueError:
                continue
        else:
            try:
                p = int(part.strip())
                if 1 <= p <= max_pages:
                    pages.add(p)
            except ValueError:
                continue

    sorted_pages = sorted(list(pages))
    return sorted_pages if sorted_pages else list(range(1, max_pages + 1))


def deskew_image_if_needed(cv_img: np.ndarray, logger: logging.Logger) -> Tuple[np.ndarray, float]:
    """Calculates skew angle and straightens image if tilt exceeds threshold."""
    if cv2 is None or np is None or cv_img is None:
        return cv_img, 0.0

    try:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
        inv = cv2.bitwise_not(gray)
        thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        angles = []
        if lines is not None:
            for line in lines:
                if len(line) > 0 and len(line[0]) == 4:
                    x1, y1, x2, y2 = line[0]
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -45 < angle < 45:
                        angles.append(angle)

        skew_angle = float(np.median(angles)) if angles else 0.0
        if abs(skew_angle) > 0.5:
            h, w = cv_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            cv_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return cv_img, skew_angle
        return cv_img, 0.0
    except Exception as e:
        logger.warning(f"Deskew failed: {e}")
        return cv_img, 0.0


def extract_mcqs_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Parses Multiple Choice Questions (MCQs), their options, and answers from text.
    Handles English and Hindi formats:
    - (A) / (B) / (C) / (D)
    - (a) / (b) / (c) / (d)
    - (क) / (ख) / (ग) / (घ)
    - Answers: 'Answer: (B)', 'Ans: C', 'उत्तर: ख'
    """
    mcqs = []
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # Pattern to match question starts: "Q1.", "Question 1:", "1.", "MCQ 1:"
    q_pattern = re.compile(r"^(?:Q(?:uestion)?\.?\s*(\d+)[:.]?|(\d+)[\.\)]\s+)(.*)", re.IGNORECASE)
    # Pattern for options
    opt_pattern = re.compile(r"^(?:\(([A-Da-dक-घ])\)|([A-Da-dक-घ])[\.\)])\s*(.*)")
    # Pattern for answer lines
    ans_pattern = re.compile(r"^(?:Ans(?:wer)?|उत्तर)\s*[:\-–]?\s*\(?([A-Da-dक-घ1-4])\)?(?:\s*[:\-–]?\s*(.*))?", re.IGNORECASE)

    current_mcq: Optional[Dict[str, Any]] = None

    for line in lines:
        q_match = q_pattern.match(line)
        opt_match = opt_pattern.match(line)
        ans_match = ans_pattern.match(line)

        if q_match and not opt_match and not ans_match:
            # Check if previous MCQ had options; if so, commit it
            if current_mcq and current_mcq.get("options"):
                mcqs.append(current_mcq)

            q_num = q_match.group(1) or q_match.group(2)
            q_text = q_match.group(3).strip()
            current_mcq = {
                "question_number": q_num,
                "question": q_text,
                "options": {},
                "answer": None,
                "explanation": None
            }
        elif opt_match and current_mcq:
            key = (opt_match.group(1) or opt_match.group(2)).upper()
            val = opt_match.group(3).strip()
            current_mcq["options"][key] = val
        elif ans_match and current_mcq:
            current_mcq["answer"] = (ans_match.group(1) or "").upper()
            if ans_match.group(2):
                current_mcq["explanation"] = ans_match.group(2).strip()
        elif current_mcq:
            # Append continuation lines to current question or last option
            if not current_mcq["options"]:
                current_mcq["question"] += " " + line
            elif current_mcq.get("answer"):
                if current_mcq["explanation"]:
                    current_mcq["explanation"] += " " + line
                else:
                    current_mcq["explanation"] = line

    if current_mcq and current_mcq.get("options"):
        mcqs.append(current_mcq)

    return mcqs


def extract_exercises_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extracts structured practice exercises and sentence drills from text.
    Identifies patterns like:
    - 'Exercise 1', 'Practice Exercise - 5', 'अभ्यास'
    - Followed by numbered items or translation prompts.
    """
    exercises = []
    ex_header_pattern = re.compile(
        r"(?:(?:Practice\s+)?Exercise|अभ्यास|Quiz|Test\s+Yourself)\s*[-–:]?\s*(\d+|[A-Za-z]+)?",
        re.IGNORECASE
    )

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    current_ex: Optional[Dict[str, Any]] = None

    for line in lines:
        header_match = ex_header_pattern.search(line)
        if header_match and len(line) < 60:
            if current_ex and current_ex.get("items"):
                exercises.append(current_ex)
            current_ex = {
                "title": line,
                "exercise_id": header_match.group(1) or "",
                "items": []
            }
        elif current_ex:
            item_match = re.match(r"^(\d+)[\.\)]\s*(.*)", line)
            if item_match:
                current_ex["items"].append({
                    "item_number": item_match.group(1),
                    "text": item_match.group(2).strip()
                })
            elif current_ex["items"] and len(current_ex["items"]) > 0:
                # Continuation line
                current_ex["items"][-1]["text"] += " " + line

    if current_ex and current_ex.get("items"):
        exercises.append(current_ex)

    return exercises


def extract_tables_from_page(page: Any) -> List[Dict[str, Any]]:
    """Extracts tables from a PyMuPDF page object using find_tables()."""
    tables = []
    try:
        tabs = page.find_tables()
        for idx, tab in enumerate(tabs):
            df_rows = tab.extract()
            if not df_rows or len(df_rows) < 2:
                continue

            # First row as header, remaining as rows
            raw_headers = df_rows[0]
            headers = [str(h).strip() if h is not None else f"Col_{i+1}" for i, h in enumerate(raw_headers)]
            rows = []
            for r in df_rows[1:]:
                row_dict = {}
                for i, col in enumerate(r):
                    header_name = headers[i] if i < len(headers) else f"Col_{i+1}"
                    row_dict[header_name] = str(col).strip() if col is not None else ""
                rows.append(row_dict)

            tables.append({
                "table_index": idx + 1,
                "bbox": [float(b) for b in tab.bbox],
                "headers": headers,
                "row_count": len(rows),
                "rows": rows
            })
    except Exception:
        pass
    return tables


def classify_text_blocks(blocks: List[Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Classifies text blocks into Headings and Paragraphs based on layout & styling."""
    headings = []
    paragraphs = []

    for block in blocks:
        # PyMuPDF block format: (x0, y0, x1, y1, text, block_no, block_type)
        if len(block) >= 5 and block[6] == 0:  # Text block
            text = block[4].strip()
            if not text:
                continue

            lines = text.split("\n")
            bbox = [float(block[0]), float(block[1]), float(block[2]), float(block[3])]

            # Heuristic for headings: short length, uppercase or title case, or isolated
            if len(lines) <= 2 and len(text) < 80 and not text.endswith("."):
                headings.append({
                    "text": text,
                    "bbox": bbox
                })
            else:
                paragraphs.append({
                    "text": text,
                    "bbox": bbox
                })

    return headings, paragraphs


def process_page(
    doc: Any,
    page_num: int,
    languages: str,
    dpi: int,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Processes a single PDF page with digital text extraction and OCR fallback/hybrid.
    """
    start_time = time.time()
    page_idx = page_num - 1
    page = doc[page_idx]
    rect = page.rect
    width, height = float(rect.width), float(rect.height)

    # 1. Try digital text extraction first
    raw_text = page.get_text("text").strip()
    blocks = page.get_text("blocks")
    tables = extract_tables_from_page(page)

    extraction_type = "digital"
    confidence_score = 100.0
    ocr_word_count = 0
    low_confidence_words = []

    # Check if page has insufficient digital text (likely scanned or raster image)
    word_count = len(raw_text.split())
    if word_count < 15 and pytesseract is not None:
        logger.info(f"Page {page_num}: Low digital text ({word_count} words). Performing Tesseract OCR ({languages}) at {dpi} DPI...")
        extraction_type = "ocr"

        try:
            pix = page.get_pixmap(dpi=dpi)
            mode = "RGB" if pix.alpha == 0 else "RGBA"
            pil_img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

            if cv2 is not None and np is not None:
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                cv_img, skew_angle = deskew_image_if_needed(cv_img, logger)
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

            # Run pytesseract
            ocr_data = pytesseract.image_to_data(pil_img, output_type=Output.DICT, lang=languages)
            tess_words = []
            confidences = []

            n_boxes = len(ocr_data["text"])
            for i in range(n_boxes):
                w_text = ocr_data["text"][i].strip()
                conf = int(ocr_data["conf"][i])
                if w_text and conf >= 0:
                    tess_words.append(w_text)
                    confidences.append(conf)
                    if conf < 65:
                        low_confidence_words.append({
                            "word": w_text,
                            "confidence": conf
                        })

            ocr_text = " ".join(tess_words).strip()
            if ocr_text:
                raw_text = ocr_text

            if confidences:
                confidence_score = round(float(np.mean(confidences)), 2)
            else:
                confidence_score = 0.0

            ocr_word_count = len(tess_words)
            del pix
            del pil_img
        except Exception as ocr_err:
            logger.error(f"Page {page_num} OCR failed: {ocr_err}")
            confidence_score = 0.0
    elif word_count >= 15:
        logger.info(f"Page {page_num}: Extracted {word_count} words via digital layout.")
    else:
        logger.warning(f"Page {page_num}: Page appears empty or pytesseract is unavailable.")

    # 2. Extract structured content
    headings, paragraphs = classify_text_blocks(blocks)
    mcqs = extract_mcqs_from_text(raw_text)
    exercises = extract_exercises_from_text(raw_text)

    proc_time = round(time.time() - start_time, 2)

    return {
        "page_number": page_num,
        "dimensions": {
            "width": width,
            "height": height
        },
        "extraction_type": extraction_type,
        "confidence_score": confidence_score,
        "word_count": len(raw_text.split()),
        "character_count": len(raw_text),
        "processing_time_seconds": proc_time,
        "headings": headings,
        "paragraphs": paragraphs,
        "tables": tables,
        "exercises": exercises,
        "mcqs": mcqs,
        "low_confidence_words": low_confidence_words[:20],  # Sample first 20 for review
        "raw_text": raw_text
    }


def save_checkpoint(checkpoint_dir: str, batch_id: str, data: Dict[str, Any], logger: logging.Logger) -> str:
    """Saves intermediate batch results for resumability."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_batch_{batch_id}.json")
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Checkpoint saved: {ckpt_path}")
    return ckpt_path


def load_checkpoint(checkpoint_path: str, expected_sha: str, logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Loads checkpoint and verifies input checksum matches."""
    if not os.path.isfile(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("pdf_sha256") == expected_sha:
            return data
        else:
            logger.warning(f"Checksum mismatch for checkpoint {checkpoint_path}. Discarding.")
            return None
    except Exception as e:
        logger.warning(f"Failed to read checkpoint {checkpoint_path}: {e}")
        return None


def run_pipeline(
    pdf_path: str,
    output_dir: str,
    page_range_str: str = "all",
    languages: str = "hin+eng",
    dpi: int = 300,
    batch_size: int = 25,
    checkpoint_dir: str = ".checkpoints",
    resume: bool = True
) -> bool:
    """Main execution pipeline."""
    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "extraction.log")
    logger = setup_logger(log_file)

    logger.info("=" * 70)
    logger.info("STARTING PDF EXTRACTION PIPELINE (GitHub Actions Runner)")
    logger.info("=" * 70)

    if not os.path.isfile(pdf_path):
        logger.error(f"Input PDF not found at: {pdf_path}")
        return False

    if fitz is None:
        logger.error("PyMuPDF (fitz) is not installed. Please install requirements.txt.")
        return False

    # Check input checksum & size
    file_size = os.path.getsize(pdf_path)
    sha256 = calculate_sha256(pdf_path)
    logger.info(f"Input File: {pdf_path}")
    logger.info(f"File Size: {file_size / (1024*1024):.2f} MB ({file_size} bytes)")
    logger.info(f"SHA256: {sha256}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF document: {e}")
        return False

    total_pdf_pages = len(doc)
    logger.info(f"Total Pages in PDF: {total_pdf_pages}")

    # Parse requested pages
    target_pages = parse_page_range(page_range_str, total_pdf_pages)
    logger.info(f"Requested Page Range: '{page_range_str}' -> {len(target_pages)} pages: {target_pages[0]}..{target_pages[-1]}")

    # Process in batches
    batches = [target_pages[i:i + batch_size] for i in range(0, len(target_pages), batch_size)]
    logger.info(f"Divided {len(target_pages)} pages into {len(batches)} batches (batch_size={batch_size})")

    processed_pages: List[Dict[str, Any]] = []

    for b_idx, batch in enumerate(batches):
        batch_id = f"{batch[0]}_{batch[-1]}"
        ckpt_file = os.path.join(checkpoint_dir, f"checkpoint_batch_{batch_id}.json")

        # Check resumability
        if resume and os.path.isfile(ckpt_file):
            cached = load_checkpoint(ckpt_file, sha256, logger)
            if cached and "pages" in cached:
                logger.info(f"Resuming batch {b_idx + 1}/{len(batches)} (pages {batch_id}) from checkpoint.")
                processed_pages.extend(cached["pages"])
                continue

        logger.info(f"--- Processing Batch {b_idx + 1}/{len(batches)}: Pages {batch[0]} to {batch[-1]} ---")
        batch_results = []
        for p_num in batch:
            try:
                page_data = process_page(doc, p_num, languages, dpi, logger)
                batch_results.append(page_data)
            except Exception as page_err:
                logger.error(f"Error processing page {p_num}: {page_err}", exc_info=True)
                batch_results.append({
                    "page_number": p_num,
                    "extraction_type": "error",
                    "error": str(page_err),
                    "confidence_score": 0.0,
                    "word_count": 0,
                    "character_count": 0,
                    "headings": [],
                    "paragraphs": [],
                    "tables": [],
                    "exercises": [],
                    "mcqs": [],
                    "raw_text": ""
                })
            # Immediate memory cleanup after each page
            gc.collect()

        # Save checkpoint for this batch
        checkpoint_payload = {
            "pdf_sha256": sha256,
            "batch_index": b_idx,
            "page_range": [batch[0], batch[-1]],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pages": batch_results
        }
        save_checkpoint(checkpoint_dir, batch_id, checkpoint_payload, logger)
        processed_pages.extend(batch_results)

    doc.close()

    # Sort pages to ensure strict page order
    processed_pages.sort(key=lambda x: x["page_number"])

    total_execution_time = round(time.time() - start_total_time, 2)
    logger.info(f"All batches processed in {total_execution_time:.2f} seconds.")

    # Generate Output Files
    logger.info("Generating output files...")

    # 1. extracted_content.json
    content_data = {
        "document": {
            "filename": os.path.basename(pdf_path),
            "filepath": pdf_path,
            "sha256": sha256,
            "file_size_bytes": file_size,
            "total_pdf_pages": total_pdf_pages
        },
        "extraction_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "page_range_requested": page_range_str,
            "page_count_processed": len(processed_pages),
            "ocr_languages": languages,
            "dpi": dpi,
            "total_execution_time_seconds": total_execution_time
        },
        "pages": processed_pages
    }
    content_json_path = os.path.join(output_dir, "extracted_content.json")
    with open(content_json_path, "w", encoding="utf-8") as f:
        json.dump(content_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Created: {content_json_path}")

    # 2. extracted_text.txt
    text_file_path = os.path.join(output_dir, "extracted_text.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(f"DOCUMENT: {os.path.basename(pdf_path)}\n")
        f.write(f"PROCESSED AT: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"TOTAL PAGES PROCESSED: {len(processed_pages)}\n")
        f.write("=" * 80 + "\n\n")

        for p in processed_pages:
            f.write(f"--- PAGE {p['page_number']} OF {total_pdf_pages} ---\n")
            if p.get("tables"):
                f.write(f"[Detected {len(p['tables'])} Table(s)]\n")
            if p.get("exercises"):
                f.write(f"[Detected {len(p['exercises'])} Exercise Section(s)]\n")
            if p.get("mcqs"):
                f.write(f"[Detected {len(p['mcqs'])} MCQ(s)]\n")
            f.write("\n")
            f.write(p.get("raw_text", ""))
            f.write("\n\n" + "-" * 80 + "\n\n")
    logger.info(f"Created: {text_file_path}")

    # 3. ocr_review_report.json
    review_pages = []
    low_confidence_pages = []
    for p in processed_pages:
        conf = p.get("confidence_score", 0.0)
        ext_type = p.get("extraction_type", "digital")
        is_low_conf = (conf < 75.0 and ext_type == "ocr") or (ext_type == "error")

        if is_low_conf:
            low_confidence_pages.append(p["page_number"])

        review_pages.append({
            "page_number": p["page_number"],
            "extraction_type": ext_type,
            "confidence_score": conf,
            "word_count": p.get("word_count", 0),
            "flagged_for_review": is_low_conf,
            "low_confidence_words_count": len(p.get("low_confidence_words", [])),
            "sample_suspect_words": p.get("low_confidence_words", [])[:10]
        })

    review_report = {
        "report_generated_at": datetime.now(timezone.utc).isoformat(),
        "total_pages_evaluated": len(processed_pages),
        "pages_flagged_for_review_count": len(low_confidence_pages),
        "pages_flagged_for_review": low_confidence_pages,
        "pages_review": review_pages
    }
    review_json_path = os.path.join(output_dir, "ocr_review_report.json")
    with open(review_json_path, "w", encoding="utf-8") as f:
        json.dump(review_report, f, ensure_ascii=False, indent=2)
    logger.info(f"Created: {review_json_path}")

    # 4. extraction_stats.json
    total_words = sum(p.get("word_count", 0) for p in processed_pages)
    total_chars = sum(p.get("character_count", 0) for p in processed_pages)
    total_tables = sum(len(p.get("tables", [])) for p in processed_pages)
    total_mcqs = sum(len(p.get("mcqs", [])) for p in processed_pages)
    total_exercises = sum(len(p.get("exercises", [])) for p in processed_pages)
    digital_count = sum(1 for p in processed_pages if p.get("extraction_type") == "digital")
    ocr_count = sum(1 for p in processed_pages if p.get("extraction_type") == "ocr")
    avg_confidence = (
        round(sum(p.get("confidence_score", 0.0) for p in processed_pages) / len(processed_pages), 2)
        if processed_pages else 0.0
    )
    coverage_pct = round((len(processed_pages) / len(target_pages)) * 100.0, 2) if target_pages else 0.0

    stats = {
        "document_name": os.path.basename(pdf_path),
        "total_pdf_pages": total_pdf_pages,
        "pages_requested_count": len(target_pages),
        "pages_processed_count": len(processed_pages),
        "page_coverage_percent": coverage_pct,
        "digital_pages_count": digital_count,
        "ocr_pages_count": ocr_count,
        "average_confidence_score": avg_confidence,
        "total_words_extracted": total_words,
        "total_characters_extracted": total_chars,
        "total_tables_detected": total_tables,
        "total_mcqs_detected": total_mcqs,
        "total_exercises_detected": total_exercises,
        "total_processing_time_seconds": total_execution_time,
        "pages_per_minute": round((len(processed_pages) / (total_execution_time / 60.0)), 2) if total_execution_time > 0 else 0.0
    }
    stats_json_path = os.path.join(output_dir, "extraction_stats.json")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Created: {stats_json_path}")

    # 5. extraction_manifest.json
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_pdf": {
            "path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "size_bytes": file_size,
            "sha256": sha256,
            "total_pages": total_pdf_pages
        },
        "execution_parameters": {
            "page_range": page_range_str,
            "ocr_languages": languages,
            "dpi": dpi,
            "batch_size": batch_size,
            "pages_processed": [p["page_number"] for p in processed_pages]
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "runner_os": os.getenv("RUNNER_OS", "local"),
            "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
            "github_run_number": os.getenv("GITHUB_RUN_NUMBER", "local"),
            "github_workflow": os.getenv("GITHUB_WORKFLOW", "local")
        },
        "output_files": {
            "extracted_content_json": {
                "filename": "extracted_content.json",
                "size_bytes": os.path.getsize(content_json_path),
                "sha256": calculate_sha256(content_json_path)
            },
            "extracted_text_txt": {
                "filename": "extracted_text.txt",
                "size_bytes": os.path.getsize(text_file_path),
                "sha256": calculate_sha256(text_file_path)
            },
            "ocr_review_report_json": {
                "filename": "ocr_review_report.json",
                "size_bytes": os.path.getsize(review_json_path),
                "sha256": calculate_sha256(review_json_path)
            },
            "extraction_stats_json": {
                "filename": "extraction_stats.json",
                "size_bytes": os.path.getsize(stats_json_path),
                "sha256": calculate_sha256(stats_json_path)
            },
            "extraction_log": {
                "filename": "extraction.log",
                "size_bytes": os.path.getsize(log_file)
            }
        }
    }
    manifest_json_path = os.path.join(output_dir, "extraction_manifest.json")
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Created: {manifest_json_path}")

    logger.info("=" * 70)
    logger.info(f"EXTRACTION COMPLETE: {len(processed_pages)} pages processed successfully.")
    logger.info(f"Artifacts located in: {output_dir}")
    logger.info("=" * 70)

    return True


def auto_detect_pdf(input_dir: str = "input") -> Optional[str]:
    """Auto-detects PDF in input directory or root."""
    if os.path.isdir(input_dir):
        for f in os.listdir(input_dir):
            if f.lower().endswith(".pdf"):
                return os.path.join(input_dir, f)
    for f in os.listdir("."):
        if f.lower().endswith(".pdf"):
            return f
    return None


def main():
    parser = argparse.ArgumentParser(description="PDF Extraction and OCR Processing Engine")
    parser.add_argument("--pdf", "--pdf-path", dest="pdf_path", default=None, help="Path to input PDF file")
    parser.add_argument("--output-dir", default="output", help="Directory for output files (default: output)")
    parser.add_argument("--pages", default="all", help="Page range to process (e.g., 'all', '1-50', '1,5,10-20')")
    parser.add_argument("--lang", default="hin+eng", help="OCR languages (default: hin+eng)")
    parser.add_argument("--dpi", type=int, default=300, help="Image rendering DPI (default: 300)")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for intermediate checkpoints (default: 25)")
    parser.add_argument("--checkpoint-dir", default=".checkpoints", help="Directory for checkpoints (default: .checkpoints)")
    parser.add_argument("--no-resume", action="store_true", help="Disable resuming from checkpoints")

    args = parser.parse_args()

    pdf_path = args.pdf_path
    if not pdf_path:
        pdf_path = auto_detect_pdf("input")
        if not pdf_path:
            print("ERROR: No input PDF specified and none found in input/ or root directory.", file=sys.stderr)
            sys.exit(1)

    success = run_pipeline(
        pdf_path=pdf_path,
        output_dir=args.output_dir,
        page_range_str=args.pages,
        languages=args.lang,
        dpi=args.dpi,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
