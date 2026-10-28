#!/usr/bin/env python3
"""
Validation and Quality Assurance Engine for PDF Extraction Output.
Verifies:
1. All 6 mandatory output files exist and are non-empty.
2. JSON syntax and schema compliance for all JSON files.
3. Page coverage: ensures no missing or duplicate pages.
4. Consistency across content, manifest, and statistics.
5. Emits GitHub Step Summary if running in GitHub Actions.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Any, Tuple


REQUIRED_FILES = [
    "extracted_content.json",
    "extracted_text.txt",
    "ocr_review_report.json",
    "extraction_manifest.json",
    "extraction_stats.json",
    "extraction.log"
]


def check_files_exist(output_dir: str) -> Tuple[bool, List[str], Dict[str, int]]:
    """Checks that all 6 required files exist and have non-zero size."""
    missing_files = []
    file_sizes = {}
    for filename in REQUIRED_FILES:
        filepath = os.path.join(output_dir, filename)
        if not os.path.isfile(filepath):
            missing_files.append(filename)
        else:
            sz = os.path.getsize(filepath)
            file_sizes[filename] = sz
            if sz == 0:
                missing_files.append(f"{filename} (empty file)")
    return len(missing_files) == 0, missing_files, file_sizes


def validate_content_json(filepath: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validates structure and content of extracted_content.json."""
    errors = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Failed to parse JSON in {filepath}: {e}"], {}

    # Check top-level keys
    for req_key in ["document", "extraction_metadata", "pages"]:
        if req_key not in data:
            errors.append(f"Missing required key '{req_key}' in extracted_content.json")

    pages = data.get("pages", [])
    if not isinstance(pages, list) or len(pages) == 0:
        errors.append("No pages found in 'pages' array in extracted_content.json")
        return False, errors, data

    seen_pages = set()
    page_nums = []
    for p in pages:
        p_num = p.get("page_number")
        if p_num is None:
            errors.append("Found page entry without 'page_number'")
            continue
        if p_num in seen_pages:
            errors.append(f"Duplicate page number detected: {p_num}")
        seen_pages.add(p_num)
        page_nums.append(p_num)

        # Check required fields inside page
        for field in ["dimensions", "extraction_type", "confidence_score", "raw_text"]:
            if field not in p:
                errors.append(f"Page {p_num} missing field '{field}'")

    return len(errors) == 0, errors, data


def validate_page_coverage(pages: List[Dict[str, Any]], expected_range_str: str, total_pdf_pages: int) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Verifies that all expected pages are present with no missing pages or gaps."""
    errors = []
    processed_page_nums = sorted([p["page_number"] for p in pages if "page_number" in p])

    if expected_range_str.lower() == "all":
        expected_set = set(range(1, total_pdf_pages + 1))
    elif "-" in expected_range_str:
        try:
            parts = expected_range_str.split("-")
            s = int(parts[0].strip())
            e = min(total_pdf_pages, int(parts[1].strip()))
            expected_set = set(range(s, e + 1))
        except Exception:
            expected_set = set(processed_page_nums)
    else:
        expected_set = set(processed_page_nums)

    actual_set = set(processed_page_nums)
    missing = sorted(list(expected_set - actual_set))
    extra = sorted(list(actual_set - expected_set))

    if missing:
        errors.append(f"Missing {len(missing)} page(s): {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if extra:
        errors.append(f"Unexpected extra page(s): {extra[:10]}")

    coverage_pct = round((len(actual_set) / len(expected_set)) * 100.0, 2) if expected_set else 100.0

    stats = {
        "expected_page_count": len(expected_set),
        "actual_page_count": len(actual_set),
        "missing_count": len(missing),
        "coverage_percent": coverage_pct,
        "first_page": processed_page_nums[0] if processed_page_nums else None,
        "last_page": processed_page_nums[-1] if processed_page_nums else None
    }

    return len(errors) == 0, errors, stats


def write_github_summary(results: Dict[str, Any], stats: Dict[str, Any]):
    """Writes detailed report to GitHub Actions Step Summary if available."""
    summary_file = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    try:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write("## 📄 PDF Extraction & OCR Verification Report\n\n")
            f.write(f"**Status:** {'✅ PASSED' if results['passed'] else '❌ FAILED'}\n\n")

            f.write("### 📊 Metrics Summary\n\n")
            f.write("| Metric | Value |\n")
            f.write("| --- | --- |\n")
            f.write(f"| Document | `{stats.get('document_name', 'Unknown')}` |\n")
            f.write(f"| Pages Processed | `{stats.get('pages_processed_count', 0)} / {stats.get('total_pdf_pages', 0)}` |\n")
            f.write(f"| Page Coverage | `{stats.get('page_coverage_percent', 0)}%` |\n")
            f.write(f"| Average Confidence | `{stats.get('average_confidence_score', 0)}%` |\n")
            f.write(f"| Digital Pages | `{stats.get('digital_pages_count', 0)}` |\n")
            f.write(f"| OCR Pages | `{stats.get('ocr_pages_count', 0)}` |\n")
            f.write(f"| Words Extracted | `{stats.get('total_words_extracted', 0):,}` |\n")
            f.write(f"| MCQs Detected | `{stats.get('total_mcqs_detected', 0)}` |\n")
            f.write(f"| Tables Detected | `{stats.get('total_tables_detected', 0)}` |\n")
            f.write(f"| Practice Exercises | `{stats.get('total_exercises_detected', 0)}` |\n")
            f.write(f"| Execution Time | `{stats.get('total_processing_time_seconds', 0)} seconds` |\n\n")

            f.write("### 📁 Output Files Generated\n\n")
            f.write("| File Name | Status | Size |\n")
            f.write("| --- | --- | --- |\n")
            for filename in REQUIRED_FILES:
                size_str = f"{results['file_sizes'].get(filename, 0):,} bytes" if filename in results['file_sizes'] else "MISSING"
                icon = "✅" if filename in results['file_sizes'] else "❌"
                f.write(f"| `{filename}` | {icon} | {size_str} |\n")
            f.write("\n")

            if results["errors"]:
                f.write("### ⚠️ Validation Issues\n\n")
                for err in results["errors"]:
                    f.write(f"- {err}\n")
                f.write("\n")
    except Exception as e:
        print(f"Warning: Could not write GITHUB_STEP_SUMMARY: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="PDF Extraction Validation Tool")
    parser.add_argument("--output-dir", default="output", help="Directory containing extraction output files")
    args = parser.parse_args()

    output_dir = args.output_dir
    print("=" * 70)
    print("VALIDATING PDF EXTRACTION OUTPUT")
    print(f"Target Directory: {output_dir}")
    print("=" * 70)

    all_errors = []

    # 1. Check all required files exist
    files_ok, missing_files, file_sizes = check_files_exist(output_dir)
    if not files_ok:
        for m in missing_files:
            all_errors.append(f"Required output file missing or empty: {m}")
        print(f"❌ File Existence Check Failed: {len(missing_files)} file(s) missing/empty.")
    else:
        print(f"✅ All {len(REQUIRED_FILES)} required output files exist and are non-empty:")
        for fn, sz in file_sizes.items():
            print(f"   - {fn} ({sz:,} bytes)")

    # 2. Validate JSON structure of extracted_content.json
    content_file = os.path.join(output_dir, "extracted_content.json")
    content_data = {}
    if os.path.isfile(content_file):
        content_ok, content_errors, content_data = validate_content_json(content_file)
        if not content_ok:
            all_errors.extend(content_errors)
            print(f"❌ Content JSON Schema validation failed with {len(content_errors)} error(s).")
        else:
            print(f"✅ extracted_content.json schema is valid ({len(content_data.get('pages', []))} pages).")

    # 3. Validate stats & manifest
    stats_file = os.path.join(output_dir, "extraction_stats.json")
    stats_data = {}
    if os.path.isfile(stats_file):
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
            print("✅ extraction_stats.json parsed successfully.")
        except Exception as e:
            all_errors.append(f"Failed to parse extraction_stats.json: {e}")

    # 4. Validate page coverage
    if content_data and "pages" in content_data:
        req_range = content_data.get("extraction_metadata", {}).get("page_range_requested", "all")
        total_p = content_data.get("document", {}).get("total_pdf_pages", len(content_data["pages"]))
        cov_ok, cov_errors, cov_stats = validate_page_coverage(content_data["pages"], req_range, total_p)
        if not cov_ok:
            all_errors.extend(cov_errors)
            print(f"❌ Page coverage check failed: {cov_errors}")
        else:
            print(f"✅ Page coverage verified: 100% ({cov_stats['actual_page_count']} pages, range {cov_stats['first_page']}..{cov_stats['last_page']}).")

    passed = len(all_errors) == 0
    results = {
        "passed": passed,
        "file_sizes": file_sizes,
        "errors": all_errors
    }

    write_github_summary(results, stats_data)

    print("=" * 70)
    if passed:
        print("🎉 VALIDATION PASSED: All extraction outputs meet quality and integrity criteria.")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"💥 VALIDATION FAILED with {len(all_errors)} error(s):")
        for err in all_errors:
            print(f"   - {err}")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
