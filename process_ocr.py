import os
import shutil
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image
from pdf2image import convert_from_path
from spellchecker import SpellChecker
import fitz  # PyMuPDF
from pdf2docx import Converter

# Folders setup
INPUT_DIR = "input"
OUT_WORD_DIR = "output_word"
OUT_PDF_DIR = "output_pdf"
ARCHIVE_DIR = "processed_archive"

for folder in [INPUT_DIR, OUT_WORD_DIR, OUT_PDF_DIR, ARCHIVE_DIR]:
    os.makedirs(folder, exist_ok=True)

spell = SpellChecker()

def deskew_and_clean_image(pil_img):
    """Tirchi photo ko seedha (deskew) karta hai aur mobile camera ki shadows hatata hai."""
    img = cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Tilt Angle pata lagayein
    inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:
                angles.append(angle)

    skew_angle = np.median(angles) if len(angles) > 0 else 0.0

    # 2. Image ko Rotate karke seedha karein
    if abs(skew_angle) > 0.2:
        print(f"   -> Straightening image (Tilt: {skew_angle:.2f} deg)...")
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

    # 3. Camera shadow hatayein aur background white karein
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    clean_img = cv2.merge(result_norm_planes)
    return Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))

def fix_word_spelling(word):
    """Punctuation, numbers aur Capital letters ko bachate hue spelling thik karta hai."""
    prefix, suffix = "", ""
    while word and not word[0].isalnum():
        prefix += word[0]
        word = word[1:]
    while word and not word[-1].isalnum():
        suffix = word[-1] + suffix
        word = word[:-1]

    if word and word.isalpha() and len(word) > 2:
        if word.isupper():
            return prefix + word + suffix
        corrected = spell.correction(word.lower())
        if corrected and corrected != word.lower():
            if word[0].isupper():
                corrected = corrected.capitalize()
            word = corrected

    return prefix + word + suffix

def detect_and_draw_tables(pil_img, pdf_page):
    """Image se tables aur borders detect karke PDF me vector lines draw karta hai."""
    img_np = np.array(pil_img.convert('L'))
    thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts_h, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_h:
        x, y, w, h = cv2.boundingRect(c)
        if w > 30:
            pdf_page.draw_line(fitz.Point(x, y + h / 2), fitz.Point(x + w, y + h / 2), color=(0, 0, 0), width=1)

    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts_v, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_v:
        x, y, w, h = cv2.boundingRect(c)
        if h > 30:
            pdf_page.draw_line(fitz.Point(x + w / 2, y), fitz.Point(x + w / 2, y + h), color=(0, 0, 0), width=1)

def create_table_and_layout_pdf(images, output_pdf_path):
    """Deskew, Table Borders aur OCR Text ke sath exact layout wala PDF banata hai."""
    doc = fitz.open()

    for raw_img in images:
        img = deskew_and_clean_image(raw_img)
        width, height = img.size
        page = doc.new_page(width=width, height=height)

        # 1. Table borders draw karein
        detect_and_draw_tables(img, page)

        # 2. OCR se text aur coordinates nikalein
        ocr_data = pytesseract.image_to_data(img, output_type=Output.DICT, lang='eng')
        n_boxes = len(ocr_data['text'])

        lines = {}
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])

            if text and conf > 30:
                key = (ocr_data['block_num'][i], ocr_data['par_num'][i], ocr_data['line_num'][i])
                if key not in lines:
                    lines[key] = []
                lines[key].append({
                    'text': fix_word_spelling(text),
                    'left': ocr_data['left'][i],
                    'top': ocr_data['top'][i],
                    'width': ocr_data['width'][i],
                    'height': ocr_data['height'][i]
                })

        # 3. Text ko exact position aur font-size ke sath insert karein
        for key, words in lines.items():
            x0 = words[0]['left']
            y0 = min(w['top'] for w in words)
            y1 = max(w['top'] + w['height'] for w in words)
            line_height = y1 - y0

            line_text = " ".join(w['text'] for w in words)
            font_size = max(8, int(line_height * 0.72))

            insert_point = fitz.Point(x0, y0 + line_height * 0.82)
            page.insert_text(insert_point, line_text, fontsize=font_size, fontname="helv", color=(0, 0, 0))

    doc.save(output_pdf_path)
    doc.close()

def convert_pdf_to_word(pdf_path, word_path):
    """PDF ko editable Word (.docx) me convert karta hai."""
    cv = Converter(pdf_path)
    cv.convert(word_path, start=0, end=None)
    cv.close()

def process_files():
    files_processed = 0
    for filename in os.listdir(INPUT_DIR):
        # Placeholder files ko ignore karein
        if filename.startswith('.') or filename.endswith('.txt'):
            continue

        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        images = []
        if ext in ['.png', '.jpg', '.jpeg']:
            images = [Image.open(filepath)]
        elif ext == '.pdf':
            images = convert_from_path(filepath, dpi=300)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        print(f"Processing: {filename}...")
        pdf_out = os.path.join(OUT_PDF_DIR, f"{name}_converted.pdf")
        word_out = os.path.join(OUT_WORD_DIR, f"{name}_converted.docx")

        create_table_and_layout_pdf(images, pdf_out)
        convert_pdf_to_word(pdf_out, word_out)

        # Process hone ke baad input file ko archive folder me move kar dein
        shutil.move(filepath, os.path.join(ARCHIVE_DIR, filename))
        files_processed += 1
        print(f"Successfully generated: {pdf_out} & {word_out}")

    if files_processed == 0:
        print("Input folder me koi nayi image ya PDF file nahi mili.")

if __name__ == "__main__":
    process_files()
