import os
import json
import logging
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict, Any

# Optional: for schema validation
try:
    from jsonschema import validate, ValidationError
    SCHEMA_PATH = Path("/app/schema/output_schema.json")
    with open(SCHEMA_PATH, "r") as f:
        OUTPUT_SCHEMA = json.load(f)
except Exception:
    OUTPUT_SCHEMA = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def extract_text_blocks(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extracts text blocks with font size, flags, position, and page number from a PDF."""
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue  # skip images, etc.
            for line in block["lines"]:
                for span in line["spans"]:
                    blocks.append({
                        "text": span["text"].strip(),
                        "font_size": span["size"],
                        "font_flags": span["flags"],
                        "font": span["font"],
                        "bbox": span["bbox"],
                        "page": page_num,
                        "origin": (block["bbox"][0], block["bbox"][1]),
                        "width": block["bbox"][2] - block["bbox"][0],
                        "height": block["bbox"][3] - block["bbox"][1],
                    })
    logging.info(f"Extracted {len(blocks)} text blocks from {pdf_path.name}")
    return blocks

def detect_title(blocks: List[Dict[str, Any]]) -> str:
    """Detects the document title as the largest, centered text on the first 2 pages."""
    # Consider only first 2 pages
    candidate_blocks = [b for b in blocks if b["page"] <= 1 and len(b["text"]) > 5]
    if not candidate_blocks:
        return ""
    max_size = max(b["font_size"] for b in candidate_blocks)
    largest_blocks = [b for b in candidate_blocks if abs(b["font_size"] - max_size) < 0.5]
    # Prefer centered text (x position near page center)
    # Assume A4 width ~595, so center is ~297
    centered_blocks = [b for b in largest_blocks if 200 < (b["origin"][0] + b["width"]/2) < 400]
    if centered_blocks:
        title = max(centered_blocks, key=lambda b: len(b["text"]))["text"]
    else:
        title = max(largest_blocks, key=lambda b: len(b["text"]))["text"]
    logging.info(f"Detected title: {title}")
    return title.strip()

def classify_headings(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Classifies headings as H1, H2, H3 based on font size and boldness."""
    # Heuristic: get unique font sizes, sort descending
    font_sizes = sorted({b["font_size"] for b in blocks if len(b["text"]) > 0}, reverse=True)
    if not font_sizes:
        return []
    # Assign thresholds
    h1_size = font_sizes[0]
    h2_size = font_sizes[1] if len(font_sizes) > 1 else h1_size - 2
    h3_size = font_sizes[2] if len(font_sizes) > 2 else h2_size - 2
    outline = []
    for b in blocks:
        text = b["text"].strip()
        if not text or len(text) < 3:
            continue
        size = b["font_size"]
        is_bold = b["font_flags"] & 2 != 0  # PyMuPDF: 2 = bold
        # H1: largest, bold
        if abs(size - h1_size) < 0.5 and is_bold:
            level = "H1"
        # H2: next, bold or large
        elif abs(size - h2_size) < 0.5 and (is_bold or size > h3_size):
            level = "H2"
        # H3: next, bold or large
        elif abs(size - h3_size) < 0.5 and (is_bold or size > 10):
            level = "H3"
        else:
            continue
        outline.append({
            "level": level,
            "text": text,
            "page": b["page"] + 1  # 1-based page
        })
    logging.info(f"Detected {len(outline)} headings (H1/H2/H3)")
    return outline

def generate_output_json(title: str, outline: List[Dict[str, Any]], output_path: Path):
    data = {"title": title, "outline": outline}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Wrote output to {output_path}")
    # Optional: validate
    if OUTPUT_SCHEMA:
        try:
            validate(instance=data, schema=OUTPUT_SCHEMA)
            logging.info(f"Output validated against schema.")
        except ValidationError as e:
            logging.warning(f"Output failed schema validation: {e}")

def process_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logging.warning("No PDF files found in input directory.")
    for pdf_file in pdf_files:
        logging.info(f"Processing {pdf_file.name}")
        try:
            blocks = extract_text_blocks(pdf_file)
            title = detect_title(blocks)
            outline = classify_headings(blocks)
            output_file = output_dir / f"{pdf_file.stem}.json"
            generate_output_json(title, outline, output_file)
        except Exception as e:
            logging.error(f"Failed to process {pdf_file.name}: {e}")

if __name__ == "__main__":
    logging.info("Starting PDF processing...")
    process_pdfs()
    logging.info("Completed PDF processing.")