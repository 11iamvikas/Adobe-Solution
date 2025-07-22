import os
import json
import logging
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
import re
import numpy as np
from sklearn.cluster import KMeans

# Optional: for schema validation
try:
    from jsonschema import validate, ValidationError
    SCHEMA_PATH = Path("/app/schema/output_schema.json")
    with open(SCHEMA_PATH, "r") as f:
        OUTPUT_SCHEMA = json.load(f)
except Exception:
    OUTPUT_SCHEMA = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class TextBlock:
    def __init__(self, text: str, font_size: float, font_flags: int, font: str, bbox: List[float], page: int, origin: Tuple[float, float], width: float, height: float):
        self.text = text
        self.font_size = font_size
        self.font_flags = font_flags
        self.font_name = font
        self.bbox = bbox
        self.page = page
        self.origin = origin
        self.width = width
        self.height = height
        self.is_bold = (font_flags & 2 != 0)
        self.is_upper = text.isupper()
        self.line_length = len(text.split())
        self.is_title_case = text.istitle()
        self.x_position = origin[0]
        self.y_position = origin[1]

    def __repr__(self):
        return f"TextBlock(text='{self.text}', font_size={self.font_size}, page={self.page})"

class HeadingDetector:
    def __init__(self, profile: Dict[str, Any] = None):
        self.profile = profile if profile else {}
        self.stopwords = set(self.profile.get('stopwords', []))
        self.heading_patterns = self.profile.get('heading_patterns', [])
        self.exclusion_patterns = self.profile.get('exclusion_patterns', [])

    def is_heading_pattern(self, text: str) -> bool:
        # Expanded and refined patterns for more heading styles
        patterns = [
            r'^\d+\.?\s+[A-Z]',                # 1. Introduction
            r'^[A-Z]\.?\s+[A-Z]',               # A. Background
            r'^[IVX]+\.?\s+[A-Z]',              # I. Methods
            r'^(Chapter|Section|Part|Appendix|Introduction|Conclusion|Summary|References|Abstract|Background|Results|Discussion|Acknowledgments|Methods|Materials)\b',
            r'^\d+\.\d+\.?\s+[A-Z]',         # 1.1 Subsection
            r'^[A-Z][a-z]+\s+[A-Z]',             # Section Name Capitalized
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,3}$', # Up to 4 capitalized words
            r'^[A-Z][A-Za-z\s\-:]{3,80}$',      # Title-like, not too short/long
            r'^[A-Z][^.!?]*[^.!?]$',              # Title case, no punctuation
            r'^\d+\s*-\s*[A-Z]',               # 1 - Introduction
            r'^[A-Z]{2,}(\s+[A-Z]{2,})*$',       # ALL CAPS headings
        ]
        exclusion_patterns = [
            r'^\d+$',                            # Page numbers
            r'^Page\s+\d+',                     # Page numbers
            r'^www\.',                           # URLs
            r'^https?://',                        # URLs
            r'^[a-z_]+\(',                       # Function calls
            r'^[A-Z]{1,3}\s*$',                  # Short abbreviations
            r'^[.,:;!?]+$',                       # Just punctuation
            r'^\s*$',                            # Whitespace only
        ]
        # Use profile patterns if in advanced mode
        if ADVANCED_MODE and self.profile:
            patterns = self.profile.heading_patterns + patterns
            exclusion_patterns = self.profile.exclusion_patterns + exclusion_patterns
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        for pattern in exclusion_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        return False

    def cluster_text_blocks(self, blocks: List[TextBlock], n_clusters: int = 3) -> List[int]:
        # Cluster blocks by font size, boldness, and y_position (normalized)
        if not blocks:
            return []
        features = np.array([
            [b.font_size, float(b.is_bold), b.y_position / (b.page + 1)] for b in blocks
        ])
        try:
            kmeans = KMeans(n_clusters=min(n_clusters, len(blocks)), random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
        except Exception:
            labels = np.zeros(len(blocks), dtype=int)
        return labels

    def calculate_heading_score(self, block: TextBlock, doc_stats: Dict[str, Any], cluster_label: int = None, heading_cluster: int = None) -> float:
        score = 0.0
        size_ratio = block.font_size / doc_stats['body_font_size']
        if size_ratio >= 1.5:
            score += 8
        elif size_ratio >= 1.3:
            score += 6
        elif size_ratio >= 1.1:
            score += 4
        elif size_ratio < 0.9:
            score -= 2
        if block.is_bold:
            score += 4
        if block.font_name != doc_stats['body_font_name']:
            score += 2
        if block.is_upper and block.line_length > 2:
            score += 2
        if block.is_title_case:
            score += 3
        if 3 <= block.line_length <= 80:
            score += 2
        elif block.line_length > 120:
            score -= 3
        if self.is_heading_pattern(block.text):
            score += 3
        # Contextual filtering: penalize headings near page edges
        if block.y_position < 50 or block.y_position > 750:
            score -= 2
        # Penalize if too close to previous/next heading (handled in classify_headings)
        if block.x_position <= doc_stats['common_left_margin'] + 20:
            score += 1
        words = block.text.split()
        if words:
            stopword_ratio = sum(1 for w in words if w.lower() in self.stopwords) / len(words)
            if stopword_ratio < 0.3:
                score += 2
        if re.match(r'^\d+$', block.text):
            score -= 10
        if block.text.lower() in ['page', 'www', 'http', 'copyright']:
            score -= 5
        # Cluster bonus: boost score if in heading-like cluster
        if cluster_label is not None and heading_cluster is not None and cluster_label == heading_cluster:
            score += 3
        return score

    def classify_headings(self, blocks: List[TextBlock], doc_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Cluster blocks to find heading-like group
        cluster_labels = self.cluster_text_blocks(blocks, n_clusters=3)
        # Heuristic: heading cluster is the one with largest mean font size
        cluster_font_means = {}
        for label in set(cluster_labels):
            cluster_font_means[label] = np.mean([b.font_size for i, b in enumerate(blocks) if cluster_labels[i] == label])
        heading_cluster = max(cluster_font_means, key=cluster_font_means.get)
        scored_blocks = []
        for i, block in enumerate(blocks):
            score = self.calculate_heading_score(block, doc_stats, cluster_label=cluster_labels[i], heading_cluster=heading_cluster)
            min_score = QUALITY_THRESHOLDS['min_heading_score'] if ADVANCED_MODE else 4
            # Contextual: penalize if too close to previous/next heading
            if i > 0 and abs(block.y_position - blocks[i-1].y_position) < 20 and block.page == blocks[i-1].page:
                score -= 2
            if i < len(blocks)-1 and abs(block.y_position - blocks[i+1].y_position) < 20 and block.page == blocks[i+1].page:
                score -= 2
            if score >= min_score:
                scored_blocks.append({
                    'block': block,
                    'score': score,
                    'font_size': block.font_size
                })
        if not scored_blocks:
            return []
        scored_blocks.sort(key=lambda x: x['score'], reverse=True)
        high_score_blocks = [sb for sb in scored_blocks if sb['score'] >= (QUALITY_THRESHOLDS['min_heading_score'] if ADVANCED_MODE else 6)]
        if not high_score_blocks:
            return []
        font_sizes = [sb['font_size'] for sb in high_score_blocks]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        level_thresholds = {}
        if len(unique_sizes) >= 3:
            level_thresholds['H1'] = unique_sizes[0]
            level_thresholds['H2'] = unique_sizes[1]
            level_thresholds['H3'] = unique_sizes[2]
        elif len(unique_sizes) == 2:
            level_thresholds['H1'] = unique_sizes[0]
            level_thresholds['H2'] = unique_sizes[1]
            level_thresholds['H3'] = unique_sizes[1]
        else:
            level_thresholds['H1'] = unique_sizes[0]
            level_thresholds['H2'] = unique_sizes[0]
            level_thresholds['H3'] = unique_sizes[0]
        outline = []
        processed_texts = set()
        for sb in scored_blocks:
            block = sb['block']
            key = (block.text.lower().strip(), block.page)
            if key in processed_texts:
                continue
            processed_texts.add(key)
            if sb['score'] >= 8 and block.font_size >= level_thresholds['H1'] * 0.95:
                level = 'H1'
            elif sb['score'] >= 6 and block.font_size >= level_thresholds['H2'] * 0.95:
                level = 'H2'
            elif sb['score'] >= 4 and block.font_size >= level_thresholds['H3'] * 0.95:
                level = 'H3'
            else:
                continue
            outline.append({
                'level': level,
                'text': block.text.strip(),
                'page': block.page + 1
            })
        outline.sort(key=lambda x: (x['page'], next(b.y_position for b in blocks if b.text.strip() == x['text'] and b.page + 1 == x['page'])))
        # Always use TOC cross-validation if available
        if hasattr(self, 'heading_detector') and self.heading_detector:
            toc_headings = self.heading_detector.detect_table_of_contents(blocks)
            outline = self.heading_detector.cross_validate_with_toc(outline, toc_headings)
        return outline

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

def merge_multiline_blocks(blocks, y_threshold=18, x_threshold=50, font_size_threshold=1):
    # Merge consecutive blocks that are close in y, x, and font size
    if not blocks:
        return []
    merged = []
    i = 0
    while i < len(blocks):
        curr = blocks[i].copy()
        j = i + 1
        while j < len(blocks):
            nextb = blocks[j]
            if (
                nextb["page"] == curr["page"]
                and abs(nextb["origin"][1] - (curr["origin"][1] + curr["height"])) < y_threshold
                and abs(nextb["origin"][0] - curr["origin"][0]) < x_threshold
                and abs(nextb["font_size"] - curr["font_size"]) < font_size_threshold
            ):
                curr["text"] += " " + nextb["text"]
                curr["height"] += nextb["height"]
                j += 1
            else:
                break
        merged.append(curr)
        i = j
    return merged

def detect_title_robust(blocks: list) -> str:
    # Consider only first 2 pages, merge multiline, and use patterns
    candidate_blocks = [b for b in blocks if b["page"] <= 1 and len(b["text"]) > 5]
    if not candidate_blocks:
        return ""
    merged_blocks = merge_multiline_blocks(candidate_blocks, y_threshold=22, x_threshold=60, font_size_threshold=2)
    max_size = max(b["font_size"] for b in merged_blocks)
    largest_blocks = [b for b in merged_blocks if abs(b["font_size"] - max_size) < 1.5]
    # Prefer blocks near top of page
    top_blocks = [b for b in largest_blocks if b["origin"][1] < 200]
    # Prefer blocks with title-like patterns
    title_pattern = re.compile(r'^[A-Z][A-Za-z0-9\s,:;\-]{5,100}$')
    pattern_blocks = [b for b in top_blocks if title_pattern.match(b["text"])]
    if pattern_blocks:
        title = max(pattern_blocks, key=lambda b: len(b["text"]))["text"]
    elif top_blocks:
        title = max(top_blocks, key=lambda b: len(b["text"]))["text"]
    elif largest_blocks:
        title = max(largest_blocks, key=lambda b: len(b["text"]))["text"]
    else:
        title = max(merged_blocks, key=lambda b: len(b["text"]))["text"]
    logging.info(f"Robust detected title: {title}")
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
    # Debug: print directory contents
    print('Files in /app/input:', list(input_dir.iterdir()))
    pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    if not pdf_files:
        logging.warning("No PDF files found in input directory.")
    for pdf_file in pdf_files:
        logging.info(f"Processing {pdf_file.name}")
        try:
            blocks = extract_text_blocks(pdf_file)
            blocks = merge_multiline_blocks(blocks)
            title = detect_title_robust(blocks)
            # Use advanced heading detector
            text_blocks = [TextBlock(**b) for b in blocks if len(b["text"]) > 1]
            # Compute doc stats for heading scoring
            font_sizes = [b.font_size for b in text_blocks]
            font_names = [b.font_name for b in text_blocks]
            body_font_size = max(set(font_sizes), key=font_sizes.count)
            body_font_name = max(set(font_names), key=font_names.count)
            left_margins = [b.x_position for b in text_blocks]
            doc_stats = {
                "body_font_size": body_font_size,
                "body_font_name": body_font_name,
                "common_left_margin": np.median(left_margins) if left_margins else 0,
            }
            detector = HeadingDetector()
            outline = detector.classify_headings(text_blocks, doc_stats)
            output_file = output_dir / f"{pdf_file.stem}.json"
            generate_output_json(title, outline, output_file)
        except Exception as e:
            logging.error(f"Failed to process {pdf_file.name}: {e}")

if __name__ == "__main__":
    logging.info("Starting PDF processing...")
    process_pdfs()
    logging.info("Completed PDF processing.")