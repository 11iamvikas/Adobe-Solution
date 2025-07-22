import fitz
from pathlib import Path
import json
from detector import score_heading, is_title_like
import numpy as np

def extract_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num, page in enumerate(doc):
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if block['type'] != 0: continue
            for line in block['lines']:
                for span in line['spans']:
                    blocks.append({
                        "text": span['text'].strip(),
                        "font_size": span['size'],
                        "font_flags": span['flags'],
                        "font": span['font'],
                        "bbox": span['bbox'],
                        "page": page_num,
                        "origin": (block['bbox'][0], block['bbox'][1])
                    })
    return blocks

def detect_title(blocks):
    top_blocks = [b for b in blocks if b['page'] <= 1 and b['origin'][1] < 200 and len(b['text']) > 5]
    top_blocks.sort(key=lambda b: -b['font_size'])
    for block in top_blocks:
        if is_title_like(block['text']):
            return block['text']
    return top_blocks[0]['text'] if top_blocks else ''

def classify_headings(blocks):
    font_sizes = [b['font_size'] for b in blocks]
    body_font = max(set(font_sizes), key=font_sizes.count)
    left_margin = np.median([b['origin'][0] for b in blocks])

    headings = []
    for b in blocks:
        score = score_heading(b, body_font, left_margin)
        if score >= 6:
            level = 'H1' if b['font_size'] >= body_font * 1.5 else 'H2' if b['font_size'] >= body_font * 1.2 else 'H3'
            headings.append({
                "text": b['text'],
                "page": b['page'] + 1,
                "level": level
            })
    return headings

def main(pdf_path: str, output_path: str):
    blocks = extract_blocks(pdf_path)
    title = detect_title(blocks)
    outline = classify_headings(blocks)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"title": title, "outline": outline}, f, indent=2)
    print(f"[✅] Processed: {pdf_path}")

if __name__ == "__main__":
    input_dir = Path("../sample_dataset/pdfs")
    output_dir = Path("../sample_dataset/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for file in input_dir.glob("*.pdf"):
        out_path = output_dir / f"{file.stem}.json"
        main(str(file), str(out_path)) 