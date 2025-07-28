# Challenge 1a: PDF Processing Solution

## Overview
This repository contains a solution for Challenge 1a of the Adobe India Hackathon 2025. The challenge is to build a PDF structure extraction tool that runs offline in a Docker container, processes all PDFs in a folder, and extracts the document title and a hierarchical outline of headings (H1, H2, H3). The output is a JSON file per PDF, conforming to a provided schema.

## Solution Approach

The solution uses **PyMuPDF (fitz)** for PDF parsing and layout analysis. It applies heuristics to identify the document title and headings based on font size, boldness, and position. The tool is modular, efficient, and works fully offline in a CPU-only Docker container.

### Key Features
- **Automatic PDF Processing:** Processes all PDFs in `/sample_dataset/input` and outputs JSONs to `/sample_dataset/output`.
- **Heuristic Heading Detection:** Uses font size and boldness to classify headings as H1, H2, or H3.
- **Title Extraction:** Detects the largest, centered text on the first 1–2 pages as the document title.
- **Schema Validation:** Optionally validates output JSONs against the provided schema using `jsonschema`.
- **Logging:** Logs key steps for traceability and debugging.
- **Containerized:** Runs in a Docker container with no network access and CPU-only.

## File Structure
```
Challenge_1a/
├── sample_dataset/
│   ├── outputs/         # Output JSON files
│   ├── pdfs/            # Input PDF files
│   └── schema/          # Output schema definition
│       └── output_schema.json
├── Dockerfile           # Docker container configuration
├── process_pdfs.py      # Main processing script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## How It Works

### Main Script: `process.py`
- **extract_text_blocks(pdf_path):** Extracts all text blocks from each page, capturing text, font size, font flags (bold/italic), and position.
- **detect_title(blocks):** Finds the largest, centered text block on the first 1–2 pages as the document title.
- **classify_headings(blocks):** Classifies headings into H1, H2, H3 based on font size and boldness heuristics.
- **generate_output_json(title, outline, output_path):** Writes the extracted title and outline to a JSON file matching the schema.
- **Logging:** Each step logs its progress and results for easy debugging.

### Heuristic Model
- **Title:** Largest, centered text on the first 1–2 pages.
- **H1:** Largest bold headings (e.g., ≥22pt)
- **H2:** Slightly smaller (18–21pt)
- **H3:** Smaller but still prominent (14–17pt)
- **Font size thresholds** are determined dynamically per document.

### Libraries Used
- **PyMuPDF (fitz):** PDF parsing, text, and layout extraction.
- **jsonschema:** (Optional) Validates output JSONs against the schema.
- **logging:** For tracing and debugging.
- **scikit-learn:** (Optional, not used in default heuristics) For ML-based heading classification if desired.

## How to Build and Run

### 1. Build the Docker Image
```powershell
docker build --platform linux/amd64 -t pdf-processor .
```

### 2. Run the Docker Container
**On Windows, use absolute Unix-style paths for volume mounts:**
```powershell
docker run --rm ^
  -v "/c/Users/asus/Desktop/Adobe Solution/sample_dataset/pdfs:/app/input:ro" ^
  -v "/c/Users/asus/Desktop/Adobe Solution/sample_dataset/outputs:/app/output" ^
  --network none pdf-processor
```

### 3. Check the Output
- Output JSON files will be in `sample_dataset/outputs`, one per input PDF.

### Command to run Process.py
Go to smart_pdf_processor directory
Run the command - python process.py

## Output Format
Each output JSON matches the schema in `sample_dataset/schema/output_schema.json`:
```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Heading Text", "page": 1 },
    { "level": "H2", "text": "Subheading Text", "page": 2 },
    ...
  ]
}
```

## Performance
- Designed to process a 50-page PDF in ≤10 seconds on a typical CPU.
- Works fully offline, with no network or GPU required.

## Testing
- Test with your own PDFs by placing them in `sample_dataset/pdfs` and running the container as above.
- Check the output JSONs for correct title and heading extraction.

## License
This project uses only open-source libraries and is intended for educational and hackathon use. 

# Smart PDF Processor

A robust, NLP-enhanced PDF structure extractor using PyMuPDF, NLTK, and scikit-learn.

## Directory Structure

```
smart_pdf_processor/
│
├── process.py          # Main script
├── detector.py         # Heading & title detector
├── models.py           # (Optional) ML/NLP helper models
├── utils.py            # Utility functions (e.g., cleaning, merging)
└── /input              # Input PDFs
└── /output             # Output JSON
```

## Usage

1. Place your PDFs in `smart_pdf_processor/input/`.
2. Run `process.py` to extract titles and headings.
3. Outputs are saved as JSON in `smart_pdf_processor/output/`.

## Requirements

- pymupdf
- numpy
- nltk
- scikit-learn

Install with:

```bash
pip install pymupdf numpy nltk scikit-learn
```

For NLTK, run once in Python:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
``` 
