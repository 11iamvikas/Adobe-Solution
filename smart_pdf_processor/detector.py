import re
import numpy as np
from nltk.corpus import stopwords
from typing import List, Dict

STOPWORDS = set(stopwords.words('english'))


def is_title_like(text: str) -> bool:
    """Title-like if capitalized, long, no end punctuation"""
    if not text:
        return False
    text = text.strip()
    if len(text) < 5 or len(text.split()) < 2:
        return False
    if text.endswith(('.', '!', '?')):
        return False
    if text.isupper() or text.istitle():
        return True
    return False


def is_heading_pattern(text: str) -> bool:
    patterns = [
        r'^\d+\.?\s+[A-Z]',               # 1. Heading
        r'^[IVX]+\.\s+[A-Z]',             # I. Heading
        r'^[A-Z][a-z]+\s+[A-Z]',           # Capitalized Heading
        r'^[A-Z\s]{3,}$',                 # ALL CAPS
        r'^[A-Z][A-Za-z\s\-]{5,50}$',     # Title-style
    ]
    return any(re.match(p, text.strip()) for p in patterns)


def score_heading(block, body_font_size, left_margin):
    """Score heading candidates"""
    score = 0
    text = block['text'].strip()
    size = block['font_size']

    if size > body_font_size * 1.5:
        score += 5
    elif size > body_font_size * 1.2:
        score += 3
    if block['font_flags'] & 2:
        score += 2  # bold
    if is_heading_pattern(text):
        score += 2
    if is_title_like(text):
        score += 2
    if len(text.split()) < 2:
        score -= 1
    if block['origin'][0] <= left_margin + 15:
        score += 1
    words = text.lower().split()
    if len(words) > 0:
        stop_ratio = sum(1 for w in words if w in STOPWORDS) / len(words)
        if stop_ratio < 0.5:
            score += 1
    # If no words, do not adjust score for stop_ratio
    return score 