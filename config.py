"""
Advanced configuration for PDF processor
Handles different document types and layouts
"""

import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class DocumentProfile:
    """Profile for different document types"""
    name: str
    font_size_multipliers: Dict[str, float]  # H1, H2, H3 multipliers
    score_weights: Dict[str, float]
    title_patterns: List[str]
    heading_patterns: List[str]
    exclusion_patterns: List[str]

class DocumentTypeDetector:
    """Detects document type and applies appropriate processing rules"""
    
    def __init__(self):
        self.profiles = {
            'academic': DocumentProfile(
                name='Academic Paper',
                font_size_multipliers={'H1': 1.4, 'H2': 1.2, 'H3': 1.1},
                score_weights={
                    'font_size': 8.0,
                    'bold': 4.0,
                    'pattern': 3.0,
                    'position': 2.0,
                    'case': 2.0
                },
                title_patterns=[
                    r'^[A-Z][^.!?]*[^.!?]$',  # Title case, no ending punctuation
                    r'^[A-Z\s]+$',  # All caps
                ],
                heading_patterns=[
                    r'^\d+\.?\s+[A-Z]',  # "1. Introduction"
                    r'^[A-Z][a-z]+\s+[A-Z]',  # "Chapter One"
                    r'^(Abstract|Introduction|Methodology|Results|Discussion|Conclusion)',
                    r'^\d+\.\d+\.?\s+',  # "1.1 Background"
                ],
                exclusion_patterns=[
                    r'^(References|Bibliography|Appendix|Figure|Table|Page)\s*\d*$',
                    r'^\d+$',  # Just numbers
                    r'^www\.',  # URLs
                ]
            ),
            'technical': DocumentProfile(
                name='Technical Document',
                font_size_multipliers={'H1': 1.3, 'H2': 1.15, 'H3': 1.05},
                score_weights={
                    'font_size': 6.0,
                    'bold': 5.0,
                    'pattern': 4.0,
                    'position': 3.0,
                    'case': 1.0
                },
                title_patterns=[
                    r'^[A-Z][^.!?]*Manual$',
                    r'^[A-Z][^.!?]*Guide$',
                    r'^[A-Z][^.!?]*Documentation$',
                ],
                heading_patterns=[
                    r'^\d+\.?\s+[A-Z]',  # "1. Setup"
                    r'^[A-Z][a-z]+ing\s',  # "Installing", "Configuring"
                    r'^(Overview|Installation|Configuration|Usage|Troubleshooting)',
                    r'^\d+\.\d+\.?\s+',  # "2.1 Requirements"
                ],
                exclusion_patterns=[
                    r'^(Note|Warning|Caution|Tip):',
                    r'^\$\s',  # Command line
                    r'^[a-z_]+\(',  # Function calls
                ]
            ),
            'legal': DocumentProfile(
                name='Legal Document',
                font_size_multipliers={'H1': 1.2, 'H2': 1.1, 'H3': 1.05},
                score_weights={
                    'font_size': 5.0,
                    'bold': 3.0,
                    'pattern': 6.0,
                    'position': 4.0,
                    'case': 2.0
                },
                title_patterns=[
                    r'^[A-Z\s]+AGREEMENT$',
                    r'^[A-Z\s]+CONTRACT$',
                    r'^[A-Z\s]+POLICY$',
                ],
                heading_patterns=[
                    r'^\d+\.?\s+[A-Z]',  # "1. DEFINITIONS"
                    r'^[A-Z]+\s+[A-Z]',  # "TERMS AND CONDITIONS"
                    r'^(Article|Section|Clause)\s+\d+',
                    r'^\([a-z]\)\s+',  # "(a) Subclause"
                ],
                exclusion_patterns=[
                    r'^Signature',
                    r'^Date:',
                    r'^Witness:',
                ]
            ),
            'report': DocumentProfile(
                name='Business Report',
                font_size_multipliers={'H1': 1.5, 'H2': 1.3, 'H3': 1.15},
                score_weights={
                    'font_size': 7.0,
                    'bold': 4.0,
                    'pattern': 2.0,
                    'position': 3.0,
                    'case': 3.0
                },
                title_patterns=[
                    r'^[A-Z][^.!?]*Report$',
                    r'^[A-Z][^.!?]*Analysis$',
                    r'^[A-Z][^.!?]*Summary$',
                ],
                heading_patterns=[
                    r'^(Executive Summary|Overview|Background|Findings|Recommendations)',
                    r'^\d+\.?\s+[A-Z]',  # "1. Market Analysis"
                    r'^[A-Z][a-z]+\s+[A-Z]',  # "Market Overview"
                ],
                exclusion_patterns=[
                    r'^(Prepared by|Date|Version)',
                    r'^\d+$',  # Page numbers
                ]
            ),
            'generic': DocumentProfile(
                name='Generic Document',
                font_size_multipliers={'H1': 1.3, 'H2': 1.2, 'H3': 1.1},
                score_weights={
                    'font_size': 6.0,
                    'bold': 4.0,
                    'pattern': 3.0,
                    'position': 2.0,
                    'case': 2.0
                },
                title_patterns=[
                    r'^[A-Z][^.!?]*[^.!?]$',
                ],
                heading_patterns=[
                    r'^\d+\.?\s+[A-Z]',
                    r'^[A-Z][a-z]+\s+[A-Z]',
                ],
                exclusion_patterns=[
                    r'^\d+$',
                ]
            )
        }
    
    def detect_document_type(self, blocks: List[Any]) -> str:
        """Detect document type based on content patterns"""
        if not blocks:
            return 'generic'
        
        # Analyze first few pages for patterns
        early_text = ' '.join([b.text for b in blocks if b.page <= 2])
        
        scores = {}
        for doc_type, profile in self.profiles.items():
            if doc_type == 'generic':
                continue
            
            score = 0
            
            # Check title patterns
            for pattern in profile.title_patterns:
                if re.search(pattern, early_text, re.IGNORECASE):
                    score += 3
            
            # Check heading patterns
            for pattern in profile.heading_patterns:
                matches = len(re.findall(pattern, early_text, re.IGNORECASE))
                score += matches * 2
            
            # Check for document type keywords
            keywords = {
                'academic': ['abstract', 'methodology', 'results', 'discussion', 'references'],
                'technical': ['installation', 'configuration', 'usage', 'troubleshooting', 'manual'],
                'legal': ['agreement', 'contract', 'clause', 'terms', 'conditions'],
                'report': ['executive summary', 'findings', 'recommendations', 'analysis']
            }
            
            if doc_type in keywords:
                for keyword in keywords[doc_type]:
                    if keyword.lower() in early_text.lower():
                        score += 1
            
            scores[doc_type] = score
        
        # Return the highest scoring type, or generic if no clear winner
        best_type = max(scores.items(), key=lambda x: x[1]) if scores else ('generic', 0)
        return best_type[0] if best_type[1] > 3 else 'generic'
    
    def get_profile(self, doc_type: str) -> DocumentProfile:
        """Get document profile for given type"""
        return self.profiles.get(doc_type, self.profiles['generic'])

class AdvancedHeadingDetector:
    """Advanced heading detection with document type awareness"""
    
    def __init__(self):
        self.detector = DocumentTypeDetector()
    
    def detect_table_of_contents(self, blocks: List[Any]) -> List[str]:
        """Detect table of contents section"""
        toc_indicators = [
            'table of contents', 'contents', 'index', 'outline'
        ]
        
        toc_headings = []
        in_toc = False
        
        for block in blocks:
            text_lower = block.text.lower()
            
            # Check if we're entering TOC
            if any(indicator in text_lower for indicator in toc_indicators):
                in_toc = True
                continue
            
            # Check if we're leaving TOC
            if in_toc and (block.page > 3 or 'introduction' in text_lower):
                break
            
            # Extract TOC entries
            if in_toc:
                # Look for numbered entries or entries with dots/page numbers
                if re.match(r'^\d+\.?\s+[A-Z]', block.text) or '...' in block.text:
                    # Clean up the text
                    clean_text = re.sub(r'\.{2,}.*$', '', block.text).strip()
                    clean_text = re.sub(r'^\d+\.?\s*', '', clean_text)
                    if clean_text:
                        toc_headings.append(clean_text)
        
        return toc_headings
    
    def cross_validate_with_toc(self, detected_headings: List[Dict], toc_headings: List[str]) -> List[Dict]:
        """Cross-validate detected headings with table of contents"""
        if not toc_headings:
            return detected_headings
        
        validated_headings = []
        
        for heading in detected_headings:
            heading_text = heading['text'].lower()
            
            # Check if this heading appears in TOC
            toc_match = False
            for toc_heading in toc_headings:
                toc_text = toc_heading.lower()
                
                # Exact match
                if heading_text == toc_text:
                    toc_match = True
                    break
                
                # Partial match (at least 70% similarity)
                if len(heading_text) > 5 and len(toc_text) > 5:
                    common_words = set(heading_text.split()) & set(toc_text.split())
                    if len(common_words) / min(len(heading_text.split()), len(toc_text.split())) > 0.7:
                        toc_match = True
                        break
            
            if toc_match:
                heading['toc_validated'] = True
                validated_headings.append(heading)
            else:
                # Keep heading but mark as not validated
                heading['toc_validated'] = False
                validated_headings.append(heading)
        
        return validated_headings
    
    def detect_numbered_sections(self, blocks: List[Any]) -> Dict[str, int]:
        """Detect numbered section patterns"""
        patterns = {
            'decimal': r'^\d+\.?\s+',  # 1. or 1
            'decimal_nested': r'^\d+\.\d+\.?\s+',  # 1.1 or 1.1.
            'roman': r'^[IVX]+\.?\s+',  # I. or I
            'alpha': r'^[A-Z]\.? a0',  # A. or A
            'parenthetical': r'^\(\d+\)\s+',  # (1)
        }
        
        pattern_counts = {}
        
        for pattern_name, pattern in patterns.items():
            count = 0
            for block in blocks:
                if re.match(pattern, block.text):
                    count += 1
            pattern_counts[pattern_name] = count
        
        return pattern_counts

# Configuration constants
FONT_SIZE_TOLERANCE = 0.5
MIN_HEADING_LENGTH = 3
MAX_HEADING_LENGTH = 200
MIN_TITLE_LENGTH = 5
MAX_TITLE_LENGTH = 300

# Common false positive patterns
FALSE_POSITIVE_PATTERNS = [
    r'^\d+$',  # Just numbers
    r'^Page\s+\d+',  # Page numbers
    r'^www\.',  # URLs
    r'^https?://',  # URLs
    r'^[a-z_]+\(',  # Function calls
    r'^[A-Z]{1,3}\s*$',  # Short abbreviations
    r'^[.,:;!?]+$',  # Just punctuation
    r'^\s*$',  # Whitespace only
]

# Quality thresholds
QUALITY_THRESHOLDS = {
    'min_headings': 1,
    'max_headings': 50,
    'min_title_score': 5,
    'min_heading_score': 4,
    'max_duplicate_ratio': 0.3,
} 