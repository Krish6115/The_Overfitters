"""
Multilingual Preprocessing & Normalization Engine
================================================
Implements Unicode NFKD decomposition, multilingual case folding,
cross-jurisdiction legal entity suffix stripping, and address canonicalization.
"""

import re
import unicodedata
from typing import Set, Tuple, Optional
import pandas as pd


# Comprehensive international corporate entity designations
_LEGAL_SUFFIX_PATTERN = re.compile(
    r'\b('
    # US / UK / Commonwealth
    r'corporation|corp|incorporated|inc|limited|ltd|llc|l\.l\.c|llp|'
    r'co|company|group|holdings?|international|intl|public\s*ltd|plc|'
    # India
    r'pvt\s*ltd|private\s*limited|pvt|private|enterprises?|'
    r'solutions?|services?|technologies|tech|traders?|stores?|'
    r'industries|associates|consultants?|'
    # France / Francophone
    r'societe|ste|sarl|s\.a\.r\.l|sas|s\.a\.s|sa|s\.a|eurl|sci|snc|gva|'
    # Germany / DACH
    r'gmbh|ag|kg|ohg|ug|haftungsbeschrankt|'
    # Others (Netherlands, Nordics, Iberia, Italy)
    r'pty|nv|bv|ab|oy|as|spa|srl|sl|sae'
    r')\b',
    re.IGNORECASE
)

# Standard address abbreviation mappings for normalising street suffixes
_ADDRESS_ABBREVIATIONS = [
    (re.compile(r'\bst\b', re.IGNORECASE), 'street'),
    (re.compile(r'\brd\b', re.IGNORECASE), 'road'),
    (re.compile(r'\bave\b', re.IGNORECASE), 'avenue'),
    (re.compile(r'\bblvd\b', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\bdr\b', re.IGNORECASE), 'drive'),
    (re.compile(r'\bln\b', re.IGNORECASE), 'lane'),
    (re.compile(r'\bct\b', re.IGNORECASE), 'court'),
    (re.compile(r'\bpl\b', re.IGNORECASE), 'place'),
    (re.compile(r'\bsq\b', re.IGNORECASE), 'square'),
    (re.compile(r'\bopp\b', re.IGNORECASE), 'opposite'),
    (re.compile(r'\bnr\b', re.IGNORECASE), 'near'),
    (re.compile(r'\bdist\b', re.IGNORECASE), 'district'),
    (re.compile(r'\bfl\b', re.IGNORECASE), 'floor'),
    (re.compile(r'\bapt\b', re.IGNORECASE), 'apartment'),
    (re.compile(r'\bste\b', re.IGNORECASE), 'suite'),
    (re.compile(r'\bdept\b', re.IGNORECASE), 'department'),
]


def normalize_unicode(text: Optional[str]) -> str:
    """Standardizes Unicode characters to NFKD representation."""
    if not text or pd.isna(text):
        return ""
    t = str(text)
    t = unicodedata.normalize("NFKD", t)
    return t


def base_clean(text: Optional[str]) -> str:
    """Converts to lowercase, expands ampersands, and strips non-alphanumeric punctuation."""
    t = normalize_unicode(text).lower()
    t = re.sub(r'&', ' and ', t)
    t = re.sub(r'[^\w\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def canonicalize_business_name(text: Optional[str]) -> str:
    """Strips jurisdiction legal suffixes and normalizes whitespace."""
    cleaned = base_clean(text)
    if not cleaned:
        return ""
    stripped = _LEGAL_SUFFIX_PATTERN.sub(' ', cleaned)
    return re.sub(r'\s+', ' ', stripped).strip()


def canonicalize_address(text: Optional[str]) -> str:
    """Expands common abbreviations in addresses and standardizes tokens."""
    cleaned = base_clean(text)
    if not cleaned:
        return ""
    for pattern, replacement in _ADDRESS_ABBREVIATIONS:
        cleaned = pattern.sub(replacement, cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def extract_numbers(text: Optional[str]) -> Set[str]:
    """Extracts all contiguous numeric token sequences (street numbers, postal codes, unit numbers)."""
    if not text or pd.isna(text):
        return set()
    return set(re.findall(r'\b\d+\b', str(text)))


def is_non_latin(text: Optional[str]) -> bool:
    """Detects presence of non-Latin characters (e.g., Cyrillic, CJK, Arabic, Devanagari)."""
    if not text or pd.isna(text):
        return False
    return any(ord(c) > 255 for c in str(text))


def preprocess_entity_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies standard preprocessing to an entity DataFrame.
    Expects columns: ['entity_id', 'business_name', 'business_address', 'country']
    Returns DataFrame augmented with:
      - 'norm_name': base normalized name
      - 'canon_name': legal-suffix stripped name
      - 'norm_addr': base normalized address
      - 'canon_addr': abbreviation expanded address
      - 'is_nonlatin': boolean indicator for script
    """
    df = df.copy()
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("").astype(str).str.strip()

    df["norm_name"] = df["business_name"].apply(base_clean)
    df["canon_name"] = df["business_name"].apply(canonicalize_business_name)
    df["norm_addr"] = df["business_address"].apply(base_clean)
    df["canon_addr"] = df["business_address"].apply(canonicalize_address)
    df["is_nonlatin"] = df["business_name"].apply(is_non_latin).astype("int8")
    return df
