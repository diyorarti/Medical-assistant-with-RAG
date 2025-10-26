import hashlib
from pathlib import Path

import re

def sha256_file(path:Path, chunk_size:int=8192)->str:
    """
    Comupute file Fingerprint
    helps to track whether a file's content has changed or avoid reprocessing duplicates
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break 
            h.update(b)
    return h.hexdigest()

def normalize_text(text: str) -> str:
    """
    clean and normalize raw text extracted from PDFs
    Raw PDF text usually contains:
         broken words due to line breaks
         extra new lines
         inconsistent spacing
    """
    if not text:
        return ""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def get_chunk_id(encoding_name:str="cl100k_base", id_prefix_len:int=16):
    """
    Build two utilities in one go:
        - chunk_id (source_abs, page, chunk_idx, norm_text) -> str
        - tok_len(s) -> int 
    
    Args:
        ebcoding_name: tiktoken endcoding to use (e.g., "cl100k_base").
        id_prefix_len: number of hex chars to keep from the sha256 digest.
    
    return:
        (chunk_id, tok_len, using_tokenizer)
        chunk_id:callable
        tok_le:callable
        using_tokenizer:bool
    """
    enc = None
    try:
        import tiktoken 
        enc = tiktoken.get_encoding(encoding_name)
        def tok_len(s:str) -> int:
            return len(enc.encode(s or ""))
        using_tokenizer = True
    except Exception:
        def tok_len(s:str) -> int:
            return len(s or "")
        using_tokenizer = False

    def chunk_id(source_abs:str, page:int, chunk_idx:int, norm_text:str) -> str:
        base = f"{source_abs}|{page}|{chunk_idx}|{norm_text}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:id_prefix_len]
    return chunk_id, tok_len, using_tokenizer