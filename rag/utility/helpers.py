import hashlib
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Any
from langchain_core.documents import Document

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

def extract_text_and_metas(chunks: Iterable[Document]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Extract plain text and metadata from LangChain Document chunks."""
    texts, metas = [], []
    for ch in chunks:
        meta = dict(ch.metadata or {})
        texts.append(ch.page_content or "")
        metas.append(meta)
    return texts, metas

def make_vector_id(meta:dict) -> str:
    """Create a stable, unique vector ID for each chunk based on its metadata."""
    file_hash = meta.get("file_sha256", "nohash")
    page = meta.get("page", "na")
    chunk_id = meta.get("chunk_id", "")
    return f"{file_hash}:{page}:{chunk_id}"


def format_context(results: List[Dict[str, Any]], max_ctx_chars: int) -> str:
    """
    Build a context string from retriever results, supporting either
    {'content', 'similarity_score'} or {'text', 'score'} shapes.
    Adds cite markers [i] per chunk and caps length.
    """
    parts = []
    for i, r in enumerate(results, start=1):
        text = r.get("content")
        if text is None:
            text = r.get("text", "")
        if not text:
            continue
        parts.append(f"[{i}] {text}")
    ctx = "\n\n".join(parts)
    return ctx[:max_ctx_chars]