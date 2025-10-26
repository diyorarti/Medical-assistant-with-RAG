import hashlib
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.utility.helpers import get_chunk_id, normalize_text
from rag.core.config import settings

def chunk_document(
        documents:List[Document],
        chunk_size:int=settings.CHUNK_SIZE,
        chunk_overlap:int=settings.CHUNK_OVERLAP,
        min_chunk_chars:int=settings.MIN_CHUNK_CHARS,
        *,
        encoding_name:str=settings.TIKTOKEN_ENCODING,
        id_prefix_len:int=settings.CHUNK_ID_PREFIX_LEN
    ) -> List[Document]:
    """
    split page-level documents into chunk-level documet with stable metadata.
    expects each input Document to represent one page (PuPDFLoader behavior)
    """
    separators=settings.CHUNK_SEPARATORS
    
    chunk_id, tok_len, using_tok = get_chunk_id(
        encoding_name=encoding_name,
        id_prefix_len=id_prefix_len
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tok_len,
        separators=separators,
        keep_separator=False,
    )

    split_docs:List[Document] = []
    seen_within_page = {}

    for d in documents:
        meta = dict(d.metadata or {})
        src_abs = str(Path(meta.get("source_file") or meta.get("source") or "").resolve())
        page = int(meta.get("page")) if meta.get("page") is not None else -1 

        parts = text_splitter.split_text(d.page_content or "")
        running = 0

        seen = seen_within_page.setdefault((src_abs, page), set())

        for idx, raw in enumerate(parts):
            raw_stripped = (raw or "").strip()
            if len(raw_stripped) < min_chunk_chars:
                continue
            norm = normalize_text(raw_stripped)
            if not norm or len(norm) < min_chunk_chars:
                continue
            sig = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            if sig in seen:
                continue
            seen.add(sig)

            child_meta = {
                **meta,
                "source": src_abs,
                "source_file": src_abs,
                "source_name": Path(src_abs).name,
                "page": page,
                "chunk_index": idx,
                "char_start_hint": running,
                "char_end_hint": running + len(raw_stripped),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_id": chunk_id(src_abs, page, idx, norm),
            }

            running += len(raw_stripped)
            split_docs.append(Document(page_content=norm, metadata=child_meta))
    
    print(
        f"Split {len(documents)} page-docs into {len(split_docs)} chunks "
        f"(tokenizer={'tiktoken' if using_tok else 'chars'})"
    )
    return split_docs