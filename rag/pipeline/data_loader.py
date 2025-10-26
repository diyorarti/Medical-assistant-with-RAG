from __future__ import annotations
from pathlib import Path
from typing import List, Union, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from rag.utility.helpers import sha256_file
from rag.core.config import settings

def load_data(
        data_dir: Union[str, Path, None] = None,
        min_chars: Optional[int] = None,
    ) -> List[Document]:

    """
    Recursively load PDFs under `data_dir` and return page-level documents
    enriched with stable file metadata (absolute path, source path, mtime, sha256)

    Args:
        data_dir:Root directory to scan for PDFs
        min_chars:Minimum non-whitespace charaters to keep a page
    """

    data_dir = Path(data_dir) if data_dir is not None else settings.DATA_DIR
    min_chars = int(min_chars) if min_chars is not None else settings.MIN_CHARS

    data_dir = Path(data_dir)
    files = list(data_dir.glob("**/*.pdf"))
    print(f"Number of files {len(files)}")
    all_documents: List[Document] = []
    skipped_pages = 0

    for file in files:
        abs_path = file.resolve()
        print(f"Processing file: {abs_path}")

        try:
            loader = PyPDFLoader(str(abs_path))
            documents = loader.load()
        except Exception as e:
            print(f"!! Skipping {abs_path} due to error: {e}")
            continue

        file_hash = sha256_file(abs_path)
        try:
            mtime = abs_path.stat().st_mtime
        except Exception:
            mtime = None

        for doc in documents:
            content = (doc.page_content or "").strip()
            if len(content) < min_chars:
                skipped_pages += 1
                continue
            meta = dict(doc.metadata or {})
            page = meta.get("page")
            meta.update({
                "source":str(abs_path),
                "source_file":str(abs_path),
                "source_name":abs_path.name,
                "file_type":"pdf",
                "file_sha256":file_hash,
                "file_mtime":mtime,
                "page":page if page is not None else None,
            })
            doc.metadata=meta
        all_documents.extend(documents)
    print(f"skipped (short/blank) so far: {skipped_pages}")
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents