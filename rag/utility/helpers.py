import hashlib
from pathlib import Path

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