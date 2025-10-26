from rag.pipeline.data_loader import load_data
from rag.pipeline.chunker import chunk_document

from rag.utility.helpers import extract_text_and_metas, make_vector_id

from rag.pipeline.embedder import Embedder



# loading data
documents = load_data()

# chunking documents 
chunks = chunk_document(documents)

# embedding chunks 
# 1. extracting text and metadata
texts, metas = extract_text_and_metas(chunks)
# 2. creating vector id 
ids = [make_vector_id(m) for m in metas]
# 3. duplicate vector ids detector
assert len(ids) == len(set(ids))
# 4. Embedd
embedder = Embedder()
embeddings = embedder.generate_embeddings(texts)