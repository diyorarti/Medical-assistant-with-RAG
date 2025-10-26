from rag.pipeline.data_loader import load_data
from rag.pipeline.chunker import chunk_document

# loading data
documents = load_data()

# chunking documents 
chunks = chunk_document(documents)
