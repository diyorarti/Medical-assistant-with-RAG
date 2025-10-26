from rag.core.config import settings
from rag.pipeline.data_loader import load_data

documents = load_data(settings.DATA_DIR)
print(f"Loaded documents length {len(documents)}")