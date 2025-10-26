from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

from rag.core.config import settings

class Embedder:
    """
    A utility class for generating text embeddings using SentenceTransformers.
    Attributes:
        model_name (str): Name of the SentenceTransformer model to load.
        normalize (bool): Whether to normalize embeddings for cosine similarity.
        batch_size (int): Number of texts to embed in each batch.
        model (Optional[SentenceTransformer]): The loaded embedding model instance.

    Methods:
        generate_embeddings(texts: List[str]) -> np.ndarray:
            Generate embeddings for a list of texts (e.g., document chunks).
        
        generate_embedding(query: str) -> np.ndarray:
            Generate an embedding for a single text (e.g., user query).
    """

    def __init__(self, 
                 model_name:str=settings.EMBEDDER_MODEL_NAME,
                 normalize:bool=settings.NORMALIZE,
                 batch_size:int=settings.BATCH_SIZE,
                 device:Optional[str]=None,
                ):
        self.model_name=model_name
        self.normalize=normalize
        self.batch_size=batch_size
        self.model:Optional[SentenceTransformer]=None
        self._initialize_model()
    
    def _initialize_model(self):
        print(f"Loading Embedding Model: {self.model_name}")
        self.model=SentenceTransformer(self.model_name)
        dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. EMbedding dimension: {dim}")
    
    def generate_embeddings(self, texts:List[str]) -> np.ndarray:
        """Embedding funtion for external knowledge"""
        if self.model is None:
            raise ValueError("embeddding model is not initialized")
        if texts is None or len(texts) == 0:
            dim = self.model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)
        print(f"Generating embedding for {len(texts)} texts")
        embs = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        ).astype(np.float32, copy=False)
        print(f"Generated emebddings with shape {embs.shape}")
        return embs
    
    def generate_embedding(self, query:str) -> np.ndarray:
        """embedding functionf for single text (queyr)"""
        if self.model is None:
            raise ValueError("Embedding model is not initialized")
        if query is None or query.strip() == "":
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((dim, ), dtype=np.float32)
        emb = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        ).astype(np.float32, copy=False)
        print(f"Generated 1 embedding with dim {emb.shape}")
        return emb
        
