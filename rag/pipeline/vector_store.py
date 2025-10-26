import os
from typing import List, Any, Optional, Dict
import chromadb
import numpy as np
from pathlib import Path

from rag.utility.helpers import make_vector_id
from rag.core.config import settings

class VectorStore:
    def __init__(
        self,
        collection_name:str=settings.COLLECTION_NAME,
        persist_directory: Path =settings.PERSIST_DIRECTORY_VS,
        embedder_model_name: Optional[str] = settings.EMBEDDER_MODEL_NAME,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedder_model_name = embedder_model_name
        self.client: Optional[chromadb.Client] = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "documents for medical RAG project",
                "hnsw:space": "cosine",                 
                "embedder_model": self.embedder_model_name 
            },
        )
        print(f"Vector Store initialized. Collection: {self.collection_name}")
        try:
            print(f"Existing items in collection: {self.collection.count()}")
        except Exception:
            pass


    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add or update vectors in the collection with deterministic IDs.
        - documents: List[langchain.schema.Document]
        - embeddings: np.ndarray of shape (N, D)
        """
        if documents is None or len(documents) == 0:
            print("No documents to add. Skipping.")
            return
        if embeddings is None or embeddings.shape[0] != len(documents):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Upserting {len(documents)} chunks to vector store...")

        ids: List[str] = []
        metadatas: List[Dict] = []
        documents_text: List[str] = []
        embeddings_list: List[List[float]] = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            meta = dict(getattr(doc, "metadata", {}) or {})

            # Normalize/alias a few helpful fields
            src = meta.get("source_file") or meta.get("source")
            if src:
                meta["source_file"] = str(Path(src).resolve())
                meta["source_name"] = Path(meta["source_file"]).name

            meta["doc_index"] = i
            meta["content_length"] = len(getattr(doc, "page_content", "") or "")

            vec_id = make_vector_id(meta)
            ids.append(vec_id)
            metadatas.append(meta)
            documents_text.append(getattr(doc, "page_content", "") or "")
            embeddings_list.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))

        # Sanity: ensure IDs are unique in this batch
        if len(ids) != len(set(ids)):
            raise ValueError("Deterministic ID collision detected. Check loader/splitter metadata (file_sha256/page/chunk_id).")

        payload = dict(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )

        # Prefer upsert if available
        if hasattr(self.collection, "upsert"):
            self.collection.upsert(**payload)
        else:
            # Fallback for older Chroma: delete then add
            try:
                self.collection.delete(ids=ids)
            except Exception:
                pass
            self.collection.add(**payload)

        print(f"✅ Upserted {len(documents)} chunks.")
        try:
            print(f"Total items in collection: {self.collection.count()}")
        except Exception:
            pass

    def delete_by_source(self, source_path: str):
        """Delete all items from a specific absolute source path."""
        source_abs = str(Path(source_path).resolve())
        try:
            # Chroma filtering API varies by version; adapt as needed:
            self.collection.delete(where={"source_file": source_abs})
            print(f"Deleted items where source_file == {source_abs}")
        except Exception as e:
            print(f"Delete by source failed: {e}")

    def stats(self):
        try:
            print("Collection count:", self.collection.count())
        except Exception as e:
            print("Stats error:", e)
