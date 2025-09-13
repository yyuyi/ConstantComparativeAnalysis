from __future__ import annotations

import os
from typing import Dict, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions


class VectorIndex:
    def __init__(self, *, api_key: str | None, embed_model: str = "text-embedding-3-large") -> None:
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="gt_segments",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name=embed_model,
            ),
        )

    def split_transcript(self, text: str, *, chunk_chars: int, overlap: int = 80) -> List[str]:
        # Recursive splitter with sentence-friendly separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(200, chunk_chars),
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
        )
        return splitter.split_text(text)

    def add_transcript(self, *, name: str, raw_text: str, segment_len_tokens: int) -> List[Dict]:
        # Approximate characters per token
        chunk_chars = max(400, int(segment_len_tokens * 4))
        chunks = self.split_transcript(raw_text, chunk_chars=chunk_chars)
        segments: List[Dict] = []
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []
        for i, chunk in enumerate(chunks, start=1):
            seg = {"segment_number": i, "text": chunk, "transcript": name}
            segments.append(seg)
            ids.append(f"{name}-{i}")
            documents.append(chunk)
            metadatas.append({"transcript": name, "segment_number": i})
        if documents:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return segments

    def query(self, *, text: str, k: int = 3) -> List[Tuple[str, Dict]]:
        if not text.strip():
            return []
        res = self.collection.query(query_texts=[text], n_results=max(1, k))
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out: List[Tuple[str, Dict]] = []
        for d, m in zip(docs, metas):
            out.append((d, m))
        return out
