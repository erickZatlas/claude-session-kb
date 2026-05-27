"""
embeddings.py — semantic index for the knowledge base.

Embeds records with all-MiniLM-L6-v2 (transformers + torch, already in the venv) and
keeps an in-memory float32 matrix for cosine search. The matrix is cached to
.cache/vectors.npy + ids.json and updated incrementally: on sync(), only records whose
id is not already indexed get embedded, so live freshness stays cheap.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np

from store import embed_text

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
VEC_PATH = os.path.join(CACHE_DIR, "vectors.npy")
IDS_PATH = os.path.join(CACHE_DIR, "ids.json")


class Embedder:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._tok = None
        self.ids: list[str] = []
        self.index: dict[str, int] = {}
        self.matrix: np.ndarray = np.zeros((0, 384), dtype=np.float32)
        self.dim = 384
        self._load_cache()

    # ---- model (lazy) ----
    def _ensure_model(self):
        if self._model is None:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(MODEL)
            self._model = AutoModel.from_pretrained(MODEL).eval()

    def _embed(self, texts: list[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        self._ensure_model()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch = texts[i:i + 64]
                enc = self._tok(batch, padding=True, truncation=True, max_length=256,
                                return_tensors="pt")
                hidden = self._model(**enc)[0]
                mask = enc["attention_mask"].unsqueeze(-1).float()
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(F.normalize(mean, p=2, dim=1).cpu().numpy())
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    # ---- cache ----
    def _load_cache(self):
        if os.path.exists(VEC_PATH) and os.path.exists(IDS_PATH):
            try:
                self.matrix = np.load(VEC_PATH)
                self.ids = json.load(open(IDS_PATH))
                self.dim = self.matrix.shape[1] if self.matrix.size else 384
                self.index = {rid: i for i, rid in enumerate(self.ids)}
            except Exception:
                self.matrix, self.ids, self.index = np.zeros((0, 384), np.float32), [], {}

    def _save_cache(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(VEC_PATH, self.matrix)
        json.dump(self.ids, open(IDS_PATH, "w"))

    # ---- indexing ----
    def sync(self, records: list[dict]) -> int:
        """Embed any records not yet indexed. Returns count newly embedded."""
        with self._lock:
            missing = [r for r in records if r["id"] not in self.index]
            if not missing:
                return 0
            vecs = self._embed([embed_text(r) for r in missing])
            start = len(self.ids)
            self.ids.extend(r["id"] for r in missing)
            for j, r in enumerate(missing):
                self.index[r["id"]] = start + j
            self.matrix = np.vstack([self.matrix, vecs]) if self.matrix.size else vecs
            self.dim = self.matrix.shape[1]
            self._save_cache()
            return len(missing)

    def search(self, query: str, candidate_records: list[dict], limit: int = 250) -> list[dict]:
        if not candidate_records or self.matrix.size == 0:
            return []
        q = self.embed_query(query)
        rows, recs = [], []
        for r in candidate_records:
            idx = self.index.get(r["id"])
            if idx is not None:
                rows.append(idx)
                recs.append(r)
        if not rows:
            return []
        sims = self.matrix[rows] @ q                      # cosine (all normalized)
        order = np.argsort(-sims)[:limit]
        return [recs[i] for i in order]
