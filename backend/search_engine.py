import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(
        self,
        csv_path: Path,
        model_name: str = "sentence-transformers/average_word_embeddings_glove.840B.300d",
        store_dir: Optional[Path] = None,
    ):
        self.csv_path = csv_path
        self.model_name = model_name
        self.backend_dir = Path(__file__).resolve().parent
        self.store_dir = store_dir or self.backend_dir / "index_store"
        self.store_dir.mkdir(exist_ok=True)
        self.db_path = self.store_dir / "index.db"
        self.vectorizer_path = self.store_dir / "vectorizer.joblib"
        self.embeddings_path = self.store_dir / "embeddings.npy"
        self.metadata_path = self.store_dir / "metadata.json"
        logger.info("Initialisation de SearchEngine avec modèle=%s, csv=%s, store=%s", self.model_name, self.csv_path, self.store_dir)
        self.cross_encoder_cache: Dict[str, CrossEncoder] = {}

        self.embedding_model = SentenceTransformer(self.model_name)
        self._prepare_store()
        self._initialize_index()

    def _prepare_store(self):
        logger.debug("Préparation de la base sqlite %s", self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS formations (position INTEGER PRIMARY KEY, title TEXT, site TEXT, university TEXT, row_hash TEXT)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        self.conn.commit()

    def _initialize_index(self):
        self.df, self.row_hashes = self._load_csv(self.csv_path)
        logger.info("CSV chargé (%s lignes) depuis %s", len(self.df), self.csv_path)
        if self._is_index_up_to_date():
            logger.info("Index à jour, chargement des données persistées")
            self._load_persisted_data()
        else:
            logger.info("Index obsolète ou absent, reconstruction de l'index")
            self._build_index()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return str(text).strip()

    def _row_hash(self, title: str, site: str, university: str) -> str:
        normalized = "\t".join(
            [self._normalize_text(title), self._normalize_text(site), self._normalize_text(university)]
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _load_csv(self, csv_path: Path) -> Tuple[pd.DataFrame, List[str]]:
        try:
            df = pd.read_csv(csv_path, sep=";", encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, sep=";", encoding="latin-1", on_bad_lines="skip")
        clean_columns = [c.strip() for c in df.columns.tolist()]
        df.columns = clean_columns
        if "Nom" not in df.columns:
            raise ValueError("Fichier CSV attendu avec une colonne 'Nom'.")
        if "Site" not in df.columns:
            df["Site"] = ""
        if "Université" not in df.columns and "UniversitÃ©" in df.columns:
            df["Université"] = df["UniversitÃ©"]
        if "Université" not in df.columns:
            df["Université"] = ""
        df = df[["Nom", "Site", "Université"]].astype(str).fillna("")
        hashes = [
            self._row_hash(row["Nom"], row["Site"], row["Université"])
            for _, row in df.iterrows()
        ]
        return df, hashes

    def _load_metadata(self) -> Dict[str, str]:
        if not self.metadata_path.exists():
            return {}
        with self.metadata_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _save_metadata(self, metadata: Dict[str, str]) -> None:
        with self.metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp)

    def _compute_csv_signature(self) -> str:
        hasher = hashlib.sha256()
        with self.csv_path.open("rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _is_index_up_to_date(self) -> bool:
        metadata = self._load_metadata()
        stored_signature = metadata.get("csv_signature")
        new_signature = self._compute_csv_signature()
        if stored_signature != new_signature:
            return False
        if not self.vectorizer_path.exists() or not self.embeddings_path.exists():
            return False
        return True

    def _load_persisted_data(self):
        logger.info("Chargement des données persistées depuis %s", self.store_dir)
        self.titles = self.df["Nom"].tolist()
        self.sites = self.df["Site"].tolist()
        self.universities = self.df["Université"].tolist()
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.tfidf_matrix = self.vectorizer.transform(self.titles)
        self.embeddings = np.load(self.embeddings_path)

    def _load_existing_embeddings(self) -> dict[str, list[np.ndarray]]:
        cursor = self.conn.execute(
            "SELECT title, site, university, row_hash, position FROM formations ORDER BY position"
        )
        available = {}
        rows = cursor.fetchall()
        if not rows:
            return available

        existing_embeddings = np.load(self.embeddings_path) if self.embeddings_path.exists() else None
        if existing_embeddings is None or len(existing_embeddings) != len(rows):
            return available

        for row, embedding in zip(rows, existing_embeddings):
            row_hash = row[3]
            available.setdefault(row_hash, []).append(embedding)
        return available

    def _build_index(self):
        self.titles = self.df["Nom"].tolist()
        self.sites = self.df["Site"].tolist()
        self.universities = self.df["Université"].tolist()

        logger.info("Construction d'un nouvel index pour %s lignes", len(self.df))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.titles)

        available_embeddings = self._load_existing_embeddings()
        new_embeddings = []
        reuse_embeddings = []
        rows_to_store = []

        for position, row in self.df.iterrows():
            row_hash = self.row_hashes[position]
            rows_to_store.append((position, row["Nom"], row["Site"], row["Université"], row_hash))
            if available_embeddings.get(row_hash):
                reuse_embeddings.append(available_embeddings[row_hash].pop())
            else:
                new_embeddings.append((position, row["Nom"], row["Site"], row["Université"]))

        if new_embeddings:
            texts = [item[1] for item in new_embeddings]
            computed_embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            reuse_idx = 0
            new_idx = 0
            final_embeddings = []
            for row_hash in self.row_hashes:
                if available_embeddings.get(row_hash) and len(available_embeddings[row_hash]) > 0:
                    final_embeddings.append(available_embeddings[row_hash].pop())
                else:
                    final_embeddings.append(computed_embeddings[new_idx])
                    new_idx += 1
            self.embeddings = np.vstack(final_embeddings)
        else:
            self.embeddings = np.vstack(
                [available_embeddings[row_hash].pop() for row_hash in self.row_hashes]
            )

        logger.info(
            "Index construit : %s lignes (%s embeddings réutilisés, %s embeddings calculés)",
            len(self.df),
            len(reuse_embeddings),
            len(new_embeddings),
        )
        self._persist_index(rows_to_store)

    def _persist_index(self, rows_to_store: List[Tuple[int, str, str, str, str]]):
        logger.info("Sauvegarde de l'index dans %s", self.store_dir)
        self.conn.execute("DELETE FROM formations")
        self.conn.executemany(
            "INSERT INTO formations (position, title, site, university, row_hash) VALUES (?, ?, ?, ?, ?)",
            rows_to_store,
        )
        self.conn.commit()
        joblib.dump(self.vectorizer, self.vectorizer_path)
        np.save(self.embeddings_path, self.embeddings)
        self._save_metadata({"csv_signature": self._compute_csv_signature()})

    def force_reindex(self):
        logger.info("Forçage de la reconstruction complète de l'index")
        self._build_index()

    def _load_cross_encoder(self, model_key: str) -> CrossEncoder:
        model_name = self._cross_encoder_models().get(model_key)
        if not model_name:
            raise ValueError(f"Modèle de reranker inconnu: {model_key}")
        logger.debug("Chargement du CrossEncoder %s pour le reranker %s", model_name, model_key)
        if model_key not in self.cross_encoder_cache:
            self.cross_encoder_cache[model_key] = CrossEncoder(model_name)
        return self.cross_encoder_cache[model_key]

    @staticmethod
    def _cross_encoder_models() -> Dict[str, str]:
        return {
            "none": "",
            "hybrid": "",
            "cross-ms-marco": "cross-encoder/ms-marco-MiniLM-L6-v2",
        }

    def _cross_encoder_rerank(self, candidates: List[tuple[int, float, float]], query: str, model_key: str, top_k: int = 10):
        logger.info("Reranking avec CrossEncoder %s pour %s candidats", model_key, len(candidates))
        cross_encoder = self._load_cross_encoder(model_key)
        query_pairs = [[query, self.titles[idx]] for idx, _, _ in candidates]
        scores = cross_encoder.predict(query_pairs)
        scored = []
        for (idx, lex_score, sem_score), score in zip(candidates, scores):
            scored.append((idx, float(score), lex_score, sem_score, 0.0))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def lexical_search(self, query: str, top_k: int = 20):
        logger.debug("Recherche lexicale pour query=%s top_k=%s", query, top_k)
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in indices]

    def semantic_search(self, query: str, top_k: int = 20):
        logger.debug("Recherche sémantique pour query=%s top_k=%s", query, top_k)
        query_emb = self.embedding_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = cosine_similarity(query_emb, self.embeddings).flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in indices]

    def rerank(self, candidates, query: str, top_k: int = 10):
        logger.debug("Rerank hybride pour query=%s candidats=%s top_k=%s", query, len(candidates), top_k)
        query_terms = set(query.lower().split())
        scored = []
        for idx, lex_score, sem_score in candidates:
            title = self.titles[idx].lower()
            overlap = len(query_terms.intersection(title.split()))
            overlap_score = overlap / max(len(query_terms), 1)
            combined_score = 0.45 * sem_score + 0.45 * lex_score + 0.10 * overlap_score
            scored.append((idx, combined_score, lex_score, sem_score, overlap_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def search(self, query: str, top_k: int = 10, reranker: str = "hybrid"):
        if not query or not query.strip():
            return []
        logger.info("Recherche query=%s top_k=%s reranker=%s", query, top_k, reranker)
        lex_results = self.lexical_search(query, top_k=top_k * 3)
        sem_results = self.semantic_search(query, top_k=top_k * 3)
        candidate_map: Dict[int, Dict[str, float]] = {}
        for idx, score in lex_results:
            candidate_map[idx] = {"lex": score, "sem": 0.0}
        for idx, score in sem_results:
            candidate_map.setdefault(idx, {"lex": 0.0, "sem": 0.0})["sem"] = score
        candidates = [
            (idx, data["lex"], data["sem"])
            for idx, data in candidate_map.items()
        ]
        logger.debug("Candidats assemblés : %s (lex=%s sem=%s)", len(candidates), len(lex_results), len(sem_results))
        if reranker == "none":
            scored = [(idx, (lex_score + sem_score) / 2, lex_score, sem_score, 0.0) for idx, lex_score, sem_score in candidates]
            scored.sort(key=lambda item: item[1], reverse=True)
            reranked = scored[:top_k]
        elif reranker == "hybrid":
            reranked = self.rerank(candidates, query, top_k=top_k)
        else:
            reranked = self._cross_encoder_rerank(candidates, query, model_key=reranker, top_k=top_k)
        logger.info("Search terminé, %s résultats renvoyés pour query=%s", len(reranked), query)
        return [
            {
                "title": self.titles[idx],
                "site": self.sites[idx],
                "university": self.universities[idx],
                "lexical_score": lex_score,
                "semantic_score": sem_score,
                "combined_score": combined_score,
            }
            for idx, combined_score, lex_score, sem_score, _ in reranked
        ]
