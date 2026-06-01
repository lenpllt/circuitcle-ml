import re
import json
import pathlib
import numpy as np
import streamlit as st
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine

BASE_DIR = pathlib.Path(__file__).parent
_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_MIN_CHUNK_CHARS = 80


def _format_log(filename: str, content: str, tableau_type: str) -> str:
    return f"Historique {tableau_type} – {filename}\n{content.strip()}"


def _format_etapes(data: dict, label: str) -> list:
    chunks = []
    for liste_name, etapes in data.get("donnees_etapes", {}).items():
        lines = [f"Étapes {label} – {liste_name}"]
        for etape in etapes:
            if isinstance(etape, list) and len(etape) >= 2:
                lines.append(f"  Action: {etape[0]} | Équipement: {etape[1]}")
        chunks.append("\n".join(lines))
    return chunks


def _chunk_metier_txt(stem: str, content: str) -> list:
    """Découpe un TXT pré-extrait (pages séparées par [Page N]) en chunks par page."""
    chunks = []
    pages = re.split(r'\[Page \d+\]', content)
    for i, page_text in enumerate(pages):
        page_text = page_text.strip()
        if len(page_text) >= _MIN_CHUNK_CHARS:
            chunks.append({
                "text": f"[Doc métier : {stem} | Page {i}]\n{page_text[:1200]}",
                "source": stem,
                "type": "metier",
            })
    return chunks


@st.cache_resource(show_spinner="Initialisation de la base de connaissances RAG…")
def build_rag_index():
    """Charge tous les documents, calcule les embeddings et l'index TF-IDF logs."""
    docs = []

    lhc_dir = BASE_DIR / "historiqueLHC"
    if lhc_dir.exists():
        for f in sorted(lhc_dir.glob("*.txt")):
            content = f.read_text(encoding="utf-8", errors="ignore")
            docs.append({"text": _format_log(f.name, content, "LHC"), "source": f.name, "type": "log_LHC"})

    lht_dir = BASE_DIR / "historiqueLHT"
    if lht_dir.exists():
        for f in sorted(lht_dir.glob("*.txt")):
            content = f.read_text(encoding="utf-8", errors="ignore")
            docs.append({"text": _format_log(f.name, content, "LHT"), "source": f.name, "type": "log_LHT"})

    for json_name, label in [("etapes_choisies_LHC.json", "LHC"), ("etapes_choisies_LHT.json", "LHT")]:
        p = BASE_DIR / json_name
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for chunk in _format_etapes(data, label):
                docs.append({"text": chunk, "source": json_name, "type": "etapes"})

    metier_dir = BASE_DIR / "contexte_metier"
    if metier_dir.exists():
        for f in sorted(metier_dir.glob("*.txt")):
            content = f.read_text(encoding="utf-8", errors="ignore")
            docs.extend(_chunk_metier_txt(f.stem, content))

    # Index sémantique (tous les docs)
    embed_model = TextEmbedding(_EMBED_MODEL)
    texts = [d["text"] for d in docs]
    embeddings = np.array(list(embed_model.embed(texts)), dtype=np.float32)

    # Index TF-IDF sur les logs uniquement (keyword matching pour les codes équipement)
    log_indices = [i for i, d in enumerate(docs) if d["type"] in ("log_LHC", "log_LHT")]
    log_texts = [docs[i]["text"] for i in log_indices]
    tfidf = TfidfVectorizer(analyzer="word", lowercase=True, ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(log_texts)

    return embed_model, docs, embeddings, tfidf, tfidf_matrix, log_indices


def retrieve_hybrid(
    query: str,
    embed_model,
    docs: list,
    embeddings: np.ndarray,
    tfidf: TfidfVectorizer,
    tfidf_matrix,
    log_indices: list,
    semantic_k: int = 3,
    keyword_k: int = 2,
) -> list:
    """Retrieval hybride : semantic (tous docs) + TF-IDF keyword (logs uniquement).

    Garantit que des logs réels remontent quand la requête concerne les historiques,
    même si les docs métier dominent en similarité sémantique.
    """
    # --- Semantic ---
    q_emb = np.array(list(embed_model.embed([query]))[0], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    norms = np.where(norms == 0, 1e-9, norms)
    sem_sims = (embeddings @ q_emb) / norms
    sem_top_idx = np.argsort(sem_sims)[::-1][: semantic_k + keyword_k]
    semantic_results = [
        {"doc": docs[i], "score": float(sem_sims[i]), "method": "semantic"}
        for i in sem_top_idx
    ]

    # --- Keyword (TF-IDF sur logs) ---
    q_tfidf = tfidf.transform([query])
    kw_sims = tfidf_cosine(q_tfidf, tfidf_matrix).flatten()
    kw_top_idx = np.argsort(kw_sims)[::-1][:keyword_k]
    keyword_results = [
        {"doc": docs[log_indices[i]], "score": float(kw_sims[i]), "method": "keyword"}
        for i in kw_top_idx
        if kw_sims[i] > 0
    ]

    # --- Fusion : semantic_k résultats sémantiques + complétion par keyword ---
    combined = list(semantic_results[:semantic_k])
    seen = {r["doc"]["source"] for r in combined}
    for r in keyword_results:
        if r["doc"]["source"] not in seen:
            combined.append(r)
            seen.add(r["doc"]["source"])

    return combined


def format_rag_context(retrieved: list, max_chars_per_doc: int = 600) -> str:
    """Formate les documents récupérés en texte à injecter dans le prompt LLM."""
    if not retrieved:
        return ""
    lines = ["=== DOCUMENTS PERTINENTS EXTRAITS DE LA BASE DE CONNAISSANCES ==="]
    for i, r in enumerate(retrieved, 1):
        doc = r["doc"]
        method_label = "keyword" if r.get("method") == "keyword" else "semantic"
        lines.append(
            f"\n[Doc {i} | Source : {doc['source']} | Score : {r['score']:.2f} | {method_label}]"
        )
        lines.append(doc["text"][:max_chars_per_doc])
    lines.append("\n=== FIN DES DOCUMENTS ===")
    return "\n".join(lines)
