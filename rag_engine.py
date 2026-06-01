import re
import json
import pathlib
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as tfidf_cosine

BASE_DIR = pathlib.Path(__file__).parent
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
    """Charge tous les documents et construit deux index TF-IDF :
    - global (tous docs) pour la recherche générale
    - logs uniquement pour la recherche par code équipement
    """
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

    # Index TF-IDF global (tous les docs) — recherche thématique
    all_texts = [d["text"] for d in docs]
    tfidf_global = TfidfVectorizer(
        analyzer="word", lowercase=True, ngram_range=(1, 2),
        sublinear_tf=True, min_df=1,
    )
    matrix_global = tfidf_global.fit_transform(all_texts)

    # Index TF-IDF logs uniquement — recherche par code équipement / action
    log_indices = [i for i, d in enumerate(docs) if d["type"] in ("log_LHC", "log_LHT")]
    log_texts = [docs[i]["text"] for i in log_indices]
    tfidf_logs = TfidfVectorizer(
        analyzer="word", lowercase=True, ngram_range=(1, 2),
        sublinear_tf=True, min_df=1,
    )
    matrix_logs = tfidf_logs.fit_transform(log_texts)

    return docs, tfidf_global, matrix_global, tfidf_logs, matrix_logs, log_indices


def retrieve_hybrid(
    query: str,
    docs: list,
    tfidf_global: TfidfVectorizer,
    matrix_global,
    tfidf_logs: TfidfVectorizer,
    matrix_logs,
    log_indices: list,
    global_k: int = 3,
    log_k: int = 2,
) -> list:
    """Retrieval hybride :
    - TF-IDF global (top global_k) pour les docs métier et les questions conceptuelles
    - TF-IDF logs (top log_k) pour garantir des logs quand la requête mentionne des équipements
    Fusion avec déduplication par source.
    """
    # --- Recherche globale ---
    q_global = tfidf_global.transform([query])
    sims_global = tfidf_cosine(q_global, matrix_global).flatten()
    top_global = np.argsort(sims_global)[::-1][:global_k]
    global_results = [
        {"doc": docs[i], "score": float(sims_global[i]), "method": "global"}
        for i in top_global if sims_global[i] > 0
    ]

    # --- Recherche dans les logs ---
    q_logs = tfidf_logs.transform([query])
    sims_logs = tfidf_cosine(q_logs, matrix_logs).flatten()
    top_logs = np.argsort(sims_logs)[::-1][:log_k]
    log_results = [
        {"doc": docs[log_indices[i]], "score": float(sims_logs[i]), "method": "keyword"}
        for i in top_logs if sims_logs[i] > 0
    ]

    # --- Fusion ---
    combined = list(global_results)
    seen = {r["doc"]["source"] for r in combined}
    for r in log_results:
        if r["doc"]["source"] not in seen:
            combined.append(r)
            seen.add(r["doc"]["source"])

    return combined


def format_rag_context(retrieved: list, max_chars_per_doc: int = 600) -> str:
    if not retrieved:
        return ""
    lines = ["=== DOCUMENTS PERTINENTS EXTRAITS DE LA BASE DE CONNAISSANCES ==="]
    for i, r in enumerate(retrieved, 1):
        doc = r["doc"]
        method_label = "keyword" if r.get("method") == "keyword" else "global"
        lines.append(
            f"\n[Doc {i} | Source : {doc['source']} | Score : {r['score']:.2f} | {method_label}]"
        )
        lines.append(doc["text"][:max_chars_per_doc])
    lines.append("\n=== FIN DES DOCUMENTS ===")
    return "\n".join(lines)
