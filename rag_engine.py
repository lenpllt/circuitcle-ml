import re
import json
import pathlib
import numpy as np
import streamlit as st
from fastembed import TextEmbedding

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
    """Charge tous les documents, calcule les embeddings et retourne l'index en cache."""
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

    # Documents métier (PDFs pré-extraits en TXT)
    metier_dir = BASE_DIR / "contexte_metier"
    if metier_dir.exists():
        for f in sorted(metier_dir.glob("*.txt")):
            content = f.read_text(encoding="utf-8", errors="ignore")
            docs.extend(_chunk_metier_txt(f.stem, content))

    model = TextEmbedding(_EMBED_MODEL)
    texts = [d["text"] for d in docs]
    embeddings = np.array(list(model.embed(texts)), dtype=np.float32)

    return model, docs, embeddings


def retrieve(query: str, model, docs: list, embeddings: np.ndarray, top_k: int = 3) -> list:
    """Retourne les top_k documents les plus pertinents pour la requête."""
    q_emb = np.array(list(model.embed([query]))[0], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    norms = np.where(norms == 0, 1e-9, norms)
    sims = (embeddings @ q_emb) / norms
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [{"doc": docs[i], "score": float(sims[i])} for i in top_idx]


def format_rag_context(retrieved: list, max_chars_per_doc: int = 600) -> str:
    """Formate les documents récupérés en texte à injecter dans le prompt LLM."""
    if not retrieved:
        return ""
    lines = ["=== DOCUMENTS PERTINENTS EXTRAITS DE LA BASE DE CONNAISSANCES ==="]
    for i, r in enumerate(retrieved, 1):
        doc = r["doc"]
        lines.append(f"\n[Doc {i} | Source : {doc['source']} | Similarité : {r['score']:.2f}]")
        lines.append(doc["text"][:max_chars_per_doc])
    lines.append("\n=== FIN DES DOCUMENTS ===")
    return "\n".join(lines)
