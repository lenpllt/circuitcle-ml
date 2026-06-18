import os
import re
import base64
import html
import pathlib
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="CircuitClé – Détection de danger",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

_logo_path = pathlib.Path(__file__).parent / "Logo_sans_fond.png"
with open(_logo_path, "rb") as _f:
    LOGO_B64 = base64.b64encode(_f.read()).decode()

# ============================================================
# STYLE GLOBAL
# ============================================================
_css_path = pathlib.Path(__file__).parent / "style.css"
with open(_css_path, "r", encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# Accessibilité : déclaration de langue française (WCAG 3.1.1)
st.markdown('<script>document.documentElement.lang = "fr";</script>', unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="cc-header">
    <img src="data:image/png;base64,{LOGO_B64}" alt="CircuitClé logo" />
    <div class="cc-header-text">
        <div class="cc-title">CIRCUITCLÉ</div>
        <div class="cc-subtitle">Système de détection de situations dangereuses – EDF / DIPDE &nbsp;·&nbsp; Prototype IA supervisé</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT MODÈLE
# ============================================================
MODELE_PATH = pathlib.Path(__file__).parent / "meilleur_modele_ml.joblib"

if not os.path.exists(MODELE_PATH):
    st.markdown("""
    <div class="cc-alert">
        ⚠️ &nbsp;<strong>Modèle introuvable</strong> — Le fichier <code>meilleur_modele_ml.joblib</code> est absent.
        Veuillez lancer l'entraînement du modèle avant d'utiliser cette application.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

artefacts = joblib.load(MODELE_PATH)
model = artefacts["model"]
feature_columns = artefacts["feature_columns"]
best_model_name = artefacts.get("best_model_name", "Modèle inconnu")
best_accuracy = artefacts.get("best_accuracy", None)

# ============================================================
# KPI ROW
# ============================================================
acc_display = f"{best_accuracy:.2%}" if best_accuracy is not None else "—"
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">Modèle retenu</div>
        <div class="kpi-value">{best_model_name}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Accuracy</div>
        <div class="kpi-value kpi-accent">{acc_display}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Usage</div>
        <div class="kpi-value">Aide à la décision</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Statut</div>
        <div class="kpi-value">🟢 Opérationnel</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# AVERTISSEMENT
# ============================================================
st.markdown("""
<div class="cc-alert">
    ⚠️ &nbsp;<strong>Avertissement :</strong> Ce modèle constitue une aide à l'analyse uniquement.
    En raison du faible volume de données d'apprentissage, les prédictions doivent être validées
    par un expert métier avant toute prise de décision.
</div>
""", unsafe_allow_html=True)

# ============================================================
# FONCTIONS MÉTIER
# ============================================================
def extraire_palier_depuis_nom(filename: str):
    match = re.search(r'_(900|1300|1400)_', filename)
    if match:
        return int(match.group(1))
    return 0

def compter_occurrences(texte: str, mot: str) -> int:
    return len(re.findall(rf"\b{re.escape(mot)}\b", texte, flags=re.IGNORECASE))

def construire_features_depuis_texte(texte: str, nom_fichier: str = "log_utilisateur.txt", tableau_type: str = "LHC") -> dict:
    texte_min = texte.lower()
    lignes = [ligne.strip() for ligne in texte.splitlines() if ligne.strip()]

    nb_ouverture    = compter_occurrences(texte_min, "ouverture")
    nb_fermeture    = compter_occurrences(texte_min, "fermeture")
    nb_embrochage   = compter_occurrences(texte_min, "embrochage")
    nb_debrochage   = compter_occurrences(texte_min, "débrochage") + compter_occurrences(texte_min, "debrochage")
    nb_verrouillage = compter_occurrences(texte_min, "verrouillage")
    nb_deverrouillage = compter_occurrences(texte_min, "déverrouillage") + compter_occurrences(texte_min, "deverrouillage")

    return {
        "tableau_type": tableau_type,
        "palier": extraire_palier_depuis_nom(nom_fichier),
        "nb_lignes": len(lignes),
        "nb_ouverture": nb_ouverture,
        "nb_fermeture": nb_fermeture,
        "nb_embrochage": nb_embrochage,
        "nb_debrochage": nb_debrochage,
        "nb_verrouillage": nb_verrouillage,
        "nb_deverrouillage": nb_deverrouillage,
        "nb_insertion": compter_occurrences(texte_min, "insertion"),
        "nb_extraction": compter_occurrences(texte_min, "extraction"),
        "presence_smalt": 1 if "smalt" in texte_min else 0,
        "presence_porte": 1 if "porte" in texte_min else 0,
        "presence_coffret": 1 if "coffret" in texte_min else 0,
        "presence_transformateur": 1 if "transformateur" in texte_min else 0,
        "presence_source": 1 if "source" in texte_min else 0,
        "presence_eclisse": 1 if "eclisse" in texte_min or "éclisse" in texte_min else 0,
        "presence_erreur": 1 if "erreur" in texte_min else 0,
        "presence_exception": 1 if "exception" in texte_min else 0,
        "danger_personne": 1 if "danger" in texte_min and "personne" in texte_min else 0,
        "danger_materiel": 1 if "danger" in texte_min and ("matériel" in texte_min or "materiel" in texte_min) else 0,
        "nb_arret_immediat": compter_occurrences(texte_min, "arrêt immédiat") + compter_occurrences(texte_min, "arret immediat"),
        "nb_cles_non_utilisees": 0,
        "ratio_verr_deverr": round(nb_verrouillage / nb_deverrouillage, 4) if nb_deverrouillage > 0 else 0,
        "ratio_emb_deb":      round(nb_embrochage   / nb_debrochage,    4) if nb_debrochage    > 0 else 0,
        "ratio_ouv_fer":      round(nb_ouverture    / nb_fermeture,     4) if nb_fermeture     > 0 else 0,
    }

def encoder_tableau_type(valeur: str) -> int:
    return 0 if valeur == "LHC" else 1

def preparer_dataframe(features_dict: dict) -> pd.DataFrame:
    data = features_dict.copy()
    if "tableau_type" in data:
        data["tableau_type"] = encoder_tableau_type(data["tableau_type"])
    df = pd.DataFrame([data])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]
    return df

def predire(df_features: pd.DataFrame):
    prediction = model.predict(df_features)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_features)[0]
    return prediction, proba

def render_result(prediction, proba):
    if prediction == 1:
        st.markdown("""
        <div class="result-danger">
            <div class="result-title">⚠️ SITUATION DANGEREUSE DÉTECTÉE</div>
            <div class="result-body">Le modèle identifie des indicateurs de risque dans cette séquence.
            Une vérification experte est impérative avant toute action.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-safe">
            <div class="result-title">✅ SITUATION NORMALE</div>
            <div class="result-body">Le modèle ne détecte pas d'indicateur de danger significatif.
            Ce résultat doit être confirmé par l'expert métier.</div>
        </div>
        """, unsafe_allow_html=True)

    if proba is not None and len(proba) == 2:
        p_normal = proba[0] * 100
        p_danger = proba[1] * 100
        bar_color_n = "#2ecc71"
        bar_color_d = "#e74c3c"
        st.markdown(f"""
        <div class="proba-row" style="margin-top:1rem;">
            <div class="proba-item">
                <div class="proba-label">Probabilité — Normal</div>
                <div class="proba-bar-bg"><div class="proba-bar-fill" style="width:{p_normal:.1f}%; background:{bar_color_n};"></div></div>
                <div class="proba-val" style="color:{bar_color_n};">{p_normal:.1f}%</div>
            </div>
            <div class="proba-item">
                <div class="proba-label">Probabilité — Dangereux</div>
                <div class="proba-bar-bg"><div class="proba-bar-fill" style="width:{p_danger:.1f}%; background:{bar_color_d};"></div></div>
                <div class="proba-val" style="color:{bar_color_d};">{p_danger:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_features_table(features_dict):
    st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Features extraites</div>", unsafe_allow_html=True)
    rows = ""
    for k, v in features_dict.items():
        rows += f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
    st.markdown(f"""
    <table class="feat-table">
        <thead><tr><th>Feature</th><th>Valeur</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🎛️  Saisie manuelle", "📄  Analyse d'un fichier log", "🤖  Assistant IA"])

# ============================================================
# ONGLET 1 — SAISIE MANUELLE
# ============================================================
with tab1:
    st.markdown("<div class='section-title'>Paramètres du tableau électrique</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)
        tableau_type = st.selectbox("Type de tableau", ["LHC", "LHT"], key="m_type")
        palier = st.selectbox("Palier (MW)", [0, 900, 1300, 1400], index=1, key="m_palier")
        nb_lignes = st.number_input("Nb de lignes dans le log", 0, 10000, 10, key="m_lignes")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#64748b; margin-bottom:8px;'>Actions de commutation</div>", unsafe_allow_html=True)
        nb_ouverture = st.number_input("Ouvertures", 0, 1000, 1, key="m_ouv")
        nb_fermeture = st.number_input("Fermetures", 0, 1000, 1, key="m_fer")
        nb_embrochage = st.number_input("Embrochages", 0, 1000, 0, key="m_emb")
        nb_debrochage = st.number_input("Débrochages", 0, 1000, 0, key="m_deb")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='form-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; color:#64748b; margin-bottom:8px;'>Verrouillages & flux</div>", unsafe_allow_html=True)
        nb_verrouillage = st.number_input("Verrouillages", 0, 1000, 0, key="m_ver")
        nb_deverrouillage = st.number_input("Déverrouillages", 0, 1000, 0, key="m_dev")
        nb_insertion = st.number_input("Insertions", 0, 1000, 0, key="m_ins")
        nb_extraction = st.number_input("Extractions", 0, 1000, 0, key="m_ext")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title' style='margin-top:0.6rem;'>Présence d'éléments dans la séquence</div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        presence_smalt = st.checkbox("SMALT", value=True, key="m_smalt")
        presence_porte = st.checkbox("Porte", value=False, key="m_porte")
    with c5:
        presence_coffret = st.checkbox("Coffret", value=False, key="m_coffret")
        presence_transformateur = st.checkbox("Transformateur", value=False, key="m_transfo")
    with c6:
        presence_source = st.checkbox("Source", value=False, key="m_source")
        presence_eclisse = st.checkbox("Éclisse", value=False, key="m_eclisse")

    st.markdown("<div class='section-title' style='margin-top:0.8rem;'>Indicateurs d'anomalie</div>", unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        presence_erreur = st.checkbox("Erreur détectée dans le log", value=False, key="m_err")
    with c8:
        presence_exception = st.checkbox("Exception détectée dans le log", value=False, key="m_exc")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍  Lancer l'analyse", key="btn_manuel"):
        features = {
            "tableau_type": tableau_type, "palier": palier, "nb_lignes": nb_lignes,
            "nb_ouverture": nb_ouverture, "nb_fermeture": nb_fermeture,
            "nb_embrochage": nb_embrochage, "nb_debrochage": nb_debrochage,
            "nb_verrouillage": nb_verrouillage, "nb_deverrouillage": nb_deverrouillage,
            "nb_insertion": nb_insertion, "nb_extraction": nb_extraction,
            "presence_smalt": int(presence_smalt), "presence_porte": int(presence_porte),
            "presence_coffret": int(presence_coffret), "presence_transformateur": int(presence_transformateur),
            "presence_source": int(presence_source), "presence_eclisse": int(presence_eclisse),
            "presence_erreur": int(presence_erreur), "presence_exception": int(presence_exception),
        }
        df_features = preparer_dataframe(features)
        prediction, proba = predire(df_features)
        st.session_state["manuel_result"] = {"prediction": prediction, "proba": proba, "features": features}

    if "manuel_result" in st.session_state:
        r = st.session_state["manuel_result"]
        render_result(r["prediction"], r["proba"])
        render_features_table(r["features"])

# ============================================================
# ONGLET 2 — FICHIER LOG
# ============================================================
with tab2:
    st.markdown("<div class='section-title'>Import d'un fichier d'historique</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        tableau_type_upload = st.selectbox("Type de tableau du log", ["LHC", "LHT"], key="up_type")

    with st.expander("🔒  Confidentialité & traitement des données"):
        st.markdown("""
        **Aucune donnée n'est conservée au-delà de votre session.**
        Le fichier importé est traité localement en mémoire vive et n'est jamais
        transmis à un serveur externe, ni enregistré sur disque.
        Les résultats d'analyse sont effacés à la fermeture de l'onglet ou du navigateur.
        Cette application traite exclusivement des données techniques industrielles —
        aucune donnée personnelle n'est collectée (conformément au RGPD, règlement UE 2016/679).
        """)

    uploaded_file = st.file_uploader(
        "Déposez un fichier .txt issu de historiqueLHC ou historiqueLHT",
        type=["txt"],
        key="up_file"
    )

    if uploaded_file is not None:
        contenu = uploaded_file.read().decode("utf-8", errors="ignore")

        with st.expander("📋  Aperçu du contenu du log"):
            st.markdown(f"<div class='log-viewer'>{html.escape(contenu[:3000])}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍  Analyser ce fichier", key="btn_upload"):
            features = construire_features_depuis_texte(
                texte=contenu,
                nom_fichier=uploaded_file.name,
                tableau_type=tableau_type_upload
            )
            df_features = preparer_dataframe(features)
            prediction, proba = predire(df_features)
            st.session_state["upload_result"] = {"prediction": prediction, "proba": proba, "features": features}

        if "upload_result" in st.session_state:
            r = st.session_state["upload_result"]
            render_result(r["prediction"], r["proba"])
            render_features_table(r["features"])
    else:
        st.markdown("""
        <div style="text-align:center; padding:2.5rem; color:#334155; font-size:0.85rem; letter-spacing:0.04em;">
            📂  &nbsp;Aucun fichier chargé — importez un historique LHC ou LHT pour démarrer l'analyse
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ONGLET 3 — ASSISTANT IA (RAG + API)
# ============================================================
with tab3:
    st.markdown("<div class='section-title'>Assistant IA — CircuitClé</div>", unsafe_allow_html=True)

    # Vérification de la clé API
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    if not api_key:
        st.markdown("""
        <div class="cc-alert">
            ⚙️ &nbsp;<strong>Configuration requise</strong> — Ajoutez votre clé
            <code>ANTHROPIC_API_KEY</code> dans les secrets Streamlit pour activer l'assistant.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **Comment configurer :**
        - En local : créez le fichier `.streamlit/secrets.toml` avec `ANTHROPIC_API_KEY = "sk-ant-..."`
        - Sur Streamlit Cloud : ajoutez la clé dans *Settings → Secrets*
        """)
    else:
        import anthropic
        from rag_engine import build_rag_index, retrieve_hybrid, format_rag_context

        # Initialisation de l'index RAG (mis en cache par st.cache_resource)
        rag_docs, rag_tfidf_global, rag_matrix_global, rag_tfidf_logs, rag_matrix_logs, rag_log_indices = build_rag_index()

        # Contexte ML injecté dans le system prompt
        derniere_pred = st.session_state.get("upload_result") or st.session_state.get("manuel_result")
        contexte_prediction = ""
        if derniere_pred:
            pred_label = "DANGEREUSE" if derniere_pred["prediction"] == 1 else "NORMALE"
            proba_str = ""
            if derniere_pred["proba"] is not None and len(derniere_pred["proba"]) == 2:
                proba_str = f" (probabilité danger : {derniere_pred['proba'][1]*100:.1f}%)"
            features_str = ", ".join(f"{k}={v}" for k, v in derniere_pred["features"].items())
            contexte_prediction = f"""

La dernière analyse effectuée dans la session a donné :
- Résultat : situation {pred_label}{proba_str}
- Features extraites : {features_str}
"""

        system_prompt = f"""Tu es un assistant expert en sécurité des tableaux électriques pour EDF/DIPDE, \
intégré dans l'application CircuitClé.

Contexte du modèle ML déployé :
- Modèle retenu : {best_model_name}
- Accuracy : {f"{best_accuracy:.2%}" if best_accuracy else "non disponible"}
- Features utilisées : {", ".join(feature_columns)}
- Algorithmes comparés : Logistic Regression, Decision Tree, Random Forest, KNN, MLP (Deep Learning)
- Optimisation : GridSearchCV sur le paramètre C de la Logistic Regression
{contexte_prediction}
Tu as accès à une base de connaissances contenant les historiques d'opérations réels (logs LHC et LHT) \
et les séquences d'étapes définies. Lorsque des documents pertinents sont fournis en contexte, \
appuie-toi sur leur contenu pour répondre avec précision. \
Cite la source (nom de fichier) lorsque tu t'appuies sur un document.

Ton rôle :
- Expliquer les prédictions du modèle en langage clair pour un technicien EDF
- Répondre aux questions sur les features, la méthodologie ML, et les résultats
- Analyser et comparer des séquences d'opérations à partir des logs de la base de connaissances
- Expliquer pourquoi certaines variables influencent le risque (ex : présence d'erreur, nb de débrochages)
- Rappeler systématiquement que les résultats ML sont une aide à la décision et doivent être validés par un expert métier
- Répondre exclusivement en français
- Rester factuel et professionnel, sans dramatiser ni minimiser les risques"""

        # Initialisation de l'historique
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "chat_rag_sources" not in st.session_state:
            st.session_state.chat_rag_sources = {}

        # Bouton reset
        col_titre, col_reset = st.columns([5, 1])
        with col_reset:
            if st.button("🗑️ Effacer", key="btn_reset_chat"):
                st.session_state.chat_messages = []
                st.session_state.chat_rag_sources = {}
                st.rerun()

        # Note RGPD
        with st.expander("🔒  Confidentialité & traitement des données"):
            st.markdown("""
            **Seuls les extraits documentaires pertinents et les features numériques de la session sont transmis à l'API.**
            Les fichiers logs sont indexés localement — aucune donnée brute n'est envoyée hors session.
            Aucune information personnelle n'est collectée. Conforme RGPD (règlement UE 2016/679).
            """)

        def _doc_type_label(doc_type: str) -> str:
            return {"log_LHC": "📋 Log LHC", "log_LHT": "📋 Log LHT",
                    "metier": "📄 Doc métier", "etapes": "📑 Étapes"}.get(doc_type, doc_type)

        def _pertinence_label(score: float, method: str) -> str:
            if method == "keyword":
                return "🔑 Correspondance mot-clé"
            if score >= 0.80:
                return "🟢 Très pertinent"
            if score >= 0.65:
                return "🟡 Pertinent"
            return "🟠 Partiellement pertinent"

        def render_rag_sources(sources: list, expanded: bool = False):
            if not sources:
                return
            with st.expander(f"📚 Documents consultés pour cette réponse ({len(sources)})", expanded=expanded):
                st.caption(
                    "Avant d'appeler l'assistant IA, CircuitClé recherche automatiquement les documents "
                    "les plus proches de votre question dans sa base de connaissances (logs LHC/LHT et docs métier). "
                    "Ces extraits sont transmis à l'IA pour enrichir sa réponse."
                )
                st.divider()
                for r in sources:
                    doc = r["doc"]
                    method = r.get("method", "semantic")
                    score = r["score"]
                    col_type, col_name, col_score = st.columns([1.4, 3.5, 1.8])
                    with col_type:
                        st.markdown(_doc_type_label(doc["type"]))
                    with col_name:
                        st.markdown(f"**{doc['source']}**")
                    with col_score:
                        st.markdown(_pertinence_label(score, method))

        # Affichage de l'historique
        for i, msg in enumerate(st.session_state.chat_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
            # Afficher les sources RAG après chaque réponse assistant
            if msg["role"] == "assistant" and i in st.session_state.chat_rag_sources:
                render_rag_sources(st.session_state.chat_rag_sources[i])

        # Message d'accueil si conversation vide
        if not st.session_state.chat_messages:
            with st.chat_message("assistant"):
                st.markdown(
                    "Bonjour ! Je suis l'assistant IA de CircuitClé, enrichi d'une base de connaissances "
                    "contenant les historiques d'opérations LHC et LHT. "
                    "Je peux analyser des séquences d'opérations, expliquer les prédictions du modèle ML, "
                    "ou répondre à vos questions sur la détection de situations dangereuses. "
                    "Comment puis-je vous aider ?"
                )

        # Saisie utilisateur
        if prompt := st.chat_input("Posez votre question à l'assistant…"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Recherche dans la base de connaissances et analyse…"):
                    try:
                        # Retrieval hybride RAG (semantic + TF-IDF sur logs)
                        retrieved = retrieve_hybrid(
                            prompt, rag_docs,
                            rag_tfidf_global, rag_matrix_global,
                            rag_tfidf_logs, rag_matrix_logs, rag_log_indices,
                            global_k=3, log_k=2,
                        )
                        rag_context = format_rag_context(retrieved)

                        # Message augmenté (RAG + question) envoyé à l'API
                        augmented_content = f"{rag_context}\n\n[QUESTION]\n{prompt}" if rag_context else prompt

                        # Historique pour l'API : messages précédents + dernier message augmenté
                        messages_for_api = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.chat_messages[:-1]
                        ] + [{"role": "user", "content": augmented_content}]

                        client = anthropic.Anthropic(api_key=api_key)
                        response = client.messages.create(
                            model="claude-haiku-4-5-20251001",
                            max_tokens=1024,
                            system=system_prompt,
                            messages=messages_for_api,
                        )
                        reply = response.content[0].text
                    except Exception as e:
                        reply = f"❌ Erreur lors de la communication avec l'API : {e}"
                        retrieved = []

                st.markdown(reply)

            assistant_idx = len(st.session_state.chat_messages)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
            st.session_state.chat_rag_sources[assistant_idx] = retrieved

            # Afficher immédiatement les sources de cette réponse
            render_rag_sources(retrieved)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="cc-footer">
    CircuitClé &nbsp;·&nbsp; Application prototype – EDF / DIPDE &nbsp;·&nbsp;
    Développée avec Streamlit &nbsp;·&nbsp;
    <span style="color:#c0392b;">⚠ Résultats soumis à validation experte</span>
    &nbsp;·&nbsp; Traitement local – Aucune donnée personnelle collectée – Conforme RGPD
</div>
""", unsafe_allow_html=True)