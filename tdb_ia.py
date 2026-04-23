import json
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ── Chemins ───────────────────────────────────────────────────────────────
BASE           = pathlib.Path(__file__).parent
DATASET_PATH   = BASE / "dataset_logs_ml.csv"
CSS_PATH       = BASE / "style.css"
BENCHMARK_PATH = BASE / "benchmark_resultats.json"

# ── Palette (cohérente avec style.css) ───────────────────────────────────
BG_CARD    = "#111827"
BG_PLOT    = "#0d1520"
RED        = "#e74c3c"
RED_DARK   = "#c0392b"
GREEN      = "#2ecc71"
BLUE_GRID  = "#1e3a5f"
TEXT       = "#e2e8f0"
TEXT_MUTED = "#94a3b8"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="CircuitClé – TDB IA", page_icon="📊", layout="wide")

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
st.markdown('<script>document.documentElement.lang = "fr";</script>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cc-header">
  <div class="cc-header-text">
    <div class="cc-title">TABLEAU DE BORD — ANALYSE IA</div>
    <div class="cc-subtitle">
      Performances du modèle ML · Détection de situations dangereuses · EDF / DIPDE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Calcul ML (mis en cache pour éviter de ré-entraîner à chaque interaction) ──
@st.cache_data
def compute_ml_metrics():
    df = pd.read_csv(DATASET_PATH)
    X  = df.drop(columns=["label_danger", "nom_fichier"])
    y  = df["label_danger"]

    X = X.copy()
    le = LabelEncoder()
    X["tableau_type"] = le.fit_transform(X["tableau_type"])
    colonnes = list(X.columns)

    pre = ColumnTransformer([("num", StandardScaler(), colonnes)])

    # Forcer les cas dangereux dans l'entraînement (cf. train_modele_ml.py)
    X_danger = X[y == 1];  y_danger = y[y == 1]
    X_normal = X[y == 0];  y_normal = y[y == 0]
    X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(
        X_normal, y_normal, test_size=0.30, random_state=42
    )
    X_train = pd.concat([X_tr_n, X_danger]).reset_index(drop=True)
    y_train = pd.concat([y_tr_n, y_danger]).reset_index(drop=True)
    X_test  = X_te_n.reset_index(drop=True)
    y_test  = y_te_n.reset_index(drop=True)

    modeles = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=3),
        "MLP (Deep Learning)": MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu",
            solver="adam", max_iter=2000, random_state=42
        ),
    }

    resultats = {}
    best_score, best_name, best_pipe, lr_pipe = -1, None, None, None

    for nom, modele in modeles.items():
        pipe = Pipeline([("preprocessor", pre), ("classifier", modele)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        score  = accuracy_score(y_test, y_pred)
        # Rappel sur cas dangereux (prédiction sur les exemples dangereux d'entraînement)
        pred_danger = pipe.predict(X_danger)
        rappel_danger = float((pred_danger == 1).sum()) / len(pred_danger)

        resultats[nom] = {
            "accuracy": round(float(score), 4),
            "rappel_danger": round(rappel_danger, 4),
        }
        if score > best_score:
            best_score, best_name, best_pipe = score, nom, pipe
        if nom == "Logistic Regression":
            lr_pipe = pipe

    y_pred_best = best_pipe.predict(X_test)
    cm_best     = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
    lr_coef     = lr_pipe.named_steps["classifier"].coef_[0]

    return {
        "resultats":  resultats,
        "best_name":  best_name,
        "best_score": round(float(best_score), 4),
        "cm":         cm_best.tolist(),
        "lr_feat":    colonnes,
        "lr_coef":    lr_coef.tolist(),
        "n_total":    len(df),
        "n_danger":   int(y.sum()),
        "n_normal":   int((y == 0).sum()),
        "n_danger_train": len(X_danger),
    }


data = compute_ml_metrics()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPIs synthèse
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">01 — Synthèse du modèle</p>', unsafe_allow_html=True)

kpis = [
    ("Modèle retenu",  data["best_name"],                            ""),
    ("Accuracy",       f"{data['best_score'] * 100:.1f} %",         "kpi-accent"),
    ("Taille dataset", f"{data['n_total']} logs",                    ""),
    ("Répartition",    f"{data['n_normal']} normaux · {data['n_danger']} dangereux", ""),
]

for col, (label, value, extra) in zip(st.columns(len(kpis)), kpis):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {extra}">{value}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Comparaison des algorithmes
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">02 — Comparaison des algorithmes</p>', unsafe_allow_html=True)

noms           = list(data["resultats"].keys())
accs           = [data["resultats"][n]["accuracy"]      for n in noms]
rappels_danger = [data["resultats"][n]["rappel_danger"] for n in noms]
colors_acc     = [RED if n == data["best_name"] else BLUE_GRID for n in noms]
colors_rappel  = [RED if r == 1.0 else "#3498db" if r > 0 else "#4b5563" for r in rappels_danger]

c2a, c2b = st.columns(2)

with c2a:
    fig_acc = go.Figure(go.Bar(
        x=accs, y=noms, orientation="h",
        marker_color=colors_acc,
        text=[f"{a * 100:.0f} %" for a in accs],
        textposition="outside", textfont=dict(color=TEXT, size=12),
    ))
    fig_acc.update_layout(
        title=dict(text="Accuracy — jeu de test (cas normaux uniquement)", font=dict(color=TEXT_MUTED, size=12)),
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT,
        font=dict(family="IBM Plex Sans", color=TEXT),
        xaxis=dict(range=[0, 1.2], gridcolor=BLUE_GRID, tickformat=".0%"),
        yaxis=dict(gridcolor=BG_PLOT),
        margin=dict(l=10, r=60, t=40, b=10),
        height=260, showlegend=False,
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    st.caption("⚠ 100% trivial : le jeu de test ne contient que des cas normaux.")

with c2b:
    fig_rappel = go.Figure(go.Bar(
        x=rappels_danger, y=noms, orientation="h",
        marker_color=colors_rappel,
        text=[f"{r * 100:.0f} %" for r in rappels_danger],
        textposition="outside", textfont=dict(color=TEXT, size=12),
    ))
    fig_rappel.update_layout(
        title=dict(
            text=f"Rappel danger — {data['n_danger_train']} cas dangereux (entraînement)",
            font=dict(color=TEXT_MUTED, size=12)
        ),
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT,
        font=dict(family="IBM Plex Sans", color=TEXT),
        xaxis=dict(range=[0, 1.2], gridcolor=BLUE_GRID, tickformat=".0%"),
        yaxis=dict(gridcolor=BG_PLOT),
        margin=dict(l=10, r=60, t=40, b=10),
        height=260, showlegend=False,
    )
    st.plotly_chart(fig_rappel, use_container_width=True)
    st.caption("Rouge = détecte tous les cas dangereux · Bleu = détection partielle · Gris = aucune détection")


# ═══════════════════════════════════════════════════════════════════════════
# SECTIONS 3 & 4 — Matrice de confusion + Importance des variables
# ═══════════════════════════════════════════════════════════════════════════
col3, col4 = st.columns(2)

with col3:
    st.markdown('<p class="section-title">03 — Matrice de confusion</p>', unsafe_allow_html=True)

    cm      = np.array(data["cm"])
    etiq    = ["Normal", "Danger"]
    cell_lbl = [["VN", "FP"], ["FN", "VP"]]
    annots  = [
        [f"<b>{cm[i, j]}</b><br><span style='font-size:10px'>{cell_lbl[i][j]}</span>"
         for j in range(2)]
        for i in range(2)
    ]

    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Prédit {e}" for e in etiq],
        y=[f"Réel {e}" for e in etiq],
        colorscale=[[0, BG_PLOT], [1, RED_DARK]],
        showscale=False,
        text=annots,
        texttemplate="%{text}",
        textfont=dict(size=15, color=TEXT),
    ))
    fig_cm.update_layout(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT,
        font=dict(family="IBM Plex Sans", color=TEXT),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption(
        "VN = Vrai Négatif · FP = Faux Positif · FN = Faux Négatif · VP = Vrai Positif\n\n"
        "⚠ Le jeu de test ne contient que des cas normaux : "
        "les exemples dangereux sont forcés en entraînement (dataset de 11 logs)."
    )

with col4:
    st.markdown('<p class="section-title">04 — Importance des variables</p>', unsafe_allow_html=True)

    pairs = sorted(zip(data["lr_feat"], data["lr_coef"]), key=lambda x: abs(x[1]))
    s_feats, s_coefs = zip(*pairs)
    feat_colors = [RED if c > 0 else "#3498db" for c in s_coefs]

    fig_feat = go.Figure(go.Bar(
        x=list(s_coefs), y=list(s_feats),
        orientation="h",
        marker_color=feat_colors,
    ))
    fig_feat.update_layout(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT,
        font=dict(family="IBM Plex Sans", color=TEXT, size=11),
        xaxis=dict(
            title="Coefficient (Logistic Regression)",
            gridcolor=BLUE_GRID, zerolinecolor=TEXT_MUTED,
        ),
        yaxis=dict(gridcolor=BG_PLOT),
        margin=dict(l=10, r=10, t=10, b=30),
        height=500,
        showlegend=False,
    )
    st.plotly_chart(fig_feat, use_container_width=True)
    st.caption("Rouge = facteur de risque (corrélé au danger) · Bleu = facteur protecteur")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Benchmark SQL
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">05 — Benchmark SQL — Impact des index</p>', unsafe_allow_html=True)

if BENCHMARK_PATH.exists():
    with open(BENCHMARK_PATH) as f:
        bench = json.load(f)

    requetes    = list(bench["avant"].keys())
    noms_affich = ["Clef / cellule", "Partie mobile / cellule", "SMALT / cellule"]
    t_avant_ms  = [bench["avant"][r] * 1_000 for r in requetes]
    t_apres_ms  = [bench["apres"][r] * 1_000 for r in requetes]
    gains       = [
        (a - b) / a * 100 if a > 0 else 0
        for a, b in zip(t_avant_ms, t_apres_ms)
    ]

    fig_sql = go.Figure([
        go.Bar(
            name="Avant index", x=noms_affich, y=t_avant_ms,
            marker_color=RED_DARK,
            text=[f"{v:.4f} ms" for v in t_avant_ms],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ),
        go.Bar(
            name="Après index", x=noms_affich, y=t_apres_ms,
            marker_color=GREEN,
            text=[f"{v:.4f} ms" for v in t_apres_ms],
            textposition="outside", textfont=dict(color=TEXT, size=11),
        ),
    ])
    fig_sql.update_layout(
        barmode="group",
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT,
        font=dict(family="IBM Plex Sans", color=TEXT),
        xaxis=dict(gridcolor=BLUE_GRID),
        yaxis=dict(title="Temps moyen (ms · 100 répétitions)", gridcolor=BLUE_GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color=TEXT)),
        margin=dict(l=10, r=10, t=50, b=10),
        height=340,
    )
    st.plotly_chart(fig_sql, use_container_width=True)

    gain_df = pd.DataFrame({
        "Requête":            noms_affich,
        "Avant index (ms)":  [f"{v:.6f}" for v in t_avant_ms],
        "Après index (ms)":  [f"{v:.6f}" for v in t_apres_ms],
        "Gain":              [f"{g:+.1f} %" for g in gains],
    })
    st.dataframe(gain_df, use_container_width=True, hide_index=True)

else:
    st.info(
        "Fichier `benchmark_resultats.json` introuvable. "
        "Lancez `benchmark_sql.py` localement pour générer les résultats, "
        "puis committez le fichier JSON dans le repo."
    )

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cc-footer">
    Circuit Clé – Tableau de Bord IA · EDF / DIPDE · M2 Data Science
</div>
""", unsafe_allow_html=True)
