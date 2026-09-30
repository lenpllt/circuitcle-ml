import html
import pathlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Chemins ───────────────────────────────────────────────────────────────
BASE      = pathlib.Path(__file__).parent
XLSX_PATH = BASE / "TDB_Pilotage_financier_CircuitCle.xlsx"
CSS_PATH  = BASE / "style.css"

# ── Palette (cohérente avec style.css et tdb_ia.py) ──────────────────────
BG_CARD    = "#111827"
BG_PLOT    = "#0d1520"
RED        = "#e74c3c"
GREEN      = "#2ecc71"
BLUE       = "#3498db"
AMBER      = "#f39c12"
BLUE_GRID  = "#1e3a5f"
TEXT       = "#e2e8f0"
TEXT_MUTED = "#94a3b8"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="CircuitClé – TDB Pilotage financier", page_icon="💰", layout="wide")

if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)
st.markdown('<script>document.documentElement.lang = "fr";</script>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cc-header">
  <div class="cc-header-text">
    <div class="cc-title">TABLEAU DE BORD — PILOTAGE FINANCIER</div>
    <div class="cc-subtitle">
      Coûts, délais, écarts et retour sur investissement · Projet CircuitClé · EDF / DIPDE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Chargement des données depuis le classeur source ─────────────────────
@st.cache_data
def charger_donnees():
    """Lit les valeurs saisies du classeur. Les indicateurs sont recalculés ici,
    afin que les curseurs de simulation agissent en direct."""
    params  = pd.read_excel(XLSX_PATH, sheet_name="Parametres")
    sprints = pd.read_excel(XLSX_PATH, sheet_name="Sprints")
    couts   = pd.read_excel(XLSX_PATH, sheet_name="Couts")
    ecarts  = pd.read_excel(XLSX_PATH, sheet_name="Ecarts")
    charge  = pd.read_excel(XLSX_PATH, sheet_name="Charge")
    p = dict(zip(params["Paramètre"], params["Valeur"]))
    return p, sprints, couts, ecarts, charge


try:
    P, SPRINTS, COUTS, ECARTS, CHARGE = charger_donnees()
except FileNotFoundError:
    st.error(f"Classeur source introuvable : {XLSX_PATH.name}")
    st.stop()

SALAIRE     = P["Salaire mensuel alternante"]
PART_TEMPS  = P["Part du temps alternante sur le projet"]
DUREE       = P["Durée du projet"]
COUT_PRESTA = P["Coût mensuel du contrat prestataire"]
QUOTE_PART  = P["Quote-part du prestataire sur CircuitClé"]
COUT_API    = P["Coût API Claude Haiku"]
COUT_INFRA  = P["Coût infrastructure (poste de travail)"]
COUT_OUTILS = P["Coût outils internes"]
TAUX_HORAIRE = P["Taux horaire ingénieur DIPDE"]
TEMPS_REVUE  = P["Temps de revue manuelle par analyse"]
ANALYSES_MOIS_DEFAUT = int(P["Analyses par mois (hypothèse)"])

GAIN_ANALYSE = TAUX_HORAIRE * TEMPS_REVUE
COUT_DEV     = round(SALAIRE * PART_TEMPS * DUREE, -2)


# ── Curseurs de simulation ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">Simulation</p>', unsafe_allow_html=True)
    quote_part = st.slider(
        "Quote-part du prestataire sur le projet",
        min_value=0, max_value=100, value=int(QUOTE_PART * 100), step=5,
        format="%d %%",
        help="Part du temps de Marius imputée à CircuitClé. Quote-part de 25 % validée par le tuteur ; le curseur permet de tester d'autres répartitions.",
    ) / 100
    analyses_mois = st.slider(
        "Analyses réalisées par mois",
        min_value=1, max_value=100, value=ANALYSES_MOIS_DEFAUT, step=1,
        help="Cadence d'utilisation opérationnelle. Paramètre de simulation, à caler avec le métier.",
    )
    st.markdown("---")
    st.caption(
        f"**Hypothèses de référence**  \n"
        f"Alternante : {PART_TEMPS * 100:.0f} % × {SALAIRE:,.0f} €/mois × {DUREE} mois  \n"
        f"Prestataire : {COUT_PRESTA:,.0f} €/mois × {DUREE} mois × quote-part  \n"
        f"Gain par analyse : {TAUX_HORAIRE:.0f} €/h × {TEMPS_REVUE} h = {GAIN_ANALYSE:.0f} €"
        .replace(",", " ")
    )

# ── Recalcul en fonction des curseurs ─────────────────────────────────────
cout_presta_impute = COUT_PRESTA * DUREE * quote_part
cout_total   = COUT_DEV + COUT_INFRA + COUT_OUTILS + cout_presta_impute + COUT_API
cout_interne = COUT_DEV + COUT_INFRA + COUT_OUTILS
cout_externe = cout_presta_impute + COUT_API
seuil        = int(-(-cout_total // GAIN_ANALYSE))          # arrondi supérieur
mois_rentab  = seuil / analyses_mois

# Répartition par sprint, au prorata de la durée de chaque sprint
poids        = SPRINTS["Poids"].tolist()
codes_sprint = SPRINTS["Sprint"].tolist()
# Coûts étalés sur la durée du projet : humains, infrastructure et licences
cout_etale   = COUT_DEV + COUT_INFRA + COUT_OUTILS + cout_presta_impute
par_sprint   = [cout_etale * w for w in poids]
for i, code in enumerate(codes_sprint):                      # API répartie sur S3 et S4
    if code in ("S3", "S4"):
        par_sprint[i] += COUT_API / 2
cumul = [sum(par_sprint[: i + 1]) for i in range(len(par_sprint))]

ECART_BUDGET = 0.0                                           # réel = prévu (cf. onglet Lisez-moi)
TAUX_CONSO   = 1.0


def bloc_kpi(colonnes, kpis):
    for col, (label, valeur, extra) in zip(colonnes, kpis):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {extra}">{valeur}</div>
        </div>""", unsafe_allow_html=True)


def mise_en_forme(fig, titre, hauteur=280):
    fig.update_layout(
        title=dict(text=titre, font=dict(color=TEXT_MUTED, size=12)),
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_PLOT,
        font=dict(family="IBM Plex Sans", color=TEXT),
        margin=dict(l=10, r=20, t=40, b=10),
        height=hauteur,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Synthèse budgétaire
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">01 — Synthèse budgétaire</p>', unsafe_allow_html=True)

bloc_kpi(st.columns(4), [
    ("Budget prévu",       f"{cout_total:,.0f} €".replace(",", " "),   ""),
    ("Budget consommé",    f"{cout_total:,.0f} €".replace(",", " "),   ""),
    ("Écart",              f"{ECART_BUDGET:+,.0f} €".replace(",", " "), "kpi-accent"),
    ("Taux de consommation", f"{TAUX_CONSO * 100:.0f} %",                       ""),
])
st.markdown("<br>", unsafe_allow_html=True)

col_jauge, col_texte = st.columns([1, 2])

with col_jauge:
    fig_jauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=TAUX_CONSO * 100,
        number=dict(suffix=" %", font=dict(color=TEXT, size=30)),
        gauge=dict(
            axis=dict(range=[0, 130], tickcolor=TEXT_MUTED, tickfont=dict(size=10)),
            bar=dict(color=GREEN, thickness=0.7),
            bgcolor=BG_PLOT,
            borderwidth=0,
            steps=[
                dict(range=[0, 90],   color="#16243c"),
                dict(range=[90, 110], color="#1b3a2c"),
                dict(range=[110, 130], color="#3a1a18"),
            ],
            threshold=dict(line=dict(color=RED, width=3), thickness=0.8, value=100),
        ),
    ))
    st.plotly_chart(mise_en_forme(fig_jauge, "Taux de consommation du budget (cible 100 %)"),
                    use_container_width=True)

with col_texte:
    st.markdown(f"""
    <div class="kpi-card" style="text-align:left; padding:1.2rem 1.4rem;">
      <div class="kpi-label">Lecture</div>
      <p style="color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.6; margin:0.6rem 0 0 0;">
        Le budget a été tenu à 100 % : les coûts réels sont conformes aux estimations initiales et
        les quatre sprints ont été livrés dans les délais. Les écarts rencontrés en cours de projet
        n'ont pas porté sur le budget mais sur le <b>périmètre</b> et les <b>ressources</b> —
        ils sont détaillés en section 04, avec la décision d'ajustement associée.
      </p>
      <p style="color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.6; margin:0.8rem 0 0 0;">
        Le coût complet intègre la ressource prestataire mutualisée
        ({cout_presta_impute:,.0f} € pour une quote-part de {quote_part * 100:.0f} %), absente du budget
        présenté en première session.
      </p>
    </div>""".replace(",", " "), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Répartition des coûts
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">02 — Répartition des coûts</p>', unsafe_allow_html=True)

col_anneau, col_barres = st.columns(2)

with col_anneau:
    fig_anneau = go.Figure(go.Pie(
        labels=["Coûts internes", "Coûts externes"],
        values=[cout_interne, cout_externe],
        hole=0.55,
        marker=dict(colors=[BLUE, RED], line=dict(color=BG_CARD, width=2)),
        textinfo="label+percent",
        textfont=dict(color=TEXT, size=12),
        hovertemplate="%{label}<br>%{value:,.0f} €<extra></extra>",
    ))
    fig_anneau.update_layout(showlegend=False)
    st.plotly_chart(mise_en_forme(fig_anneau, "Interne / externe"), use_container_width=True)

with col_barres:
    postes = {
        "Développement (alternante)":          COUT_DEV,
        "Prestataire (Marius)":                cout_presta_impute,
        "Infrastructure (poste de travail)":   COUT_INFRA,
        "Outils internes (Jira, Confluence…)": COUT_OUTILS,
        "API Claude Haiku":                    COUT_API,
        "Hébergement et données":              0,
    }
    noms    = list(postes.keys())
    valeurs = list(postes.values())
    fig_postes = go.Figure(go.Bar(
        x=valeurs, y=noms, orientation="h",
        marker_color=[BLUE, RED, BLUE, BLUE, AMBER, "#4b5563"],
        text=[f"{v:,.0f} €".replace(",", " ") for v in valeurs],
        textposition="outside", textfont=dict(color=TEXT, size=12),
    ))
    fig_postes.update_layout(
        xaxis=dict(range=[0, max(valeurs) * 1.35], gridcolor=BLUE_GRID),
        yaxis=dict(gridcolor=BG_PLOT, autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(mise_en_forme(fig_postes, "Coût par poste"), use_container_width=True)

st.caption(
    "Infrastructure et outils internes sont valorisés au tarif catalogue, conformément à la thèse, "
    "bien que mutualisés au niveau du groupe EDF et sans facturation additionnelle pour le projet. "
    "L'hébergement Streamlit Community Cloud et les données métier existantes restent à 0 €."
)
st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Évolution du coût par sprint
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">03 — Évolution du coût par sprint</p>', unsafe_allow_html=True)

col_cumul, col_sprint = st.columns(2)

with col_cumul:
    fig_cumul = go.Figure()
    fig_cumul.add_trace(go.Scatter(
        x=codes_sprint, y=cumul, name="Prévu",
        mode="lines+markers", line=dict(color=TEXT_MUTED, width=3, dash="dash"),
        marker=dict(size=9),
    ))
    fig_cumul.add_trace(go.Scatter(
        x=codes_sprint, y=cumul, name="Réel",
        mode="lines+markers", line=dict(color=GREEN, width=2),
        marker=dict(size=6, symbol="circle-open"),
        hovertemplate="%{x} — %{y:,.0f} €<extra></extra>",
    ))
    fig_cumul.update_layout(
        xaxis=dict(gridcolor=BLUE_GRID), yaxis=dict(gridcolor=BLUE_GRID, title="€ cumulés"),
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=11)),
    )
    st.plotly_chart(mise_en_forme(fig_cumul, "Coût cumulé — prévu vs réel"), use_container_width=True)
    st.caption("Les deux courbes sont confondues : le budget a été tenu à 100 % sur les quatre sprints.")

with col_sprint:
    fig_sprint = go.Figure(go.Bar(
        x=codes_sprint, y=par_sprint,
        marker_color=BLUE,
        text=[f"{v:,.0f} €".replace(",", " ") for v in par_sprint],
        textposition="outside", textfont=dict(color=TEXT, size=11),
    ))
    fig_sprint.update_layout(
        xaxis=dict(gridcolor=BG_PLOT),
        yaxis=dict(gridcolor=BLUE_GRID, range=[0, max(par_sprint) * 1.25], title="€"),
        showlegend=False,
    )
    st.plotly_chart(mise_en_forme(fig_sprint, "Coût par sprint"), use_container_width=True)
    st.caption("Coûts humains répartis au prorata de la durée de chaque sprint ; API imputée sur S3 et S4.")

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Écarts constatés et décisions d'ajustement
# ═══════════════════════════════════════════════════════════════════════════
TITRE_04 = "04 — Écarts et décisions d'ajustement"
st.markdown(f'<p class="section-title">{TITRE_04}</p>', unsafe_allow_html=True)

nb_ecarts   = len(ECARTS)
nb_resolus  = int((ECARTS["Statut"].isin(["Résolu", "Clos"])).sum())
impact_depense   = ECARTS["Impact dépense (€)"].sum()
impact_perimetre = ECARTS["Impact périmètre (€)"].sum()
impact_jour      = ECARTS["Impact délai (j)"].sum()

bloc_kpi(st.columns(5), [
    ("Écarts identifiés",       str(nb_ecarts),                                    ""),
    ("Traités",                 f"{nb_resolus} / {nb_ecarts}",                     ""),
    ("Dépense supplémentaire",  f"{impact_depense:+,.0f} €".replace(",", " "),     ""),
    ("Élargissement de périmètre", f"{impact_perimetre:+,.0f} €".replace(",", " "), "kpi-accent"),
    ("Impact sur le délai",     f"{impact_jour:+.0f} j",                           ""),
])
st.markdown("<br>", unsafe_allow_html=True)

COL_TYPE     = "Type d'écart"
COL_DECISION = "Décision d'ajustement"

lignes = []
for _, r in ECARTS.iterrows():
    cellules = [
        "<b>" + html.escape(str(r["ID"])) + "</b>",
        html.escape(str(r["Sprint"])),
        html.escape(str(r[COL_TYPE])),
        html.escape(str(r["Écart constaté"])),
        f"{r['Impact dépense (€)']:+.0f} €",
        f"{r['Impact périmètre (€)']:+.0f} €",
        f"{r['Impact délai (j)']:+.0f} j",
        html.escape(str(r[COL_DECISION])),
        html.escape(str(r["Arbitré par"])),
        html.escape(str(r["Statut"])),
    ]
    centre = {4, 5, 6}
    lignes.append(
        "<tr>"
        + "".join(
            f"<td style='text-align:center'>{c}</td>" if i in centre else f"<td>{c}</td>"
            for i, c in enumerate(cellules)
        )
        + "</tr>"
    )
lignes_html = "".join(lignes)
st.markdown(f"""
<table class="feat-table">
  <thead><tr>
    <th>ID</th><th>Sprint</th><th>Type</th><th>Écart constaté</th>
    <th style="text-align:center">Dépense</th><th style="text-align:center">Périmètre</th>
    <th style="text-align:center">Délai</th>
    <th>Décision d'ajustement</th><th>Arbitré par</th><th>Statut</th>
  </tr></thead>
  <tbody>{lignes_html}</tbody>
</table>""", unsafe_allow_html=True)

st.caption(
    "Aucun écart n'a entraîné de dépense supplémentaire ni de retard sur le calendrier global. "
    "L'élargissement de périmètre de +4 200 € (E6) correspond à la réintégration de la ressource "
    "prestataire dans le coût complet, et non à un dépassement de budget."
)
st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Charge par intervenant (indicateurs individuels)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">05 — Charge par intervenant</p>', unsafe_allow_html=True)

charge_tri   = CHARGE.sort_values("Charge estimée (h)", ascending=True)
total_heures = CHARGE["Charge estimée (h)"].sum()
couleurs     = [RED if c == "Externe" else BLUE for c in charge_tri["Catégorie"]]

col_charge, col_detail = st.columns([3, 2])

with col_charge:
    fig_charge = go.Figure(go.Bar(
        x=charge_tri["Charge estimée (h)"], y=charge_tri["Intervenant"],
        orientation="h", marker_color=couleurs,
        text=[f"{h:.0f} h · {h / total_heures * 100:.0f} %" for h in charge_tri["Charge estimée (h)"]],
        textposition="outside", textfont=dict(color=TEXT, size=11),
    ))
    fig_charge.update_layout(
        xaxis=dict(range=[0, charge_tri["Charge estimée (h)"].max() * 1.35], gridcolor=BLUE_GRID,
                   title="heures estimées"),
        yaxis=dict(gridcolor=BG_PLOT),
        showlegend=False,
    )
    st.plotly_chart(mise_en_forme(fig_charge, f"Charge estimée par intervenant — {total_heures:.0f} h au total",
                                  hauteur=300), use_container_width=True)

with col_detail:
    part_lena = CHARGE.loc[CHARGE["Intervenant"] == "Léna Pillet", "Charge estimée (h)"].iloc[0] / total_heures
    st.markdown(f"""
    <div class="kpi-card" style="text-align:left; padding:1.2rem 1.4rem;">
      <div class="kpi-label">Indicateur individuel</div>
      <p style="color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.6; margin:0.6rem 0 0 0;">
        Ma part dans la charge projet est de <b>{part_lena * 100:.0f} %</b> des {total_heures:.0f} heures
        mobilisées. La thèse annonçait 80 % : l'écart s'explique par la prise en compte de la charge
        réelle du prestataire, jusque-là non comptabilisée — même correction de périmètre que sur
        le budget.
      </p>
      <p style="color:{TEXT_MUTED}; font-size:0.86rem; line-height:1.6; margin:0.8rem 0 0 0;">
        Bleu : ressources internes · Rouge : ressource externe.
      </p>
    </div>""", unsafe_allow_html=True)

with st.expander("Détail des bases de charge et des sources"):
    st.dataframe(CHARGE, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — Retour sur investissement
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">06 — Retour sur investissement</p>', unsafe_allow_html=True)

bloc_kpi(st.columns(4), [
    ("Gain par analyse",       f"{GAIN_ANALYSE:.0f} €",       ""),
    ("Coût complet du projet", f"{cout_total:,.0f} €".replace(",", " "), ""),
    ("Seuil de rentabilité",   f"{seuil} analyses",           "kpi-accent"),
    ("Délai d'amortissement",  f"{mois_rentab:.1f} mois",     ""),
])
st.markdown("<br>", unsafe_allow_html=True)

x_max     = max(seuil * 2, 100)
abscisses = list(range(0, x_max + 1, max(1, x_max // 40)))
gains     = [n * GAIN_ANALYSE for n in abscisses]

fig_roi = go.Figure()
fig_roi.add_trace(go.Scatter(
    x=abscisses, y=gains, name="Gain cumulé",
    mode="lines", line=dict(color=GREEN, width=3),
    hovertemplate="%{x} analyses — %{y:,.0f} €<extra></extra>",
))
fig_roi.add_trace(go.Scatter(
    x=abscisses, y=[cout_total] * len(abscisses), name="Coût complet du projet",
    mode="lines", line=dict(color=RED, width=2, dash="dash"),
    hoverinfo="skip",
))
fig_roi.add_trace(go.Scatter(
    x=[seuil], y=[cout_total], name="Seuil de rentabilité",
    mode="markers+text", marker=dict(color=AMBER, size=13, symbol="diamond"),
    text=[f"  {seuil} analyses"], textposition="top right",
    textfont=dict(color=AMBER, size=12),
    hovertemplate="Seuil : %{x} analyses<extra></extra>",
))
fig_roi.update_layout(
    xaxis=dict(gridcolor=BLUE_GRID, title="nombre d'analyses réalisées"),
    yaxis=dict(gridcolor=BLUE_GRID, title="€"),
    legend=dict(orientation="h", y=1.14, x=0, font=dict(size=11)),
)
st.plotly_chart(mise_en_forme(fig_roi, "Gain cumulé et seuil de rentabilité", hauteur=360),
                use_container_width=True)

st.caption(
    f"À la cadence de {analyses_mois} analyses par mois, le projet est amorti en "
    f"{mois_rentab:.1f} mois. Les curseurs de simulation, dans le volet de gauche, permettent de "
    f"tester d'autres hypothèses de quote-part du prestataire et de cadence d'utilisation."
)


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cc-footer">
  Source : TDB_Pilotage_financier_CircuitCle.xlsx · Coûts humains répartis au prorata de la durée
  des sprints · API imputée sur S3 et S4 · Maintenance (1 000 €/an) exclue, car coût d'exploitation
  après projet · Quote-part du prestataire validée par le tuteur<br>
  CircuitClé · EDF DIPDE · Léna Pillet · 2026
</div>""", unsafe_allow_html=True)
