# app.py — Clasificador de Tópicos Flexible (Zero‑Shot) con Streamlit y BART-MNLI
# Autor: Tu Nombre
# Descripción: App bonita y práctica para clasificar texto en categorías no vistas (zero‑shot)
# -----------------------------------------------------------------------------------------
# ▶ Requisitos (instalar en tu entorno):
#   pip install streamlit transformers torch accelerate huggingface_hub
# ▶ Ejecutar:
#   streamlit run app.py

import os
import platform
from typing import List

import streamlit as st
from transformers import pipeline

# ==========================
# 🎨 Estilos sutiles (CSS)
# ==========================
CSS = """
<style>
:root { --radius: 16px; }
.block-container { padding-top: 2rem; }
.card { background: #ffffff; border-radius: var(--radius); padding: 1.25rem 1.25rem; box-shadow: 0 8px 30px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.06); }
.badge { display: inline-block; padding: .25rem .6rem; border-radius: 999px; font-size: .8rem; background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe; }
.small { color: #6b7280; font-size: .9rem; }
footer { visibility: hidden; }
</style>
"""

st.set_page_config(page_title="Zero‑Shot Clasificador (BART-MNLI)", page_icon="🧠", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# ======================================
# ⚙️ Sidebar: Configuración de la app
# ======================================
st.sidebar.title("⚙️ Configuración")

st.sidebar.markdown(
    "Esta app usa **Zero‑Shot Text Classification** (NLI) con el modelo\n"
    "`facebook/bart-large-mnli`. No requiere re‑entrenamiento."
)

# Token opcional de Hugging Face (por si el repo necesita autenticación o quieres más velocidad)
hf_token = st.sidebar.text_input("Hugging Face Token (opcional)", type="password", help="Si lo agregas, lo uso como HF_TOKEN para la descarga del modelo.")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

model_name = st.sidebar.selectbox(
    "Modelo",
    options=["facebook/bart-large-mnli"],
    index=0,
    help="BART‑MNLI es ideal para Zero‑Shot por su entrenamiento en inferencia."
)

use_gpu = st.sidebar.checkbox("Usar GPU si está disponible", value=True)

st.sidebar.caption(
    f"🖥️ Sistema: {platform.system()} • Python: {platform.python_version()}"
)

# =====================================================
# 🧠 Carga eficiente del modelo (cacheado una sola vez)
# =====================================================
@st.cache_resource(show_spinner=True)
def load_pipeline(model_id: str, prefer_gpu: bool = True):
    """Carga el pipeline de zero‑shot con caché de recursos."""
    device_map = "auto" if prefer_gpu else None
    try:
        clf = pipeline(
            "zero-shot-classification",
            model=model_id,
            device_map=device_map,  # usa GPU si está
        )
        return clf
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
        st.stop()

clf = load_pipeline(model_name, use_gpu)

# ======================================
# 🧩 Encabezado de la aplicación
# ======================================
st.title("🧠 Clasificador de Tópicos Zero‑Shot")
st.markdown(
    "Convierte tu texto en **probabilidades** para etiquetas que *ni siquiera vio en el entrenamiento*.\n"
    "Basado en **NLI**: el modelo evalúa si el texto **implica** hipótesis del tipo ‘Este texto trata sobre [etiqueta]’."
)

# ======================================
# 📝 Entradas del usuario
# ======================================
with st.container():
    col1, col2 = st.columns([2,1])
    with col1:
        text = st.text_area(
            "Texto a analizar",
            value="Lionel Messi ganó el Balón de Oro y lideró a su equipo a la victoria.",
            height=160,
            placeholder="Escribe aquí el texto…",
        )
    with col2:
        labels_csv = st.text_input(
            "Categorías (separadas por comas)",
            value="deportes, política, economía, entretenimiento, tecnología",
            help="Ej.: deportes, política, economía"
        )
        multi_label = st.toggle(
            "Permitir múltiples etiquetas verdaderas (multi‑label)",
            value=True,
            help="Si está activo, el modelo asigna probabilidades de forma independiente a cada etiqueta."
        )
        hypothesis_template = st.text_input(
            "Plantilla de hipótesis",
            value="Este texto trata sobre {}.",
            help="Cómo se enuncia la hipótesis para cada etiqueta."
        )

# Sanitizar etiquetas
def parse_labels(s: str) -> List[str]:
    labels = [x.strip() for x in s.split(",") if x.strip()]
    # Eliminar duplicados preservando orden
    seen = set()
    uniq = []
    for x in labels:
        if x.lower() not in seen:
            uniq.append(x)
            seen.add(x.lower())
    return uniq

candidate_labels = parse_labels(labels_csv)

# ======================================
# ▶ Botón de ejecución
# ======================================
run = st.button("🚀 Clasificar", type="primary")

if run:
    if not text.strip():
        st.warning("Por favor escribe un texto para analizar.")
        st.stop()
    if not candidate_labels:
        st.warning("Agrega al menos una etiqueta.")
        st.stop()

    with st.spinner("Clasificando con BART‑MNLI…"):
        result = clf(
            sequences=text,
            candidate_labels=candidate_labels,
            multi_label=multi_label,
            hypothesis_template=hypothesis_template,
        )

    # ‘result’ puede ser dict o lista dependiendo del batch; aquí es dict
    labels = result["labels"]
    scores = result["scores"]

    # Ordenar descendentemente
    pairs = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_scores = zip(*pairs)

    st.subheader("Resultados")
    st.markdown("Probabilidad por etiqueta (cuanto mayor, más afinidad):")

    # Vista en tarjetas
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for lbl, sc in pairs:
            st.markdown(
                f"<span class='badge'>{lbl}</span> "
                f"<span class='small'>prob = {sc:.4f}</span>",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Gráfico de barras
    st.bar_chart({"Etiqueta": list(sorted_labels), "Probabilidad": list(sorted_scores)}, x="Etiqueta", y="Probabilidad")

    # Tabla cruda
    st.dataframe({"Etiqueta": list(sorted_labels), "Probabilidad": [round(s, 6) for s in sorted_scores]}, use_container_width=True)

# ======================================
# 📎 Tips y notas
# ======================================
with st.expander("Notas y buenas prácticas"):
    st.markdown(
        "- **Cacheo**: Usamos `@st.cache_resource` para que el modelo se cargue una sola vez.\n"
        "- **Etiquetas**: Mantén las etiquetas *cortas y claras* para mejores resultados.\n"
        "- **Multi‑label**: Actívalo si tu texto puede pertenecer a varias categorías a la vez.\n"
        "- **Rendimiento**: Si tienes GPU, deja marcada la opción para acelerar la inferencia.\n"
        "- **Privacidad**: Si agregas un token de Hugging Face, solo se usa para descargar el modelo."
    )
