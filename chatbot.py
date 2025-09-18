# chat_groq_app.py — Chatbot Conversacional con Memoria (Stateful) usando Groq
# ---------------------------------------------------------------------------------
# Requisitos mínimos (añádelos a requirements.txt):
#   streamlit
#   groq
#
# Ejecución local:
#   streamlit run chat_groq_app.py
#
# Clave API:
#   • El usuario la ingresa MANUALMENTE en la barra lateral (no se hardcodea).
#   • (Opcional) Si existe st.secrets["GROQ_API_KEY"], se usa como respaldo.

from typing import List, Dict
import os
import streamlit as st
from groq import Groq

st.set_page_config(page_title="Chat con Memoria — Groq Llama3-8B", page_icon="💬", layout="centered")

# ==========================
# 🎨 Estilos mínimos
# ==========================
st.markdown(
    """
    <style>
    .small { color:#6b7280; font-size:.9rem }
    .dim { opacity:.8 }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================
# ⚙️ Sidebar: clave y opciones
# ======================================
st.sidebar.header("⚙️ Configuración")

# 1) Entrada manual (preferida)
manual_key = st.sidebar.text_input(
    "GROQ_API_KEY (manual)", type="password",
    help="Pega aquí tu clave de Groq. No se guardará en el código.")

# 2) Respaldo vía secrets (opcional)
secrets_key = None
try:
    secrets_key = st.secrets.get("GROQ_API_KEY", None)  # requiere .streamlit/secrets.toml
except Exception:
    secrets_key = None

api_key = manual_key or secrets_key

model_name = st.sidebar.selectbox(
    "Modelo Groq",
    ["llama3-8b-8192"],
    index=0,
    help="Modelo recomendado para chats rápidos."
)

system_prompt = st.sidebar.text_area(
    "Instrucción del sistema (opcional)",
    value="Eres un asistente claro, conciso y útil. Responde en español.",
    height=100
)

st.sidebar.caption("Usa la entrada manual si no quieres gestionar secrets.")

# ======================================
# 🧠 Estado de la sesión (memoria)
# ======================================
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, str]] = []

# ======================================
# 🧩 Encabezado
# ======================================
st.title("💬 Chatbot con Memoria (Groq)")
st.markdown(
    "Este demo mantiene un **historial de conversación** en `st.session_state` y lo envía en cada turno al **Chat Completions API** de Groq."
)

# ======================================
# 🧷 Funciones auxiliares
# ======================================
@st.cache_resource(show_spinner=False)
def get_client(key: str):
    if not key:
        return None
    return Groq(api_key=key)


def build_messages(history: List[Dict[str, str]], system_msg: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_msg and system_msg.strip():
        messages.append({"role": "system", "content": system_msg.strip()})
    messages.extend(history)
    return messages

# ======================================
# 🗨️ Render del historial (UI)
# ======================================
for msg in st.session_state.history:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# ======================================
# ⌨️ Entrada del usuario
# ======================================
user_input = st.chat_input("Escribe tu mensaje…")

if user_input is not None:
    if not api_key:
        st.warning("🔐 Ingresa tu GROQ_API_KEY en la barra lateral para continuar.")
        st.stop()

    # Añadimos el mensaje del usuario al historial y lo mostramos
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Cliente Groq
    client = get_client(api_key)
    if client is None:
        st.error("No fue posible inicializar el cliente de Groq. Verifica tu clave.")
        st.stop()

    # Construimos el payload con todo el historial + system
    messages = build_messages(st.session_state.history, system_prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Pensando…"):
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.6,
                    max_tokens=512,
                    stream=False,
                )
                reply = completion.choices[0].message.content if completion.choices else "(Sin respuesta)"
            except Exception as e:
                reply = f"Ocurrió un error llamando a Groq: {e}"
            st.markdown(reply)

    # Guardamos la respuesta del asistente en el historial
    st.session_state.history.append({"role": "assistant", "content": reply})

# ======================================
# 🧽 Controles de sesión
# ======================================
with st.expander("Opciones de sesión"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Limpiar historial"):
            st.session_state.history = []
            st.experimental_rerun()
    with col2:
        st.markdown("<span class='small dim'>El historial vive solo en la sesión de tu navegador.</span>", unsafe_allow_html=True)
