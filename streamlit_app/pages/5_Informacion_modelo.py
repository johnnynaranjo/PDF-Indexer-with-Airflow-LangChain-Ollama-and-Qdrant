# Utils
from utils import ollama_model_info
# App and models
import streamlit as st
import ollama

OLLAMA_URL = "http://ollama:11434"
OLLAMA_CONTAINER = "Ollama"
QDRANT_URL = "http://qdrant:6333"
QDRANT_CONTAINER = "Qdrant"

# ----------------------------- MAIN -----------------------------

def main():
   
    st.set_page_config(
        page_title="Información del Modelo de Ollama",
        page_icon="ℹ️",
    )
    st.title("ℹ️ Información del Modelo de Ollama")

    st.selectbox("Please select the model:", [model["model"] for model in ollama.list()["models"]], key = "selected_model")
    if st.button("Obtener información"):
        info = ollama_model_info(OLLAMA_URL, st.session_state.selected_model)
        st.json(info)

if __name__ == "__main__":
    main()
