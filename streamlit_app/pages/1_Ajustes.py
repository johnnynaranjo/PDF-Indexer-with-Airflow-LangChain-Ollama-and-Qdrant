# Utils
from utils import ollama_check_model, qdrant_check_db, qdrant_create_db, qdrant_delete_db, ollama_pull_model, ollama_delete_model
# App and models
import streamlit as st

# ----------------------------- MAIN -----------------------------

def main():

    st.set_page_config(
        page_title="Ajustes de la aplicación",
        page_icon="⚙️",
    )
    st.title("⚙️ Ajustes de la aplicación")

    # Verificar si hay modelos cargados en Ollama
    ollama_check_model("http://ollama:11434", "Ollama")

    # Opción para descargar nuevo modelo
    with st.form("llm_form"):
        model_to_pull = st.text_input("Nombre del modelo a descargar (ej. `llama3.2` o `mxbai-embed-large`):")
        submit_pull = st.form_submit_button("Descargar modelo")
        if submit_pull and model_to_pull:
            ollama_pull_model(model_to_pull)
 
    # Verificar si hay modelos cargados en Qdrant
    client = qdrant_check_db("http://qdrant:6333", "Qdrant")

    if client is not None:
        # Opción para crear nueva colección
        with st.form("vector_form"):
            db_to_create = st.text_input("Nombre de la nueva colección:")
            submit_pull = st.form_submit_button("Crear colección")
            if submit_pull and db_to_create:
                qdrant_create_db(
                    db_name=db_to_create,
                    embedding_size=st.session_state.embedding_dim,
                    client=client
                )
    else:
        st.error("No se pudo conectar con Qdrant. No se puede crear colección.")

    # Eliminar modelos
    with st.expander("🗑️ Eliminar modelos de Ollama"):
        ollama_delete_model("http://ollama:11434", "Ollama")

    # Eliminar colecciones
    with st.expander("🗑️ Eliminar colecciones de Qdrant"):
        qdrant_delete_db("http://qdrant:6333", "Qdrant")

if __name__ == "__main__":
    main()