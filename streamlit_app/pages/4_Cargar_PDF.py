# Utils
from utils import ollama_check_model, qdrant_check_db, load_pdf, load_pdfs_from_folder, qdrant_create_vector_index
import os
# App and models
import streamlit as st

# ----------------------------- MAIN -----------------------------

def main():
    
    st.set_page_config(
        page_title="Cargar PDF",
        page_icon="📁",
    )
    st.title("📁 Cargar PDF")
    
    # Verificar si hay modelos cargados en Ollama
    ollama_check_model("http://ollama:11434", "Ollama")

    # Mostrar parámetros solo si hay modelos disponibles
    st.sidebar.title('Ajustes')
    
    # Campo para seleccionar el tamaño del embedding
    embedding_dim = st.sidebar.selectbox(
        "Tamaño del embedding:",
        options=[384, 512, 768, 1024, 1536],
        index=2,  # Por defecto 768
        key="embedding_dim"
    )

    # Opción para subir archivo o carpeta
    upload_option = st.sidebar.radio("Selecciona fuente de datos:", ["Archivo PDF", "Carpeta con PDFs"])

    doc = None

    if upload_option == "Archivo PDF":
        uploaded_file = st.file_uploader("Sube tu archivo PDF", type="pdf")
        if uploaded_file is not None:
            doc = load_pdf(uploaded_file)
            st.success("Archivo PDF cargado con éxito!")

    elif upload_option == "Carpeta con PDFs":
        folder_path = st.text_input("Ruta local a la carpeta con PDFs")
        if folder_path and os.path.isdir(folder_path):
            doc = load_pdfs_from_folder(folder_path)
            st.success(f"Se cargaron {len(doc)} documentos desde la carpeta.")
        elif folder_path:
            st.warning("La ruta proporcionada no es válida.")

    # Conexión con Qdrant
    client = qdrant_check_db("http://qdrant:6333", "Qdrant")

    if st.button("Crear índice vectorial"):
        if not doc:
            st.warning("Primero sube un archivo o selecciona una carpeta.")
        else:
            if client:
                qdrant_create_vector_index(
                    url="http://qdrant:6333",
                    container_name="Qdrant",
                    embedding_model_name=st.session_state.selected_model,
                    embedding_size=st.session_state.embedding_dim,
                    collection_name=st.session_state.selected_db,
                    documents=doc
                )
                st.success(f"Índice vectorial '{st.session_state.selected_db}' creado con éxito!")

if __name__ == "__main__":
    main()