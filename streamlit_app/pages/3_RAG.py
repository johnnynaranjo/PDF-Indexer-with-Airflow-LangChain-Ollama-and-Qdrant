# Utils
from utils import ollama_check_model, qdrant_check_db, retrieve_with_scores, generate_response_with_context
# App and models
import streamlit as st
import ollama

# ----------------------------- MAIN -----------------------------

def main():
    st.set_page_config(
        page_title="Chatea con RAG",
        page_icon="🤖",
    )
    st.title("🤖 Chatea con RAG")

    # Verificar si hay modelos cargados en Ollama
    ollama_check_model("http://ollama:11434", "Ollama")
    # y si hay colecciones en Qdrant
    client = qdrant_check_db("http://qdrant:6333", "Qdrant")
    
    # Mostrar parámetros solo si hay modelos disponibles
    st.sidebar.title('Ajustes')

    # Campo para seleccionar el tamaño del embedding
    embedding_dim = st.sidebar.selectbox(
        "Tamaño del embedding:",
        options=[384, 512, 768, 1024, 1536],
        index=2,  # Por defecto 768
        key="embedding_dim"
    )

    st.sidebar.selectbox(
        "Embeddings disponibles:",
        [model["model"] for model in ollama.list()["models"]],
        key="embedding_model"
    )

    # Número de respuestas similares a mostrar
    st.sidebar.slider("Número de opciones similares:", min_value=1, max_value=10, value=3, key= "top_k")
    # Temperatura
    st.sidebar.slider("Temperatura:", min_value=0.0, max_value=2.0, value=0.7, step=0.1, key="temp")

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Escribe tu consulta..."):
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.rag_messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            st.markdown("### 🔍 Documentos similares encontrados:")

            results = retrieve_with_scores(
                client=client,
                collection_name=st.session_state.selected_db,
                query=query,
                embedding_model=st.session_state.embedding_model,
                embedding_size=st.session_state.embedding_dim,
                top_k=st.session_state.top_k,
            )

            for i, (content, score) in enumerate(results):
                st.markdown(f"**Opción {i+1}** - Similitud: `{score:.2f}`")
                st.code(content, language="markdown")

            st.markdown("---")
            st.markdown("### 🤖 Generando respuesta...")

            response_stream = generate_response_with_context(
                model_name=st.session_state.selected_model,
                context_docs=[doc for doc, _ in results],
                query=query,
                temp=st.session_state.temp,
            )

            response_collected = ""
            for chunk in st.write_stream(response_stream):
                response_collected += chunk

        st.session_state.rag_messages.append({"role": "assistant", "content": response_collected})

if __name__ == "__main__":
    main()
