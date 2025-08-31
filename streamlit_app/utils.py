import streamlit as st
import requests
import ollama
import tempfile
import os
# generators and typing
from typing import Dict, Generator, List, Tuple
# load data
from langchain_community.document_loaders import PDFPlumberLoader
# import files to vector store
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
# db split documents into chunks
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
# db save chunks to vector store
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
# RAG
from collections.abc import Iterator
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------- CONEXIONES Y CHEQUEOS -----------------------------

def check_connection(url, container_name: str):
    session_key = f"{container_name}_connection_ok"
    if st.session_state.get(session_key, False):
        return True
    try:
        response = requests.get(url, timeout=(1, 2))
        if response.status_code == 200:
            st.success(f"✅ Conexión exitosa con {container_name}.")
            st.session_state[session_key] = True
            return True
        else:
            st.error(f"⚠️ El contenedor de {container_name} respondió con código {response.status_code}.")
            return False
    except requests.exceptions.ConnectTimeout:
        st.error(f"⏱️ Tiempo de espera agotado al conectar con {container_name}.")
        return False
    except requests.exceptions.ReadTimeout:
        st.error(f"⏱️ Tiempo de espera agotado esperando respuesta de {container_name}.")
        return False
    except requests.exceptions.ConnectionError:
        st.error(f"❌ No se pudo establecer conexión con {container_name}. Asegúrate de que esté en ejecución.")
        return False
    except Exception as e:
        st.error(f"⚠️ Error inesperado al conectar con {container_name}: {e}")
        return False

def ollama_check_model(url, container_name: str):
    if not check_connection(url, container_name):
        return
    try:
        modelos = ollama.list()["models"]
        if modelos:
            st.sidebar.selectbox("Modelos disponibles:", [model["model"] for model in modelos], key="selected_model")
        else:
            st.warning("No hay modelo disponible. Descarga uno.")
    except Exception as e:
        st.error(f"Error al obtener modelos de Ollama: {e}")

def qdrant_check_db(url, container_name: str):
    if not check_connection(url, container_name):
        return None
    try:
        client = QdrantClient(url=url)
        collections = client.get_collections().collections
        existing_collections = [col.name for col in collections]
        if existing_collections:
            st.sidebar.selectbox("Colecciones disponibles:", existing_collections, key="selected_db")
        else:
            st.warning("No hay colecciones existentes. Crea una nueva.")
        return client
    except Exception as e:
        st.error(f"Error al conectar con Qdrant: {e}")
        return None

# ----------------------------- FUNCIONES DE PDF -----------------------------

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    loader = PDFPlumberLoader(tmp_path)
    return loader.load()

def load_pdfs_from_folder(folder_path):
    all_documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                loader = PDFPlumberLoader(file_path)
                all_documents.extend(loader.load())
    return all_documents

# ----------------------------- MODELOS Y COLECCIONES -----------------------------
def ollama_pull_model(model_name: str):
    try:
        with st.spinner(f"Descargando modelo '{model_name}'...", show_time=True):
            ollama.pull(model_name)
        st.success(f"Modelo '{model_name}' descargado con éxito.")
    except Exception as e:
        st.error(f"No se pudo descargar el modelo '{model_name}': {e}")

def ollama_delete_model(url, container_name: str):
    """
    Elimina un modelo de Ollama.
    Devuelve True si se eliminó correctamente, False en caso contrario.
    """
    if not check_connection(url, container_name):
        return
    try:
        modelos = ollama.list()["models"]
        if modelos:
            selected_model = st.selectbox("Selecciona el modelo a eliminar:", [model["model"] for model in modelos], key="delete_model")
            if st.button("Eliminar modelo", key="delete_model_button"):
                ollama.delete(model=selected_model)
                st.success(f"Modelo '{selected_model}' eliminado con éxito.")
        else:
            st.info("No hay modelos disponibles para eliminar.")
    except Exception as e:
        st.error(f"Error al eliminar modelo: {e}")

def ollama_model_info(url, model_name):
    try:
        res = requests.post(f"{url}/api/show", json={"name": model_name})
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"No se pudo obtener la información: {e}"}

def qdrant_create_db(db_name, embedding_size, client):

    # Crea la coleccion en Qdrant
    client.create_collection(
        collection_name=db_name,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        sparse_vectors_config={"bm25": {}}
    )
    # Verifica que se ha creado la colección
    if client.collection_exists(db_name):
        st.success(f"Colección '{db_name}' creada correctamente.")
    else:
        st.error(f"No se pudo crear la colección '{db_name}'.")

def qdrant_delete_db(url, container_name: str):
    if not check_connection(url, container_name):
        return

    try:
        client = QdrantClient(url=url)
        collections = client.get_collections().collections
        existing_collections = [col.name for col in collections]

        if existing_collections:
            selected_collection = st.selectbox("Selecciona la colección a eliminar:", existing_collections, key="delete_collection")
            if st.button("Eliminar colección", key="delete_collection_button"):
                client.delete_collection(collection_name=selected_collection)
                st.success(f"Colección '{selected_collection}' eliminada con éxito.")
        else:
            st.info("No hay colecciones disponibles para eliminar.")
    except Exception as e:
        st.error(f"Error al eliminar colección: {e}")

# ----------------------------- VECTOR STORE -----------------------------

def qdrant_create_vector_index(url, container_name, embedding_model_name, embedding_size, collection_name, documents):
    # verifica si el contenedor de Qdrant está en ejecución
    status = check_connection(url, container_name)

    # carga la url de la base de datos en el cliente
    client = QdrantClient(url=url)

    # Verifica si hay colecciones existentes
    if status == True:

        embeddings_model = OllamaEmbeddings(model=embedding_model_name, embedding_size=embedding_size)

        # seleccionar el modelo y hacer el pull si no existe
        with st.spinner("Semantic Chunker", show_time=True):
            text_splitter = SemanticChunker(embeddings_model)
            st.success("1/3 Chunking completado")
        
        with st.spinner("Dividiendo documento", show_time=True):
            document = text_splitter.split_documents(documents)
            st.success("2/3 División completada")
        
        with st.spinner("Creando índice vectorial", show_time=True):
            # definir el modelo sparse
            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

            # crear el vector store
            vector_store = QdrantVectorStore.from_documents(
                document,
                embedding=embeddings_model,
                sparse_embedding=sparse_embeddings,
                location=url,
                prefer_grpc=True,
                collection_name=collection_name,
                retrieval_mode=RetrievalMode.HYBRID,
                force_recreate=True,
            )
            st.success("3/3 Índice vectorial creado")

# ----------------------------- RECUPERACIÓN DE DOCUMENTOS -----------------------------

def retrieve_with_scores(client: QdrantClient, collection_name: str, query: str, embedding_model: str, embedding_size, top_k: int = 5) -> List[Tuple[str, float]]:
    dense_embeddings = OllamaEmbeddings(model=embedding_model, embedding_size=embedding_size)
    query_vector = dense_embeddings.embed_query(query)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
    return [(hit.payload.get("page_content", ""), hit.score) for hit in search_result]

# ----------------------------- GENERACIÓN RAG -----------------------------

def generate_response_with_context(model_name: str, context_docs: List[str], query: str, temp: float = 0.1) -> Iterator[str]:

    llm = OllamaLLM(model=model_name, temperature=temp)
    prompt = PromptTemplate.from_template(
        """Responde la siguiente pregunta usando el contexto proporcionado. 
        Si no puedes responder basándote únicamente en el contexto, responde "No sé".

        Contexto:
        {context}

        Pregunta:
        {question}
        """
    )

    chain: RunnableSequence = (
        (lambda _: {"context": "\n\n".join(context_docs), "question": query})
        | prompt
        | llm
        | StrOutputParser()
    )

    full_response = ""
    for chunk in chain.stream(query):
        full_response += chunk
        yield chunk

# ----------------------------- GENERACIÓN LLM -----------------------------

def ollama_generator(model_name: str, messages: Dict) -> Generator:
    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True
        )
    for chunk in stream:
        yield chunk['message']['content']

    stream = ollama.chat(
        model=model_name,
        messages=messages,
        stream=True
    )
    for chunk in stream:
        yield chunk['message']['content']