from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import json
import hashlib
import requests
import logging

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, PointsSelector, FieldCondition, MatchValue

# Rutas
BASE_FOLDER = "/opt/airflow/user_data"
WATCH_FOLDER = os.path.join(BASE_FOLDER, "incoming")
PROCESSED_FOLDER = os.path.join(BASE_FOLDER, "processed")
INDEX_LOG = os.path.join(BASE_FOLDER, "indexed_files.json")

# Configuración de servicios
QDRANT_URL = 'http://qdrant:6333'
OLLAMA_URL = 'http://ollama:11434'
COLLECTION_NAME = 'airflow_ingestion'
EMBEDDING_MODEL_NAME = 'nomic-embed-text' # Cambia esto al modelo que estés usando
EMBEDDING_MODEL_SIZE = 2048  # Tamaño del modelo de embedding, ajusta según tu modelo

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 6, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

logger = logging.getLogger("airflow.task")

# ------------------------ Funciones auxiliares ------------------------

def check_service(url, name):
    try:
        logger.info(f"Verificando {name} en {url}")
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            logger.info(f"✅ {name} disponible.")
            return True
    except Exception as e:
        logger.error(f"❌ Error conectando con {name}: {e}")
    return False

def compute_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def load_index_log():
    if os.path.exists(INDEX_LOG):
        with open(INDEX_LOG, 'r') as f:
            return json.load(f)
    return {}

def save_index_log(log):
    with open(INDEX_LOG, 'w') as f:
        json.dump(log, f, indent=2)

def find_unindexed_pdfs():
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    index_log = load_index_log()
    unindexed = []

    for filename in os.listdir(WATCH_FOLDER):
        if not filename.endswith('.pdf'):
            continue

        full_path = os.path.join(WATCH_FOLDER, filename)
        file_hash = compute_file_hash(full_path)
        file_mtime = os.path.getmtime(full_path)
        file_mtime_iso = datetime.fromtimestamp(file_mtime).isoformat()

        if (
            filename not in index_log or
            index_log[filename]["hash"] != file_hash
        ):
            unindexed.append((filename, full_path, file_hash, file_mtime_iso))

    return unindexed

# ------------------------ Tarea principal ------------------------

def process_and_index():
    if not check_service(QDRANT_URL + "/collections", "Qdrant"):
        raise Exception("Qdrant no disponible.")
    if not check_service(OLLAMA_URL + "/api/tags", "Ollama"):
        raise Exception("Ollama no disponible.")

    unindexed_files = find_unindexed_pdfs()
    if not unindexed_files:
        logger.info("No hay archivos nuevos o modificados para procesar.")
        return

    index_log = load_index_log()
    all_documents = []

    qdrant = QdrantClient(url=QDRANT_URL)
    
    # Solo crea la colección si no existe
    if not qdrant.collection_exists(COLLECTION_NAME):
        logger.info(f"Creando {COLLECTION_NAME} en Qdrant.")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_MODEL_SIZE, distance=Distance.COSINE),
            # RetrievalMode=RetrievalMode.HYBRID,
            sparse_vectors_config={"bm25": {}}
        )
    else:
        logger.info(f"Coleccion {COLLECTION_NAME} ya existe en Qdrant.")

    for filename, file_path, file_hash, file_mtime in unindexed_files:
        logger.info(f"📄 Procesando: {filename}")

        # 🔥 Eliminar documentos anteriores del mismo archivo en Qdrant
        logger.info(f"🧹 Eliminando chunks anteriores de {filename} en Qdrant...")
        try:
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=PointsSelector.filter(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="source",
                                match=MatchValue(value=filename)
                            )
                        ]
                    )
                )
            )
            logger.info(f"🗑️  Eliminación completada para {filename}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo eliminar {filename}: {e}")

        try:
            loader = PDFPlumberLoader(file_path)
            docs = loader.load()
            if docs:
                for doc in docs:
                    doc.metadata["source"] = filename  # necesario para la eliminación posterior
                all_documents.extend(docs)
                index_log[filename] = {
                    "hash": file_hash,
                    "last_modified": file_mtime
                }
            else:
                logger.warning(f"{filename} está vacío o no tiene texto válido.")
        except Exception as e:
            logger.error(f"Error procesando {filename}: {e}")

    if not all_documents:
        logger.warning("Ningún documento válido encontrado.")
        return

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    if not embeddings:
        raise Exception("No se pudo inicializar OllamaEmbeddings.")

    splitter = SemanticChunker(embeddings)
    logger.info("✅ 1/3 Chunking completado")

    chunks = splitter.split_documents(all_documents)
    logger.info("✅ 2/3 División completada")

    sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")

    QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        sparse_embedding=sparse_model,
        location=QDRANT_URL,
        prefer_grpc=True,
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    logger.info("✅ 3/3 Índice vectorial creado")

    save_index_log(index_log)

    for filename, file_path, _, _ in unindexed_files:
        dest = os.path.join(PROCESSED_FOLDER, filename)
        os.rename(file_path, dest)
        logger.info(f"✅ Procesado y movido: {dest}")

# ------------------------ DAG ------------------------

with DAG(
    'semantic_pdf_chunking_dag',
    description='Procesa PDFs desde carpeta local y los indexa con LangChain, Ollama y Qdrant',
    default_args=default_args,
    # schedule_interval=None,
    catchup=False,
    tags=['ollama', 'langchain', 'qdrant', 'semantic', 'dedup'],
) as dag:

    ingest = PythonOperator(
        task_id='process_and_index_pdfs',
        python_callable=process_and_index
    )

    ingest
