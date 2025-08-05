# 🧠 Semantic PDF Indexer with Airflow, LangChain, Ollama and Qdrant

Este proyecto es una pipeline de indexación semántica de documentos PDF, gestionada mediante **Apache Airflow** y utilizando componentes modernos de IA como:

- 🦜 **LangChain** para el procesamiento e incrustación de textos
- 🤖 **Ollama** como motor de embeddings y LLM
- 📦 **Qdrant** como base de datos vectorial para recuperación semántica
- 🗂 **Airflow DAG** para orquestar la ingesta y procesamiento de documentos

Contiene una aplicación web construida con **Streamlit** que funciona como una interfaz gráfica (GUI) para interactuar con un modelo de lenguaje (LLM). La aplicación corre dentro de un contenedor Docker, junto con otros dos servicios: **Ollama** (para servir el modelo) y **Qdrant** (para almacenamiento vectorial y funcionalidades RAG).

---

## 📁 Estructura del proyecto
```graphql
.
├── dags/
│ └── semantic_pdf_chunking_dag.py
├── streamlit_app/
│ ├── pages/
│ │ ├── 1_Ajustes.py
│ │ ├── 2_Chat.py
│ │ ├── 3_RAG.py
│ │ ├── 4_Cargar_PDF.py
│ │ ├── 4_Cargar_PDF.py
│ │ └── 6_Modelos_disponibles.py
│ ├── Main.py
│ ├── Dockerfile
│ └── requirements.txt
├── .env
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```
---

## ⚙️ Servicios y tecnologías

- **Airflow Webserver + Scheduler**: Orquesta la ejecución de DAGs.
- **PostgreSQL**: Backend de metadatos para Airflow.
- **Streamlit App** - Interfaz gráfica del usuario.
- **Ollama**: Servidor de modelos LLM. Provee embeddings a partir de los textos de los PDFs.
- **Qdrant**: Base de datos vectorial para funcionalidades RAG. Almacena los vectores híbridos (dense + sparse).
- **Volumen compartido**: Permite que los contenedores lean archivos desde una carpeta compartida del host.

---

## 📄 Requisitos

- Docker / Podman
- Docker Compose v1.28+ / Podman Compose
- Python 3.9+ si deseas ejecutar manualmente el DAG
- Crear un archivo `.env` con esta variable, ver ejemplo:

```
USER_DESKTOP_PATH=/ruta/completa/a/tu/carpeta/pdf
```

---

## 🚀 Iniciar el proyecto

Renombra el archivo .env.example y modifica la ruta a tu carpeta de archivos pdf a indexar.

Con Docker Compose:

1. Construye las imagenes
```bash
docker compose --build
```
2. Lanza el contenedor **airflow-init** 
```bash
docker compose up airflow-init
```
3. Inicia todos los servicios 
```bash
docker compose up
```
Accede a la interfaz web de Airflow:

- 📍 http://localhost:8080

- Usuario: admin

- Contraseña: admin

---

## 🛠️ Descripción del DAG

El DAG semantic_pdf_chunking_dag:

1. Busca PDFs nuevos en la carpeta incoming.

2. Verifica que no hayan sido procesados antes (por hash).

3. Extrae su texto usando PDFPlumber.

4. Divide los textos en chunks semánticos.

5. Genera embeddings (denso y sparse).

6. Indexa los chunks en Qdrant.

7. Mueve el archivo PDF a processed.

---

## 📂 Estructura esperada dentro del volumen compartido:
```graphql
user_data/
├── incoming/      # Aquí colocas los PDFs nuevos
├── processed/     # Airflow mueve aquí los PDFs procesados
└── indexed_files.json  # Control de duplicados por hash
```
---

## ✅ Ejecutar el DAG manualmente

1. Abre Airflow en el navegador.

2. Habilita y ejecuta el DAG ```semantic_pdf_chunking_dag```.

3. Los logs te mostrarán el progreso y errores.

---

## 📦 Reinstalar dependencias

Si modificas requirements.txt, reconstruye los servicios:
```bash
docker compose up --build
```
---

## 🧪 Verificación

Para confirmar que los servicios están funcionando:
```bash
curl http://localhost:6333/collections   # Qdrant
curl http://localhost:11434/api/tags     # Ollama
```
---

## 🧹 Problemas comunes

- El DAG no aparece en la UI: Asegúrate de que esté dentro de airflow/dags/ y que el contenedor haya sido reiniciado.

- Archivos PDF no se procesan: Actualiza la ruta al volumen compatido en el archivo .env antes de iniciar los contenedores. Confirma que están en la carpeta incoming y que no hayan sido procesados antes.

- Error de location en QdrantVectorStore: Cambia el argumento por url=.

---

## 📜 Licencia

Este proyecto está bajo la licencia MIT.
✨ Créditos

Este pipeline usa herramientas de código abierto de LangChain, Qdrant, y Ollama.

---
