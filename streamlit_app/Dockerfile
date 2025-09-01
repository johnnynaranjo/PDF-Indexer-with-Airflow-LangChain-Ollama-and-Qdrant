FROM python:latest

# Update & install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copiar todo lo que está en ./streamlit_app (contexto) a /app/streamlit_app
COPY . /app/streamlit_app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "streamlit_app/Main.py", "--server.port=8501", "--server.enableCORS=false"]
