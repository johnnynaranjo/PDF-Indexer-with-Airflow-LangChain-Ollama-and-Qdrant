from streamlit_app.pages.utils import generate_response_with_context

COLLECTION_NAME = "test_collection"
QUERY = "¿Qué es LangChain?"
MODEL_NAME = "llama3"
EMBEDDING_SIZE = 768

def test_generate_response_stream():
    context = ["LangChain es una librería útil.", "Ollama provee LLMs."]
    output = "".join(generate_response_with_context(MODEL_NAME, context, QUERY, temp=0.1))
    assert "LangChain" in output or "No sé" in output
