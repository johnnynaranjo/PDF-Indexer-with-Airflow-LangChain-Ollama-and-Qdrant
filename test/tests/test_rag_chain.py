from streamlit_app.Main import generate_response_with_context

def test_generate_response_stream():
    context = ["LangChain es una librería útil.", "Ollama provee LLMs."]
    output = "".join(generate_response_with_context("llama3", context, "¿Qué es LangChain?", temp=0.1))
    assert "LangChain" in output or "No sé" in output
