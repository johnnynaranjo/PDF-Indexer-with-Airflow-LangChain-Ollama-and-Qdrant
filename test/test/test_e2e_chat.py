from streamlit_app.main import retrieve_with_scores, generate_response_with_context

def test_full_chat_pipeline():
    class MockClient:
        def search(self, *args, **kwargs):
            return [type("Hit", (), {"payload": {"page_content": "contenido"}, "score": 0.9})]

    client = MockClient()
    results = retrieve_with_scores(client, "test_collection", "¿Qué es LangChain?", "llama3")
    assert results

    respuesta = "".join(generate_response_with_context("llama3", [doc for doc, _ in results], "¿Qué es LangChain?"))
    assert respuesta
