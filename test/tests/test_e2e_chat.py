from streamlit_app.pages.utils import retrieve_with_scores, generate_response_with_context

COLLECTION_NAME = "test_collection"
QUERY = "¿Qué es LangChain?"
MODEL_NAME = "llama3"
EMBEDDING_SIZE = 768

def test_full_chat_pipeline():
    class MockClient:
        def search(self, *args, **kwargs):
            return [type("Hit", (), {"payload": {"page_content": "contenido"}, "score": 0.9})]

    client = MockClient()
    results = retrieve_with_scores(client, COLLECTION_NAME, QUERY, MODEL_NAME, EMBEDDING_SIZE)
    assert results

    respuesta = "".join(generate_response_with_context(MODEL_NAME, [doc for doc, _ in results], QUERY))
    assert respuesta
