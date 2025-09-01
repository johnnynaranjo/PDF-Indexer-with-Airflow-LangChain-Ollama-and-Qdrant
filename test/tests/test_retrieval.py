from streamlit_app.pages.utils import retrieve_with_scores
from unittest.mock import MagicMock

COLLECTION_NAME = "test_collection"
QUERY = "¿Qué es LangChain?"
MODEL_NAME = "llama3"
EMBEDDING_SIZE = 768

def test_retrieve_with_scores_returns_data():
    mock_client = MagicMock()
    mock_client.search.return_value = [
        type("Hit", (), {"payload": {"page_content": "doc 1"}, "score": 0.9}),
        type("Hit", (), {"payload": {"page_content": "doc 2"}, "score": 0.8}),
    ]
    result = retrieve_with_scores(mock_client, COLLECTION_NAME, QUERY, MODEL_NAME, EMBEDDING_SIZE)
    assert len(result) == 2
