from streamlit_app.main import retrieve_with_scores
from unittest.mock import MagicMock

def test_retrieve_with_scores_returns_data():
    mock_client = MagicMock()
    mock_client.search.return_value = [
        type("Hit", (), {"payload": {"page_content": "doc 1"}, "score": 0.9}),
        type("Hit", (), {"payload": {"page_content": "doc 2"}, "score": 0.8}),
    ]
    result = retrieve_with_scores(mock_client, "test_collection", "pregunta", "llama3")
    assert len(result) == 2
