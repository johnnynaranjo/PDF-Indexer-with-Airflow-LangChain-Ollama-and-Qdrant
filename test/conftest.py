import pytest
from qdrant_client import QdrantClient

@pytest.fixture(scope="session")
def qdrant_client():
    return QdrantClient(url="http://qdrant:6333")