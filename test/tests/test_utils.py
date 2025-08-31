from streamlit_app import utils
import pytest

def test_check_connection_success(monkeypatch):
    import requests
    class MockResponse:
        status_code = 200
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())
    assert utils.check_connection("http://mock-url", "mock") == True
