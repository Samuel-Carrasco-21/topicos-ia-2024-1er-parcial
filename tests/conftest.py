import pytest

from fastapi.testclient import TestClient
from src.copy_main_test import create_app

@pytest.fixture
def app():
  return create_app()

@pytest.fixture
def client(app):
  return TestClient(app)
