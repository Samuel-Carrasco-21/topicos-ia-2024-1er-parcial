from fastapi.testclient import TestClient
from src.copy_main_test import app
import pytest

client = TestClient(app)

gun_1 = 'gun1.jpg'
gun_2 = 'gun2.jpg'
gun_3 = 'gun6.jpg'

@pytest.mark.asyncio
def test_model_info(client):
  response = client.get("/model_info")
  assert response.status_code == 200
  assert response.json()["model_name"] == "Gun detector"

@pytest.mark.asyncio
def test_detect_guns(client):
  with open(gun_3, "rb") as img_file:
    files = {"file": img_file}
    response = client.post("/detect_guns", files=files, data={"threshold": 0.5})

  assert response.status_code == 200
  assert "labels" in response.json()
  assert len(response.json()["labels"]) > 0

@pytest.mark.asyncio
def test_annotate_guns(client):
  with open(gun_2, "rb") as img_file:
    files = {"file": img_file}
    response = client.post("/annotate_guns", files=files, data={"threshold": 0.5})

  assert response.status_code == 200
  assert response.headers["content-type"] == "image/jpeg"


@pytest.mark.asyncio
def test_detect_people(client):
  with open(gun_1, "rb") as img_file:
    files = {"file": img_file}
    response = client.post("/detect_people", files=files, data={"threshold": 0.5})

  assert response.status_code == 200
  assert "labels" in response.json()
  assert len(response.json()["labels"]) > 0

@pytest.mark.asyncio
def test_annotate_people(client):
  with open(gun_1, "rb") as img_file:
    files = {"file": img_file}
    response = client.post("/annotate_people", files=files, data={"threshold": 0.5})

  assert response.status_code == 200
  assert response.headers["content-type"] == "image/jpeg"

@pytest.mark.asyncio
def test_guns_endpoint(client):
  with open(gun_2, "rb") as img_file:
    files = {"file": img_file}
    response = client.post("/guns", files=files, data={"threshold": 0.5})

  assert response.status_code == 200
  assert isinstance(response.json(), list)
  assert len(response.json()) > 0

  for gun in response.json():
    assert gun["gun_type"] in ["pistol", "rifle"]

@pytest.mark.asyncio
def test_people_endpoint(client):
  with open(gun_3, "rb") as img_file:
    files = {"file": img_file}
    response = client.post("/people", files=files, data={"threshold": 0.5})

  assert response.status_code == 200
  assert isinstance(response.json(), list)
  assert len(response.json()) > 0

  for person in response.json():
    assert person["person_type"] in ["safe", "danger"]
