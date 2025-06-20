import os
import shutil
import sqlite3
import uuid
import pytest

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app, UPLOAD_DIR, PREDICTED_DIR, DB_PATH, db, model

client = TestClient(app)

TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_image.jpg")
INVALID_IMAGE_PATH = "invalid_file.txt"
DUMMY_UID = "12345678-1234-5678-1234-567812345678"
DUMMY_LABEL = "cat"
DUMMY_SCORE = 0.95
DUMMY_BOX = [10.0, 20.0, 30.0, 40.0]
DUMMY_EXT = ".jpg"
DUMMY_S3_KEY = "dummy_key.jpg"


# Clean the DB and folders before/after all tests
@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(PREDICTED_DIR):
        shutil.rmtree(PREDICTED_DIR)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PREDICTED_DIR, exist_ok=True)

    yield

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(PREDICTED_DIR):
        shutil.rmtree(PREDICTED_DIR)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "message": "Service is running"}


def test_predict_missing_s3_key():
    resp = client.post("/predict")
    assert resp.status_code == 422


def test_predict_invalid_file_type():
    with open(INVALID_IMAGE_PATH, "w") as f:
        f.write("this is not an image")

    try:
        with open(INVALID_IMAGE_PATH, "rb") as f:
            # NOTE: this is not valid since /predict does not accept files
            resp = client.post("/predict", files={"file": ("invalid_file.txt", f, "text/plain")})
            assert resp.status_code != 200
    finally:
        os.remove(INVALID_IMAGE_PATH)

