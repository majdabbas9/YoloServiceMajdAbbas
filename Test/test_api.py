import os
import shutil
import sqlite3
import uuid
import pytest

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app, UPLOAD_DIR, PREDICTED_DIR, DB_PATH, db

client = TestClient(app)

TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_image.jpg")
INVALID_IMAGE_PATH = "invalid_file.txt"
DUMMY_UIDS = ["1","2", "3"]
DUMMY_ORIGINAL_IMAGES = ["original_image1.jpg", "original_image2.jpg", "original_image3.jpg"]
DUMMY_PREDICTED_IMAGES = ["predicted_image1.jpg", "predicted_image2.jpg", "predicted_image3.jpg"]
DUMMY_LABELS = ["cat","dog", "bird"]
DUMMY_SCORES = ["0.95", "0.85", "0.75"]
DUMMY_BOXS = [[10.0, 20.0, 30.0, 40.0], [15.0, 25.0, 35.0, 45.0], [5.0, 10.0, 15.0, 20.0]]


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
    for i in range(len(DUMMY_UIDS)):
        db.save_prediction_session(
            uid=DUMMY_UIDS[i],
            original_image=DUMMY_ORIGINAL_IMAGES[i],
            predicted_image=DUMMY_PREDICTED_IMAGES[i]
        )
        db.save_detection_object(
            c=None,
            prediction_uid=DUMMY_UIDS[i],
            label=DUMMY_LABELS[i],
            score=DUMMY_SCORES[i],
            box=DUMMY_BOXS[i]
        )
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

def test_prediction_by_uid():
    resp = client.get("/prediction/1")
    assert resp.status_code == 200
    assert resp.json().get('uid')  == 1

def test_prediction_by_correct_label():
    resp = client.get("/predictions/label/bird")
    assert resp.status_code == 200
    assert resp.json().get('uid') == 3

def test_prediction_by_wrong_label():
    resp = client.get("/predictions/label/dogg")
    assert resp.status_code == 200
    assert  resp.json() == []

def test_prediction_by_score():
    resp = client.get("/predictions/score/0.8")
    assert resp.status_code == 200
    assert resp.json() == [{'uid': '1'}, {'uid': '2'}]

