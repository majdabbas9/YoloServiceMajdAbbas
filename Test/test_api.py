import os
import shutil
import sqlite3
import pytest
from db_for_prediction import DatabaseFactory
from fastapi.testclient import TestClient
from app import app
client = TestClient(app)


#Clean The DB
@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():

    UPLOAD_DIR = "uploads/original"
    PREDICTED_DIR = "uploads/predicted"
    DB_PATH = "predictions.db"
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(PREDICTED_DIR):
        shutil.rmtree(PREDICTED_DIR)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PREDICTED_DIR, exist_ok=True)

    db = DatabaseFactory.create_database("sqlite", db_path=DB_PATH)
    DUMMY_UIDS = ["1", "2", "3"]
    DUMMY_ORIGINAL_IMAGES = ["original_image1.jpg", "original_image2.jpg", "original_image3.jpg"]
    DUMMY_PREDICTED_IMAGES = ["predicted_image1.jpg", "predicted_image2.jpg", "predicted_image3.jpg"]
    DUMMY_LABELS = ["cat", "dog", "bird"]
    DUMMY_SCORES = ["0.95", "0.85", "0.75"]
    DUMMY_BOXS = [[10.0, 20.0, 30.0, 40.0], [15.0, 25.0, 35.0, 45.0], [5.0, 10.0, 15.0, 20.0]]
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

def test_prediction_by_uid():
    resp = client.get("/prediction/1")
    assert resp.status_code == 200
    assert resp.json().get("uid") == "1"

def test_prediction_by_correct_label():
    resp = client.get("/predictions/label/bird")
    assert resp.status_code == 200
    assert resp.json()[0].get("uid") == "3"

def test_prediction_by_wrong_label():
    resp = client.get("/predictions/label/dogg")
    assert resp.status_code == 200
    assert  resp.json() == []

def test_prediction_by_score():
    resp = client.get("/predictions/score/0.8")
    assert resp.status_code == 200
    assert resp.json() == [{'uid': '1'}, {'uid': '2'}] or resp.json() == [{'uid': '2'}, {'uid': '1'}]


#Check This Test again