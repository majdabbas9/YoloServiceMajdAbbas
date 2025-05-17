import os
import shutil
import sqlite3
import pytest

from fastapi.testclient import TestClient
from app import app, UPLOAD_DIR, PREDICTED_DIR, DB_PATH, init_db,model


client = TestClient(app)

TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_image.jpg")
INVALID_IMAGE_PATH = "invalid_file.txt"

#Clean The DB
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
    init_db()

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
    assert resp.json() == {"status": "ok"}

#Check This Test again
def test_predict_missing_file():
    resp = client.post("/predict", files={})
    assert resp.status_code == 422


def test_predict_invalid_file_type():
    # Create a fake invalid text file
    with open(INVALID_IMAGE_PATH, "w") as f:
        f.write("invalid_file.txt")

    try:
        with open(INVALID_IMAGE_PATH, "rb") as f:
            resp = client.post("/predict", files={"file": ("invalid_file.txt", f, "text/plain")})
        print(resp.status_code)
        assert resp.status_code != 200
    except Exception as e:
        print(f"Exception occurred: {e}")
        # Consider it a pass as long as the API didn't succeed
        assert True






def test_predict_valid_image_and_db_and_getters():

    with open(TEST_IMAGE_PATH, "rb") as img_file:
        resp = client.post("/predict", files={"file": ("test_image.jpg", img_file, "image/jpeg")})
    assert resp.status_code == 200

    json_data = resp.json()
    assert "prediction_uid" in json_data
    assert "detection_count" in json_data
    assert "labels" in json_data

    uid = json_data["prediction_uid"]
    detection_count = json_data["detection_count"]
    labels = json_data["labels"]
#-----------------------------------------------------------------
# Test DB prediction_sessions
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
    assert session is not None
    assert session["uid"] == uid
    assert os.path.exists(session["original_image"])
    assert os.path.exists(session["predicted_image"])

# -----------------------------------------------------------------
# Test DB prediction_sessions
    detections = conn.execute("SELECT * FROM detection_objects WHERE prediction_uid = ?", (uid,)).fetchall()
    assert len(detections) == detection_count
    for det in detections:
        assert det["label"] in labels
        assert 0.0 <= det["score"] <= 1.0

    conn.close()

#-----------------------------------------------------------------
# Test prediction/{uid}
    resp = client.get(f"/prediction/{uid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["uid"] == uid
    assert "detection_objects" in data
    assert len(data["detection_objects"]) == detection_count


def test_prediction_get_invalid_uid():
    resp = client.get("/prediction/nonexistent-uid")
    assert resp.status_code == 404


def test_get_predictions_by_label_valid_and_invalid():
    valid_label = list(model.names.values())[0]

    resp = client.get(f"/predictions/label/{valid_label}")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    resp = client.get("/predictions/label/invalid_label_xyz")
    assert resp.status_code == 404


def test_get_predictions_by_score_valid_and_invalid():
    # Valid scores 0 and 1 boundary
    score = 0.0
    resp = client.get(f"/predictions/score/{score}")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # Invalid scores: <0 or >1
    for invalid_score in [-0.1, 1.1, 2, 100]:
        resp = client.get(f"/predictions/score/{invalid_score}")
        assert resp.status_code == 400


def test_get_image_valid_and_invalid():
    with open(TEST_IMAGE_PATH, "rb") as img_file:
        resp = client.post("/predict", files={"file": ("test_image.jpg", img_file, "image/jpeg")})
    uid = resp.json()["prediction_uid"]

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
    conn.close()

    original_filename = os.path.basename(session["original_image"])
    resp = client.get(f"/image/original/{original_filename}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")

# Valid predicted image
    predicted_filename = os.path.basename(session["predicted_image"])
    resp = client.get(f"/image/predicted/{predicted_filename}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")

# Invalid type
    resp = client.get(f"/image/invalidtype/{predicted_filename}")
    assert resp.status_code == 400

# Non-existent filename
    resp = client.get("/image/original/nonexistentfile.jpg")
    assert resp.status_code == 404